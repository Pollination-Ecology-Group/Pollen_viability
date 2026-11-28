#!/usr/bin/env python3
"""
pollen_size_analyzer.py
══════════════════════════════════════════════════════════════════════════════
Robust Pollen Grain Stitching, Boundary Filtering, and Size Extraction
for Ploidy & Viability Analysis.

Features:
1. Physical Scale: 0.877242 µm/px (Zeiss slide scanner default calibration).
2. Tile Boundary Stitching: Reconnects grains split across tile edges and
   deduplicates grains detected across overlapping tiles.
3. Whole-Scan Edge Filtering: Grains touching the outermost boundary of the
   entire scan (or exhibiting clipped straight lines) are counted in total &
   viability ratio, but excluded from size measurements.
4. Clump / Overlap Filtering: Grains with low solidity (<0.88), elongated
   aspect ratio (>1.35), or outlier area are flagged as clumps and excluded
   from single-grain size distributions to prevent clumps from masquerading
   as higher ploidies.
5. Separate reporting for Viable, Non-Viable, and Intermediate classes.
6. Outputs CSV table of individual grains + summary statistics per slide.
7. S3 and local directory batch loading with multi-threaded downloads.
══════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import io
import sys
import json
import math
import argparse
import concurrent.futures
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import boto3
from botocore.client import Config

# ── Environment & Defaults ───────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

S3_ENDPOINT    = os.environ.get("S3_ENDPOINT",   "https://s3.cl4.du.cesnet.cz")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "1Y920BKC0SAWPNDE8RD6")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD")
S3_BUCKET      = os.environ.get("S3_BUCKET",     "peg-bucket")
TILE_PREFIX    = os.environ.get("TILE_PREFIX",   "Ostatni/Pollen_viability/tiles_640/")

# Detection cache is versioned per model so stale detections from an older model
# are never silently reused after re-running inference.
DET_CACHE_TAG  = os.environ.get("DET_CACHE_TAG", "model_20261122_r3")
DET_CACHE_DIR  = os.path.join("data", "det_cache", DET_CACHE_TAG)

# Default Zeiss Axio Scan pixel scale for this dataset
DEFAULT_UM_PER_PX = 0.8772419513050967  # µm / px
DEFAULT_TILE_SIZE = 640

CLASS_NAMES = {
    0: "viable",
    1: "non_viable",
    2: "intermediate"
}


def make_s3():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        config=Config(signature_version='s3v4', max_pool_connections=24)
    )


_PLANT_METADATA_CACHE = None

def load_master_plant_metadata() -> dict:
    """Loads master plant metadata table (540 plants) if available locally."""
    global _PLANT_METADATA_CACHE
    if _PLANT_METADATA_CACHE is not None:
        return _PLANT_METADATA_CACHE
    _PLANT_METADATA_CACHE = {}
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metadata", "plant_metadata_540.csv"),
        "metadata/plant_metadata_540.csv",
        "/home/meow/Documents/Antigravity/Pollen_viability/metadata/plant_metadata_540.csv",
        "/home/meow/Documents/Antigravity/Stomata/metadata/stomata_sampling_trait_comparison_by_sample.xlsx"
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                if p.endswith('.csv'):
                    import csv
                    with open(p, 'r', encoding='utf-8') as f:
                        for row in csv.DictReader(f):
                            pid = row.get('plant_ID', '').strip()
                            if pid:
                                _PLANT_METADATA_CACHE[pid] = row
                    break
            except Exception:
                pass
    return _PLANT_METADATA_CACHE


def parse_slide_metadata(slide_name: str) -> dict:
    """
    Extracts metadata from slide filename and cross-references the master 540-plant table.
    Example: '20260724_0001_27-3-B_AA025_n4x_Pollen_viability'
    """
    clean_name = os.path.splitext(os.path.basename(slide_name))[0]
    
    # Ploidy patterns: n4x, s2x, 2x, 4x, 6x, etc. (use lookaround since underscores are \w in regex)
    ploidy_match = re.search(r'(?<![a-zA-Z0-9])(n4x|s2x|2x|4x|6x|8x)(?![a-zA-Z0-9])', clean_name, re.IGNORECASE)
    ploidy = ploidy_match.group(1).lower() if ploidy_match else "unknown"
    
    # Accession pattern: AA followed by digits
    acc_match = re.search(r'(?<![a-zA-Z0-9])(AA\d+)(?![a-zA-Z0-9])', clean_name, re.IGNORECASE)
    accession = acc_match.group(1).upper() if acc_match else "unknown"
    
    # Plant/Sample pattern: e.g. 27-3-B, 28-8-E
    sample_match = re.search(r'(?<![a-zA-Z0-9])(\d+-\d+-[A-Za-z])(?![a-zA-Z0-9])', clean_name)
    sample_id = sample_match.group(1) if sample_match else "unknown"
    
    # Replicate: 'rep' present in filename
    is_rep = bool(re.search(r'(?<![a-zA-Z0-9])rep(?![a-zA-Z0-9])', clean_name, re.IGNORECASE))
    
    CYTOTYPE_MAP = {
        "2x": "nat_2x",
        "s2x": "ory_2x",
        "n4x": "neo_4x",
        "4x": "nat_4x"
    }
    canonical_cytotype = CYTOTYPE_MAP.get(ploidy, ploidy)
    
    meta = {
        "slide_name": clean_name,
        "sample_id": sample_id,
        "accession": accession,
        "ploidy": ploidy,
        "replicate": "replicate" if is_rep else "primary",
        "cultivation_number": "unknown",
        "population": accession,
        "cytotype": canonical_cytotype,
        "numeric_ploidy": 2 if "2" in ploidy else (4 if "4" in ploidy else 0),
        "cultivation_run": "unknown"
    }

    # Cross-reference with master plant metadata if available
    plant_db = load_master_plant_metadata()
    if sample_id in plant_db:
        pdata = plant_db[sample_id]
        meta["cultivation_number"] = pdata.get("cultivation_number", "unknown")
        meta["population"] = pdata.get("population", accession)
        meta["cytotype"] = pdata.get("cytotype", canonical_cytotype)
        meta["numeric_ploidy"] = int(pdata.get("ploidy", meta["numeric_ploidy"]))
        meta["cultivation_run"] = pdata.get("cultivation_run", "unknown")
        
    return meta



def parse_tile_offset(tile_filename: str) -> tuple:
    """
    Extracts (y_offset, x_offset) from tile filename.
    Pattern: *_tile_{y}_{x}.*
    """
    base = os.path.basename(tile_filename)
    match = re.search(r'_tile_(\d+)_(\d+)', base)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


class PollenParticle:
    def __init__(self, global_box, global_poly, conf, cls_id, tile_offset):
        self.box = np.array(global_box, dtype=float)  # [x1, y1, x2, y2]
        self.poly = np.array(global_poly, dtype=np.float32)  # [[x, y], ...]
        self.conf = float(conf)
        self.cls_id = int(cls_id)
        self.cls_name = CLASS_NAMES.get(self.cls_id, f"class_{self.cls_id}")
        self.tile_offset = tile_offset  # (y, x)
        
        # Computed morphological metrics
        self.area_px = 0.0
        self.area_um2 = 0.0
        self.diameter_um = 0.0
        self.major_axis_um = 0.0
        self.minor_axis_um = 0.0
        self.aspect_ratio = 1.0
        self.solidity = 1.0
        
        # Validation flags
        self.is_outer_edge = False
        self.is_clump = False
        self.valid_for_size = True
        
    @property
    def center(self):
        return (float((self.box[0] + self.box[2]) / 2.0),
                float((self.box[1] + self.box[3]) / 2.0))


def compute_particle_metrics(p: PollenParticle, scale_um: float):
    """Computes exact geometric, ellipse, and area properties for a particle."""
    if len(p.poly) < 3:
        w = max(1.0, p.box[2] - p.box[0])
        h = max(1.0, p.box[3] - p.box[1])
        p.area_px = float(w * h * (math.pi / 4.0))  # ellipse approx
        p.aspect_ratio = float(max(w, h) / max(1.0, min(w, h)))
        p.solidity = 0.85
        major_px = max(w, h)
        minor_px = min(w, h)
    else:
        # Exact contour area
        p.area_px = float(cv2.contourArea(p.poly))
        if p.area_px <= 0:
            w = max(1.0, p.box[2] - p.box[0])
            h = max(1.0, p.box[3] - p.box[1])
            p.area_px = float(w * h * (math.pi / 4.0))
            
        # Convex hull & solidity
        hull = cv2.convexHull(p.poly)
        hull_area = float(cv2.contourArea(hull))
        p.solidity = float(p.area_px / max(1e-5, hull_area))
        
        # Fit ellipse / Min Area Rect
        if len(p.poly) >= 5:
            try:
                ellipse = cv2.fitEllipse(p.poly)
                (cx, cy), (ax1, ax2), angle = ellipse
                major_px = max(ax1, ax2)
                minor_px = min(ax1, ax2)
            except Exception:
                rect = cv2.minAreaRect(p.poly)
                major_px = max(rect[1][0], rect[1][1])
                minor_px = min(rect[1][0], rect[1][1])
        else:
            rect = cv2.minAreaRect(p.poly)
            major_px = max(rect[1][0], rect[1][1])
            minor_px = min(rect[1][0], rect[1][1])
            
        p.aspect_ratio = float(major_px / max(1e-3, minor_px))

    # Scale conversions
    p.area_um2 = p.area_px * (scale_um ** 2)
    p.diameter_um = 2.0 * math.sqrt(p.area_um2 / math.pi)
    p.major_axis_um = major_px * scale_um
    p.minor_axis_um = minor_px * scale_um


def box_iou(b1, b2):
    """Computes IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def stitch_and_deduplicate(particles, tile_size=DEFAULT_TILE_SIZE, iou_thresh=0.35, cell_size=320):
    """
    Merges duplicate detections in overlap regions and stitches grains
    split across tile boundaries into single consolidated particles.
    Uses spatial hash grid for O(N) performance.
    """
    if not particles:
        return []
        
    particles = sorted(particles, key=lambda p: p.conf, reverse=True)
    
    # Build spatial grid
    grid = defaultdict(list)
    for idx, p in enumerate(particles):
        cx, cy = p.center
        gx, gy = int(cx // cell_size), int(cy // cell_size)
        grid[(gx, gy)].append(idx)
        
    kept = []
    merged_indices = set()
    
    for i, p1 in enumerate(particles):
        if i in merged_indices:
            continue
            
        merged_poly = p1.poly.copy()
        merged_box = p1.box.copy()
        best_conf = p1.conf
        cls_id = p1.cls_id
        
        # Check neighboring cells
        cx, cy = p1.center
        gx, gy = int(cx // cell_size), int(cy // cell_size)
        
        candidate_indices = set()
        for dx_grid in (-1, 0, 1):
            for dy_grid in (-1, 0, 1):
                cell_key = (gx + dx_grid, gy + dy_grid)
                if cell_key in grid:
                    candidate_indices.update(grid[cell_key])
                    
        # Filter candidate indices: only j > i that haven't been merged
        sorted_candidates = sorted([j for j in candidate_indices if j > i and j not in merged_indices])
        
        for j in sorted_candidates:
            if j in merged_indices:
                continue
            p2 = particles[j]
            
            # Must be of the same class to stitch
            if p1.cls_id != p2.cls_id:
                continue
                
            iou = box_iou(merged_box, p2.box)
            
            # Check if boxes touch or overlap along a shared tile boundary
            dx = min(abs(merged_box[2] - p2.box[0]), abs(p2.box[2] - merged_box[0]))
            dy = min(abs(merged_box[3] - p2.box[1]), abs(p2.box[3] - merged_box[1]))
            
            y_overlap = max(0.0, min(merged_box[3], p2.box[3]) - max(merged_box[1], p2.box[1]))
            x_overlap = max(0.0, min(merged_box[2], p2.box[2]) - max(merged_box[0], p2.box[0]))
            
            touching_x = (dx <= 6.0 and y_overlap > 8.0)
            touching_y = (dy <= 6.0 and x_overlap > 8.0)
            
            is_same_grain = (iou > iou_thresh) or touching_x or touching_y
            
            if is_same_grain:
                merged_box[0] = min(merged_box[0], p2.box[0])
                merged_box[1] = min(merged_box[1], p2.box[1])
                merged_box[2] = max(merged_box[2], p2.box[2])
                merged_box[3] = max(merged_box[3], p2.box[3])
                
                if len(merged_poly) > 0 and len(p2.poly) > 0:
                    combined_pts = np.vstack([merged_poly, p2.poly])
                    hull = cv2.convexHull(combined_pts.astype(np.float32)).squeeze()
                    if hull.ndim == 1:
                        hull = hull.reshape(-1, 2)
                    merged_poly = hull
                        
                best_conf = max(best_conf, p2.conf)
                merged_indices.add(j)
                
        new_p = PollenParticle(merged_box, merged_poly, best_conf, cls_id, p1.tile_offset)
        kept.append(new_p)
        
    return kept


def has_clipped_boundary_edge(poly, total_w, total_h, edge_margin=6.0, min_chord=12.0):
    """
    Checks if a particle's polygon touches the outer scan boundary or has
    a characteristic flat/straight cut line along an image boundary.
    """
    if len(poly) < 3:
        return False
        
    pts = np.atleast_2d(poly)
    x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    
    # Direct touch of slide outer boundaries
    touches_outer = (x_min <= edge_margin or y_min <= edge_margin or
                     x_max >= total_w - edge_margin or y_max >= total_h - edge_margin)
                     
    if not touches_outer:
        return False
        
    # Check for straight edge along boundary
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        chord = math.hypot(dx, dy)
        
        if chord >= min_chord:
            # Horizontal cut line near top/bottom
            if dy <= 1.5 and (p1[1] <= edge_margin or p1[1] >= total_h - edge_margin):
                return True
            # Vertical cut line near left/right
            if dx <= 1.5 and (p1[0] <= edge_margin or p1[0] >= total_w - edge_margin):
                return True
                
    return touches_outer


def filter_whole_image_edges(particles, total_w, total_h, edge_margin=6.0):
    """
    Flags grains that touch or are clipped by the outer boundary of the whole scan.
    These grains are retained in total counts and viability ratios,
    but marked invalid for size distributions.
    """
    for p in particles:
        if has_clipped_boundary_edge(p.poly, total_w, total_h, edge_margin=edge_margin):
            p.is_outer_edge = True
            p.valid_for_size = False


def filter_clumps(particles, min_solidity=0.88, max_aspect_ratio=1.35, max_area_mult=2.5):
    """
    Flags clumped or overlapping grains based on solidity, elongation, and outlier area.
    """
    viable_areas = [p.area_px for p in particles if p.cls_id == 0 and not p.is_outer_edge and p.area_px > 0]
    med_area = np.median(viable_areas) if viable_areas else 2000.0
    
    for p in particles:
        if p.is_outer_edge:
            continue
            
        is_concave_clump = (p.solidity < min_solidity)
        is_elongated_clump = (p.aspect_ratio > max_aspect_ratio)
        is_giant_blob = (p.area_px > max_area_mult * med_area and p.cls_id == 0)
        
        if is_concave_clump or is_elongated_clump or is_giant_blob:
            p.is_clump = True
            p.valid_for_size = False


def analyze_slide_detections(det_data_list, slide_name, scale_um=DEFAULT_UM_PER_PX, conf_thresh=0.55):
    """
    Processes all tile detections for one slide end-to-end.
    """
    meta = parse_slide_metadata(slide_name)
    raw_particles = []
    
    max_x = 0
    max_y = 0
    
    for item in det_data_list:
        ty = item["y_offset"]
        tx = item["x_offset"]
        max_x = max(max_x, tx + DEFAULT_TILE_SIZE)
        max_y = max(max_y, ty + DEFAULT_TILE_SIZE)
        
        boxes = item.get("boxes", [])
        masks_xyn = item.get("masks_xyn", [])
        
        for i, b in enumerate(boxes):
            lx1, ly1, lx2, ly2, conf, cls_id = b
            if conf < conf_thresh:
                continue
                
            gx1, gy1 = lx1 + tx, ly1 + ty
            gx2, gy2 = lx2 + tx, ly2 + ty
            
            if i < len(masks_xyn) and len(masks_xyn[i]) > 0:
                poly_local = np.array(masks_xyn[i], dtype=float) * DEFAULT_TILE_SIZE
                poly_global = poly_local.copy()
                poly_global[:, 0] += tx
                poly_global[:, 1] += ty
            else:
                poly_global = np.array([
                    [gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]
                ])
                
            p = PollenParticle([gx1, gy1, gx2, gy2], poly_global, conf, cls_id, (ty, tx))
            raw_particles.append(p)
            
    # Step 1: Stitch tile boundaries & deduplicate overlaps
    stitched_particles = stitch_and_deduplicate(raw_particles)
    
    # Step 2: Compute morphological metrics on stitched particles
    for p in stitched_particles:
        compute_particle_metrics(p, scale_um)
        
    # Step 3: Flag whole-image outer edge grains
    filter_whole_image_edges(stitched_particles, max_x, max_y)
    
    # Step 4: Flag clumps / overlapping grains
    filter_clumps(stitched_particles)
    
    # Compile per-class statistics
    summary = compile_slide_summary(stitched_particles, meta)
    
    return {
        "metadata": meta,
        "summary": summary,
        "particles": stitched_particles,
        "slide_dims": (max_y, max_x)
    }


def compile_slide_summary(particles, meta):
    """Compiles counts and size statistics separately for viable, non-viable, and intermediate."""
    counts_total = defaultdict(int)
    counts_valid_size = defaultdict(int)
    counts_outer_edge = defaultdict(int)
    counts_clump = defaultdict(int)
    
    sizes_area_um2 = defaultdict(list)
    sizes_diameter_um = defaultdict(list)
    sizes_aspect_ratio = defaultdict(list)
    
    for p in particles:
        c = p.cls_name
        counts_total[c] += 1
        
        if p.is_outer_edge:
            counts_outer_edge[c] += 1
        if p.is_clump:
            counts_clump[c] += 1
            
        if p.valid_for_size:
            counts_valid_size[c] += 1
            sizes_area_um2[c].append(p.area_um2)
            sizes_diameter_um[c].append(p.diameter_um)
            sizes_aspect_ratio[c].append(p.aspect_ratio)
            
    total_grains = sum(counts_total.values())
    viable_count = counts_total["viable"]
    non_viable_count = counts_total["non_viable"]
    intermediate_count = counts_total["intermediate"]
    
    # Viability rates
    viability_pct = (viable_count / total_grains * 100.0) if total_grains > 0 else 0.0
    denom_viab_nv = viable_count + non_viable_count
    viab_ratio_binary = (viable_count / denom_viab_nv * 100.0) if denom_viab_nv > 0 else 0.0
    
    stats_by_class = {}
    for c in ["viable", "non_viable", "intermediate"]:
        areas = sizes_area_um2[c]
        diams = sizes_diameter_um[c]
        if areas:
            stats_by_class[c] = {
                "count_total": counts_total[c],
                "count_valid_size": counts_valid_size[c],
                "count_outer_edge": counts_outer_edge[c],
                "count_clump": counts_clump[c],
                "mean_area_um2": float(np.mean(areas)),
                "std_area_um2": float(np.std(areas)),
                "median_area_um2": float(np.median(areas)),
                "q25_area_um2": float(np.percentile(areas, 25)),
                "q75_area_um2": float(np.percentile(areas, 75)),
                "mean_diameter_um": float(np.mean(diams)),
                "std_diameter_um": float(np.std(diams)),
                "median_diameter_um": float(np.median(diams)),
                "q25_diameter_um": float(np.percentile(diams, 25)),
                "q75_diameter_um": float(np.percentile(diams, 75)),
            }
        else:
            stats_by_class[c] = {
                "count_total": counts_total[c],
                "count_valid_size": 0,
                "count_outer_edge": counts_outer_edge[c],
                "count_clump": counts_clump[c],
                "mean_area_um2": 0.0,
                "std_area_um2": 0.0,
                "median_area_um2": 0.0,
                "q25_area_um2": 0.0,
                "q75_area_um2": 0.0,
                "mean_diameter_um": 0.0,
                "std_diameter_um": 0.0,
                "median_diameter_um": 0.0,
                "q25_diameter_um": 0.0,
                "q75_diameter_um": 0.0,
            }
            
    return {
        "metadata": meta,
        "total_grains": total_grains,
        "viable_count": viable_count,
        "non_viable_count": non_viable_count,
        "intermediate_count": intermediate_count,
        "viability_pct": viability_pct,
        "viab_ratio_binary": viab_ratio_binary,
        "classes": stats_by_class
    }


def export_particles_csv(particles, meta, output_csv_path, append=False):
    """Exports individual grain measurements to CSV for downstream R/Python mixed modeling."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    mode = 'a' if append and os.path.exists(output_csv_path) else 'w'
    write_header = (mode == 'w') or (os.path.getsize(output_csv_path) == 0 if os.path.exists(output_csv_path) else True)
    
    with open(output_csv_path, mode) as f:
        if write_header:
            headers = [
                "slide_name", "plant_id", "cultivation_number", "population", "accession",
                "cytotype", "ploidy_str", "numeric_ploidy", "cultivation_run", "replicate",
                "grain_id", "class_id", "class_name", "conf",
                "center_x_px", "center_y_px",
                "area_px", "area_um2", "diameter_um",
                "major_axis_um", "minor_axis_um", "aspect_ratio", "solidity",
                "is_outer_edge", "is_clump", "valid_for_size"
            ]
            f.write(",".join(headers) + "\n")
            
        for idx, p in enumerate(particles):
            row = [
                meta["slide_name"], meta["sample_id"], meta.get("cultivation_number", "unknown"),
                meta.get("population", meta["accession"]), meta["accession"],
                meta.get("cytotype", meta["ploidy"]), meta["ploidy"],
                str(meta.get("numeric_ploidy", "")), meta.get("cultivation_run", "unknown"),
                meta["replicate"],
                str(idx + 1), str(p.cls_id), p.cls_name, f"{p.conf:.4f}",
                f"{p.center[0]:.1f}", f"{p.center[1]:.1f}",
                f"{p.area_px:.1f}", f"{p.area_um2:.2f}", f"{p.diameter_um:.2f}",
                f"{p.major_axis_um:.2f}", f"{p.minor_axis_um:.2f}",
                f"{p.aspect_ratio:.3f}", f"{p.solidity:.3f}",
                str(int(p.is_outer_edge)), str(int(p.is_clump)),
                str(int(p.valid_for_size))
            ]
            f.write(",".join(row) + "\n")


def export_slide_summaries_csv(summaries: list, output_csv_path: str):
    """Writes a consolidated slide-level summary CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    headers = [
        "slide_name", "plant_id", "cultivation_number", "population", "accession",
        "cytotype", "ploidy_str", "numeric_ploidy", "cultivation_run", "replicate",
        "total_grains", "viable_count", "non_viable_count", "intermediate_count",
        "viability_pct", "viab_ratio_binary",
        "viable_valid_size", "viable_mean_area_um2", "viable_std_area_um2", "viable_median_area_um2",
        "viable_q25_area_um2", "viable_q75_area_um2",
        "viable_mean_diam_um", "viable_std_diam_um", "viable_median_diam_um",
        "non_viable_valid_size", "non_viable_mean_area_um2", "non_viable_std_area_um2", "non_viable_median_area_um2",
        "non_viable_mean_diam_um", "non_viable_median_diam_um"
    ]
    with open(output_csv_path, 'w') as f:
        f.write(",".join(headers) + "\n")
        for s in summaries:
            meta = s["metadata"]
            v = s["classes"].get("viable", {})
            nv = s["classes"].get("non_viable", {})
            row = [
                meta["slide_name"], meta["sample_id"], meta.get("cultivation_number", "unknown"),
                meta.get("population", meta["accession"]), meta["accession"],
                meta.get("cytotype", meta["ploidy"]), meta["ploidy"],
                str(meta.get("numeric_ploidy", "")), meta.get("cultivation_run", "unknown"),
                meta["replicate"],
                str(s["total_grains"]), str(s["viable_count"]), str(s["non_viable_count"]), str(s["intermediate_count"]),
                f"{s['viability_pct']:.2f}", f"{s['viab_ratio_binary']:.2f}",
                str(v.get("count_valid_size", 0)),
                f"{v.get('mean_area_um2', 0):.2f}", f"{v.get('std_area_um2', 0):.2f}", f"{v.get('median_area_um2', 0):.2f}",
                f"{v.get('q25_area_um2', 0):.2f}", f"{v.get('q75_area_um2', 0):.2f}",
                f"{v.get('mean_diameter_um', 0):.2f}", f"{v.get('std_diameter_um', 0):.2f}", f"{v.get('median_diameter_um', 0):.2f}",
                str(nv.get("count_valid_size", 0)),
                f"{nv.get('mean_area_um2', 0):.2f}", f"{nv.get('std_area_um2', 0):.2f}", f"{nv.get('median_area_um2', 0):.2f}",
                f"{nv.get('mean_diameter_um', 0):.2f}", f"{nv.get('median_diameter_um', 0):.2f}"
            ]
            f.write(",".join(row) + "\n")
    print(f"📊 Saved summary table for {len(summaries)} slides → {output_csv_path}")


# ── S3 & Local Data Loaders ──────────────────────────────────────────────────

def fetch_s3_json(s3, bucket, key):
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(resp['Body'].read().decode('utf-8'))
        ty, tx = parse_tile_offset(key)
        return {
            "tile_name": os.path.basename(key),
            "y_offset": ty,
            "x_offset": tx,
            "boxes": data.get("boxes", []),
            "masks_xyn": data.get("masks_xyn", [])
        }
    except Exception as e:
        print(f"⚠️ Failed fetching S3 det {key}: {e}", file=sys.stderr)
        return None


def fetch_slide_detections_from_s3(s3, bucket, slide_folder_prefix, max_workers=24, cache_dir=None):
    """Downloads and parses all _det.json files for a given slide folder on S3, caching locally."""
    slide_name = slide_folder_prefix.rstrip('/').split('/')[-1]
    slide_cache = Path(cache_dir or DET_CACHE_DIR) / slide_name
    # Check if cache is already populated
    use_cache = os.environ.get("NO_DET_CACHE", "0") != "1"
    if use_cache:
        slide_cache.mkdir(parents=True, exist_ok=True)
        cached_files = list(slide_cache.glob("*_det.json"))
        if cached_files and len(cached_files) > 50:
            print(f"   ⚡ Using {len(cached_files)} cached detection files from {slide_cache}")
            results = []
            for cf in cached_files:
                try:
                    with open(cf, 'r') as f:
                        data = json.load(f)
                    ty, tx = parse_tile_offset(cf.name)
                    results.append({
                        "tile_name": cf.name,
                        "y_offset": ty,
                        "x_offset": tx,
                        "boxes": data.get("boxes", []),
                        "masks_xyn": data.get("masks_xyn", [])
                    })
                except Exception:
                    pass
            return results

    paginator = s3.get_paginator('list_objects_v2')
    jpg_stems = set()
    det_cands = []
    for page in paginator.paginate(Bucket=bucket, Prefix=slide_folder_prefix):
        for o in page.get('Contents', []):
            k = o['Key']
            if k.lower().endswith('.jpg'):
                jpg_stems.add(k[:-4])
            elif k.endswith('_det.json'):
                det_cands.append(k)

    det_keys = [k for k in det_cands if k[:-9] in jpg_stems] if jpg_stems else det_cands
    if not det_keys:
        return []
        
    def fetch_and_cache(k):
        try:
            resp = s3.get_object(Bucket=bucket, Key=k)
            raw = resp['Body'].read()
            local_name = os.path.basename(k)
            if use_cache:
                (slide_cache / local_name).write_bytes(raw)
            data = json.loads(raw.decode('utf-8'))
            ty, tx = parse_tile_offset(k)
            return {
                "tile_name": local_name,
                "y_offset": ty,
                "x_offset": tx,
                "boxes": data.get("boxes", []),
                "masks_xyn": data.get("masks_xyn", [])
            }
        except Exception as e:
            return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch_and_cache, k) for k in det_keys]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res is not None:
                results.append(res)
    return results


def load_slide_detections_from_local(folder_or_file):
    """Loads tile detections from a local directory or detections.json file."""
    p = Path(folder_or_file)
    results = []
    if p.is_file() and p.name.endswith('.json'):
        with open(p, 'r') as f:
            data = json.load(f)
        if "tiles" in data:
            for t in data["tiles"]:
                ty = t.get("tile_y", 0)
                tx = t.get("tile_x", 0)
                boxes = []
                masks = []
                for d in t.get("detections", []):
                    b = d["bbox_tile"]
                    boxes.append([b[0], b[1], b[2], b[3], d.get("confidence", 1.0), d.get("class", 0)])
                    if "mask_xyn" in d:
                        masks.append(d["mask_xyn"])
                results.append({
                    "tile_name": f"tile_{ty}_{tx}",
                    "y_offset": ty,
                    "x_offset": tx,
                    "boxes": boxes,
                    "masks_xyn": masks
                })
        return results
        
    for jf in p.glob("*_det.json"):
        with open(jf, 'r') as f:
            data = json.load(f)
        ty, tx = parse_tile_offset(jf.name)
        results.append({
            "tile_name": jf.name,
            "y_offset": ty,
            "x_offset": tx,
            "boxes": data.get("boxes", []),
            "masks_xyn": data.get("masks_xyn", [])
        })
    return results


# ── CLI & Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pollen Size & Ploidy Extraction Analyzer")
    parser.add_argument("--slide", type=str, default=None,
                        help="Specific slide folder name or pattern on S3 to process")
    parser.add_argument("--all", action="store_true",
                        help="Process all slide folders on S3 that have _det.json files")
    parser.add_argument("--local-dir", type=str, default=None,
                        help="Path to local directory or detections.json to analyze")
    parser.add_argument("--conf", type=float, default=0.50,
                        help="Confidence threshold for pollen grains (default: 0.50)")
    parser.add_argument("--scale", type=float, default=DEFAULT_UM_PER_PX,
                        help="Physical scale in µm/px (default: 0.877242)")
    parser.add_argument("--out-dir", type=str, default="output/size_analysis",
                        help="Directory to write output CSVs and reports")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent slide processing workers (default: 1)")
    parser.add_argument("--upload-s3", action="store_true",
                        help="Upload output CSVs to S3 after completion")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_grains_csv = str(out_dir / "pollen_grain_measurements.csv")
    slide_summary_csv = str(out_dir / "pollen_slide_summaries.csv")

    summaries = []

    # 1. Local directory mode
    if args.local_dir:
        slide_name = Path(args.local_dir).name
        print(f"🔬 Analyzing local detections: {args.local_dir}")
        det_data = load_slide_detections_from_local(args.local_dir)
        if not det_data:
            sys.exit(f"❌ No detections found in {args.local_dir}")
        res = analyze_slide_detections(det_data, slide_name, scale_um=args.scale, conf_thresh=args.conf)
        summaries.append(res["summary"])
        slide_csv = str(out_dir / f"{slide_name}_grains.csv")
        export_particles_csv(res["particles"], res["metadata"], slide_csv)
        export_particles_csv(res["particles"], res["metadata"], all_grains_csv, append=False)
        export_slide_summaries_csv(summaries, slide_summary_csv)
        print(f"✅ Finished {slide_name}: {res['summary']['total_grains']} grains (Viability: {res['summary']['viability_pct']:.1f}%)")
        return

    # 2. S3 mode
    s3 = make_s3()
    
    # Discover slide folders (MUST paginate: >1,000 slide prefixes exceed single S3 page!)
    print("📋 Discovering slide folders on S3...")
    slide_prefixes = []
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=S3_BUCKET, Prefix=TILE_PREFIX, Delimiter='/'):
        for p in page.get('CommonPrefixes', []):
            slide_prefixes.append(p['Prefix'])
    slide_prefixes = sorted(set(slide_prefixes))
    print(f"   Found {len(slide_prefixes)} slide folders under s3://{S3_BUCKET}/{TILE_PREFIX}")
    
    if args.slide:
        target_prefixes = [p for p in slide_prefixes if args.slide.lower() in p.lower()]
        if not target_prefixes:
            sys.exit(f"❌ No slide folder found on S3 matching '{args.slide}'")
    elif args.all:
        target_prefixes = slide_prefixes
    else:
        # Default: pick first matching slide or show usage
        if slide_prefixes:
            target_prefixes = [slide_prefixes[0]]
            print(f"ℹ️  No slide specified, running on first slide: {target_prefixes[0]}")
        else:
            sys.exit("❌ No tile folders found on S3.")

    # Remove existing all-grains CSV before multi-slide run
    if os.path.exists(all_grains_csv):
        os.remove(all_grains_csv)

    print(f"🚀 Processing {len(target_prefixes)} slide(s) (workers={args.workers})...")

    def process_one_slide(item):
        idx, spfx = item
        slide_name = spfx.rstrip('/').split('/')[-1]
        local_s3 = make_s3()
        det_data = fetch_slide_detections_from_s3(local_s3, S3_BUCKET, spfx)
        if not det_data:
            return None
        res = analyze_slide_detections(det_data, slide_name, scale_um=args.scale, conf_thresh=args.conf)
        slide_csv = str(out_dir / f"{slide_name}_grains.csv")
        export_particles_csv(res["particles"], res["metadata"], slide_csv)
        s = res["summary"]
        print(f"  [{idx+1}/{len(target_prefixes)}] ✅ {slide_name}: {s['total_grains']} grains (Viability: {s['viability_pct']:.1f}%)", flush=True)
        return res["summary"]

    if args.workers > 1:
        items = list(enumerate(target_prefixes))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one_slide, it) for it in items]
            for f in concurrent.futures.as_completed(futures):
                try:
                    sm = f.result()
                    if sm:
                        summaries.append(sm)
                except Exception as e:
                    print(f"  ⚠️ Slide processing error: {e}", flush=True)
    else:
        for idx, spfx in enumerate(target_prefixes):
            sm = process_one_slide((idx, spfx))
            if sm:
                summaries.append(sm)

    if summaries:
        # Sort summaries by slide_name for consistent order
        summaries.sort(key=lambda s: s["metadata"]["slide_name"])
        export_slide_summaries_csv(summaries, slide_summary_csv)

        # Consolidate all slide grain CSVs into pollen_grain_measurements.csv
        print("📑 Consolidating individual slide grain CSVs into master table...")
        header_written = False
        with open(all_grains_csv, 'w') as out_f:
            for s in summaries:
                s_name = s["metadata"]["slide_name"]
                s_csv = out_dir / f"{s_name}_grains.csv"
                if s_csv.exists():
                    with open(s_csv, 'r') as in_f:
                        lines = in_f.readlines()
                        if not lines:
                            continue
                        if not header_written:
                            out_f.write(lines[0])
                            header_written = True
                        out_f.writelines(lines[1:])

        print(f"\n🎉 Batch analysis complete! Results written to: {out_dir}/")
        print(f"   Slides analyzed: {len(summaries)}")
        print(f"   Summary CSV: {slide_summary_csv}")
        print(f"   Grain measurements CSV: {all_grains_csv}")

        if args.upload_s3:
            print("\n☁️ Uploading results to S3...")
            s3_client = make_s3()
            s3_pfx = "Ostatni/Pollen_viability/size_analysis/"
            for fn in [slide_summary_csv, all_grains_csv]:
                if os.path.exists(fn):
                    k = s3_pfx + os.path.basename(fn)
                    print(f"   Uploading {os.path.basename(fn)} → s3://{S3_BUCKET}/{k}...")
                    s3_client.upload_file(fn, S3_BUCKET, k)
            print("   ✅ Upload to S3 complete.")


if __name__ == "__main__":
    main()
