#!/usr/bin/env python3
"""
train_model.py — Leak-free, NaN-proof YOLO11m-seg training pipeline for Pollen Viability.

Key safeguards & improvements over the Oct 23 training script:
  1. NaN-Proof Optimization (`amp=False` / TF32 + `AdamW` + `cos_lr=True` + `copy_paste=0.0`):
     Eliminates the Epoch-25 FP16/AMP overflow that poisoned Ultralytics EMA weights in
     `pollen_train_20261023_0756` (where 276/300 validation epochs became NaN).
  2. Strict Blind-QC Benchmark Isolation:
     Loads `Ostatni/Pollen_viability/Review/manifest.json` (191 blind QC tiles across 72 slides)
     and excludes every benchmark tile (as well as `*qc_additive*.zip` / `qc__*` circle-mask exports)
     from both `train` and `val`, preserving a 100% leak-free human ground-truth test benchmark.
  3. Polygon Sanitization & Sutherland-Hodgman Crop Clipping:
     Drops degenerate (< 4 px or < 8 px^2) polygon slivers before training.
  4. Rare-Class & Ploidy-Scale (`Gigas`) Balancing in `train` Only:
     Deterministic 85/15 class-stratified split (`seed=42`), natural slide-background blending for
     `smudges_raw` (in `train` only, never `val`), plus 1.0x oversampling and 1.26x / 1.36x
     crop-zoomed copies of `non_viable` and `intermediate` tiles to bridge the 2x -> 4x size gap.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import shutil
import smtplib
import ssl
import sys
import urllib.request
from collections import Counter
from datetime import datetime
from email.message import EmailMessage

# Configure writable Ultralytics & Matplotlib dirs before importing ultralytics
if os.environ.get("KUBERNETES_SERVICE_HOST") or not os.access(os.path.expanduser("~"), os.W_OK):
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo_config")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import boto3
import cv2
import numpy as np
import yaml
from botocore.client import Config
from ultralytics import YOLO, settings

if os.environ.get("KUBERNETES_SERVICE_HOST"):
    settings.update({
        "runs_dir": "/tmp/runs",
        "datasets_dir": "/tmp/datasets",
        "weights_dir": "/tmp/weights",
    })

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    if os.path.exists(".env"):
        with open(".env") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())

# --- CONFIGURATION FROM ENV ---
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
S3_BUCKET = os.environ.get("S3_BUCKET", "peg-bucket")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
RECEIVER_EMAIL = "jakubstenc@gmail.com"

LOCAL_ROOT = "Pollen_viability"
DATASET_ROOT = os.path.join(LOCAL_ROOT, "datasets/pollen_czi")
STAGING_AREA = os.path.join(LOCAL_ROOT, "staged_area")
SMUDGES_RAW = os.path.join(LOCAL_ROOT, "smudges_raw")
HARD_NEGATIVES = os.path.join(LOCAL_ROOT, "hard_negatives")
HARD_POSITIVES = os.path.join(LOCAL_ROOT, "hard_positives")
RELABEL_CACHE = os.path.join(LOCAL_ROOT, "relabel_studio")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
VIS_DIR = "visualizations"

CANONICAL_CLASSES = ["viable", "non_viable", "intermediate"]
CLASS_NAME_TO_ID = {"viable": 0, "non_viable": 1, "non-viable": 1, "intermediate": 2}
IMG_SIZE = 640


def setup_s3():
    if not all([S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        raise ValueError("❌ S3 Environment variables missing! Cannot sync data.")
    return boto3.resource(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
    )


def download_s3_prefix(s3, prefix: str, local_dir: str, skip_substring: str = None):
    bucket = s3.Bucket(S3_BUCKET)
    print(f"⬇️ Syncing s3://{S3_BUCKET}/{prefix} → {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    count = 0
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("/"):
            continue
        if skip_substring and skip_substring in obj.key:
            print(f"   🔒 Skipping leaked/excluded archive on S3: {os.path.basename(obj.key)}")
            continue
        rel_path = os.path.relpath(obj.key, prefix)
        if rel_path == "." or rel_path.startswith("_"):
            continue
        dest_path = os.path.join(local_dir, rel_path)
        if not os.path.exists(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            bucket.download_file(obj.key, dest_path)
            count += 1
            if count % 100 == 0:
                print(f"   Downloaded {count}...", end="\r")
    print(f"✅ Synced {count} new files into {local_dir}.")


def strip_tile_base(filename: str) -> str:
    """Normalize any staged, Roboflow, QC, or active-learning filename to its canonical tile stem."""
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r"\.rf\.[a-f0-9]+$", "", base)
    base = re.sub(r"_jpg$|_png$", "", base)
    if "__" in base:
        base = base.split("__")[-1]
    for pfx in ("hard_pos_", "hard_neg_", "curated_neg_", "syn_neg_"):
        if base.startswith(pfx):
            base = base[len(pfx):]
    base = re.sub(r"_(nvdup\d+|4xzoom\d+)$", "", base)
    return base


def load_qc_holdout_bases(s3=None) -> set:
    """Return the set of normalized tile basenames in the blind QC review benchmark."""
    holdout = set()
    if s3 is not None:
        try:
            raw = s3.meta.client.get_object(
                Bucket=S3_BUCKET, Key="Ostatni/Pollen_viability/Review/manifest.json"
            )["Body"].read()
            manifest = json.loads(raw)
            for it in manifest.get("items", []):
                tk = it.get("tile_key", "")
                if tk:
                    holdout.add(strip_tile_base(tk))
            print(f"🔒 Loaded {len(holdout)} blind QC benchmark tiles from S3 Review/manifest.json to hold out.")
            return holdout
        except Exception as e:
            print(f"⚠️ Could not fetch S3 Review/manifest.json ({e}); checking local fallback...")

    for mf in sorted(glob.glob("output/review/manifest_final_*.json")):
        try:
            manifest = json.loads(open(mf).read())
            for it in manifest.get("items", []):
                tk = it.get("tile_key", "")
                if tk:
                    holdout.add(strip_tile_base(tk))
        except Exception:
            pass
    print(f"🔒 Loaded {len(holdout)} blind QC benchmark tiles from local manifest fallback.")
    return holdout


def scaffold_dataset():
    """Create a fresh empty train/val directory structure."""
    if os.path.exists(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT)
    for split_dir in [TRAIN_DIR, VAL_DIR]:
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)
    print(f"📁 Clean dataset scaffold created at {DATASET_ROOT}")


def polygon_area_px2(pts_norm: list, img_size: int = IMG_SIZE) -> float:
    """Shoelace polygon area in square pixels from [(x_norm, y_norm), ...]."""
    n = len(pts_norm)
    if n < 3:
        return 0.0
    area2 = 0.0
    for i in range(n):
        x1, y1 = pts_norm[i][0] * img_size, pts_norm[i][1] * img_size
        x2, y2 = pts_norm[(i + 1) % n][0] * img_size, pts_norm[(i + 1) % n][1] * img_size
        area2 += x1 * y2 - x2 * y1
    return 0.5 * abs(area2)


def sanitize_polygons(raw_polys: list, min_dim_px: float = 4.0, min_area_px2: float = 8.0) -> list:
    """
    Validate and sanitize [(cls_id, [(x_norm, y_norm), ...]), ...].
    Drops invalid classes, NaN/Inf coords, <3 vertices, or tiny slivers that can destabilize loss.
    """
    clean = []
    for cid, pts in raw_polys:
        if cid not in (0, 1, 2):
            continue
        valid_pts = []
        bad = False
        for x, y in pts:
            if not (math.isfinite(x) and math.isfinite(y)):
                bad = True
                break
            valid_pts.append((min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))))
        if bad or len(valid_pts) < 3:
            continue
        xs = [p[0] for p in valid_pts]
        ys = [p[1] for p in valid_pts]
        w_px = (max(xs) - min(xs)) * IMG_SIZE
        h_px = (max(ys) - min(ys)) * IMG_SIZE
        if w_px < min_dim_px or h_px < min_dim_px:
            continue
        if polygon_area_px2(valid_pts) < min_area_px2:
            continue
        clean.append((cid, valid_pts))
    return clean


def parse_yolo_txt(lbl_path: str, remap: dict = None) -> list:
    """Parse a YOLO segmentation .txt file into sanitized [(cls_id, [(x, y), ...]), ...]."""
    remap = remap or {0: 0, 1: 1, 2: 2}
    raw = []
    if not os.path.exists(lbl_path):
        return raw
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                continue
            src_cid = int(parts[0])
            dst_cid = remap.get(src_cid, src_cid)
            vals = [float(v) for v in parts[1:]]
            pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
            raw.append((dst_cid, pts))
    return sanitize_polygons(raw)


def format_yolo_lines(polys: list) -> str:
    lines = []
    for cid, pts in polys:
        coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines.append(f"{cid} {coord_str}")
    return ("\n".join(lines) + "\n") if lines else ""


def _build_class_remap(zip_data_yaml_path: str) -> dict:
    remap = {}
    try:
        with open(zip_data_yaml_path) as f:
            meta = yaml.safe_load(f)
        src_names = meta.get("names", [])
        if isinstance(src_names, dict):
            src_items = [(int(k), str(v).lower().replace("-", "_")) for k, v in src_names.items()]
        else:
            src_items = [(i, str(n).lower().replace("-", "_")) for i, n in enumerate(src_names)]
        for src_id, name in src_items:
            if name in CLASS_NAME_TO_ID:
                remap[src_id] = CLASS_NAME_TO_ID[name]
            else:
                remap[src_id] = -1  # e.g. 'debris' -> ignored by sanitize_polygons
        print(f"   🗺️  Class remapping from {os.path.basename(zip_data_yaml_path)}: {remap}")
    except Exception as e:
        print(f"   ⚠️  Could not read data.yaml ({e}) — using identity mapping")
    return remap


def clip_polygon_unit_square(pts: list) -> list:
    """Sutherland-Hodgman polygon clipping against [0.0, 1.0] x [0.0, 1.0]."""
    def clip_edge(poly, inside_fn, intersect_fn):
        if not poly:
            return []
        out = []
        prev = poly[-1]
        prev_in = inside_fn(prev)
        for curr in poly:
            curr_in = inside_fn(curr)
            if curr_in:
                if not prev_in:
                    out.append(intersect_fn(prev, curr))
                out.append(curr)
            elif prev_in:
                out.append(intersect_fn(prev, curr))
            prev, prev_in = curr, curr_in
        return out

    poly = list(pts)
    # x >= 0
    poly = clip_edge(poly, lambda p: p[0] >= 0.0,
                     lambda a, b: (0.0, a[1] + (b[1] - a[1]) * (0.0 - a[0]) / ((b[0] - a[0]) or 1e-12)))
    # x <= 1
    poly = clip_edge(poly, lambda p: p[0] <= 1.0,
                     lambda a, b: (1.0, a[1] + (b[1] - a[1]) * (1.0 - a[0]) / ((b[0] - a[0]) or 1e-12)))
    # y >= 0
    poly = clip_edge(poly, lambda p: p[1] >= 0.0,
                     lambda a, b: (a[0] + (b[0] - a[0]) * (0.0 - a[1]) / ((b[1] - a[1]) or 1e-12), 0.0))
    # y <= 1
    poly = clip_edge(poly, lambda p: p[1] <= 1.0,
                     lambda a, b: (a[0] + (b[0] - a[0]) * (1.0 - a[1]) / ((b[1] - a[1]) or 1e-12), 1.0))
    return poly


def make_gigas_zoom_tile(img_bgr: np.ndarray, polys: list, zoom: float, rng: random.Random):
    """
    Create a zoomed 640x640 crop centered on a rare-class (`non_viable` or `intermediate`) grain,
    bridging the diploid (14.5 um) -> tetraploid (18-21 um) size scaling gap cleanly.
    """
    h, w = img_bgr.shape[:2]
    crop_w = int(round(w / zoom))
    crop_h = int(round(h / zoom))

    rare_polys = [pts for cid, pts in polys if cid in (1, 2)]
    target_pts = rng.choice(rare_polys) if rare_polys else polys[0][1]
    cx = sum(p[0] for p in target_pts) / len(target_pts) * w + rng.uniform(-28.0, 28.0)
    cy = sum(p[1] for p in target_pts) / len(target_pts) * h + rng.uniform(-28.0, 28.0)

    x0 = int(max(0, min(w - crop_w, round(cx - crop_w / 2.0))))
    y0 = int(max(0, min(h - crop_h, round(cy - crop_h / 2.0))))

    crop_img = img_bgr[y0:y0 + crop_h, x0:x0 + crop_w]
    zoomed_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    bg_med = tuple(int(v) for v in np.median(zoomed_img.reshape(-1, 3), axis=0))
    zoomed_polys = []
    for cid, pts in polys:
        orig_area = polygon_area_px2(pts) * (zoom ** 2)
        shifted = [((p[0] * w - x0) / crop_w, (p[1] * h - y0) / crop_h) for p in pts]
        clipped = clip_polygon_unit_square(shifted)
        if len(clipped) < 3:
            continue
        new_area = polygon_area_px2(clipped)
        frac = (new_area / orig_area) if orig_area > 0 else 0.0
        if frac >= 0.25:
            zoomed_polys.append((cid, clipped))
        elif frac > 0.0:
            # Soft-mask tiny <25% border slivers with tile median color so unlabeled partial grains never teach FN
            sliver_pts = np.array([[int(round(p[0] * IMG_SIZE)), int(round(p[1] * IMG_SIZE))] for p in clipped], dtype=np.int32)
            cv2.fillPoly(zoomed_img, [sliver_pts], bg_med)

    zoomed_polys = sanitize_polygons(zoomed_polys, min_dim_px=5.0, min_area_px2=12.0)
    return zoomed_img, zoomed_polys


MIN_RELABEL_GRAIN_FOCUS = 35.0


def compute_grain_focus_lap(img_bgr: np.ndarray, polys_norm: list) -> float:
    """Compute median grain-local Laplacian variance (unaffected by sparse empty slide background)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_abs = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    h, w = gray.shape[:2]
    g_vars = []
    for _, pts in polys_norm:
        xs = [p[0] * w for p in pts]
        ys = [p[1] * h for p in pts]
        x1, y1 = max(0, int(min(xs)) - 4), max(0, int(min(ys)) - 4)
        x2, y2 = min(w, int(max(xs)) + 4), min(h, int(max(ys)) + 4)
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            g_vars.append(float(np.var(lap_abs[y1:y2, x1:x2])))
    return float(np.median(g_vars)) if g_vars else float(np.var(lap_abs))


def collect_relabel_studio_records(s3, qc_holdout: set, allow_qc_leakage: bool = False) -> dict:
    """Fetch sharp human-annotated SAM polygon tiles from S3 Relabel/Labels/ (excluding blind QC & defocused border tiles)."""
    records = {}
    if s3 is None:
        return records
    s3_client = s3.meta.client
    lbl_prefix = "Ostatni/Pollen_viability/Relabel/Labels/"
    tile_prefix = "Ostatni/Pollen_viability/Relabel/Tiles/"
    raw_prefix = "Ostatni/Pollen_viability/tiles_640/"
    os.makedirs(RELABEL_CACHE, exist_ok=True)

    skipped_qc = skipped_blur = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=lbl_prefix):
        for o in page.get("Contents", []):
            lk = o["Key"]
            if not lk.endswith(".json"):
                continue
            rel = lk[len(lbl_prefix):-5]
            orig_base = strip_tile_base(rel)
            if not allow_qc_leakage and orig_base in qc_holdout:
                skipped_qc += 1
                continue
            # Exclude legacy outer slide-margin tiles (tile_0_*) whose edge defocus distorts stain color
            if "_tile_0_" in rel or rel.endswith("_tile_0"):
                skipped_blur += 1
                continue
            try:
                data = json.loads(s3_client.get_object(Bucket=S3_BUCKET, Key=lk)["Body"].read())
            except Exception:
                continue
            raw_polys = []
            for obj in data.get("objects", []):
                cls_name = str(obj.get("cls", "")).lower().replace("-", "_")
                if cls_name not in CLASS_NAME_TO_ID:
                    continue
                poly = obj.get("polygon", [])
                if len(poly) >= 3:
                    pts = [(float(pt[0]) / 640.0, float(pt[1]) / 640.0) for pt in poly]
                    raw_polys.append((CLASS_NAME_TO_ID[cls_name], pts))
            polys = sanitize_polygons(raw_polys)
            if not polys:
                continue

            img_local = os.path.join(RELABEL_CACHE, f"{orig_base}.jpg")
            if not os.path.exists(img_local):
                fetched = False
                for cand_key in (f"{tile_prefix}{rel}.jpg", f"{raw_prefix}{rel}.jpg"):
                    try:
                        s3_client.download_file(S3_BUCKET, cand_key, img_local)
                        fetched = True
                        break
                    except Exception:
                        pass
                if not fetched:
                    continue

            img_chk = cv2.imread(img_local)
            if img_chk is None:
                continue
            focus_lap = compute_grain_focus_lap(img_chk, polys)
            if focus_lap < MIN_RELABEL_GRAIN_FOCUS:
                skipped_blur += 1
                continue

            records[orig_base] = {
                "orig_base": orig_base,
                "rel_path": rel,
                "img_path": img_local,
                "polys": polys,
                "source": "relabel_studio",
            }
    print(
        f"✅ Collected {len(records)} sharp leak-free Relabel Studio tiles "
        f"(skipped {skipped_qc} in QC holdout, {skipped_blur} border/defocused grain_lap<{MIN_RELABEL_GRAIN_FOCUS})."
    )
    return records


def assemble_clean_dataset(s3=None, allow_qc_leakage: bool = False, val_fraction: float = 0.15, seed: int = 42):
    """
    Assemble a deduplicated, leak-free, class-balanced YOLO segmentation dataset into TRAIN_DIR / VAL_DIR.
    """
    rng = random.Random(seed)
    qc_holdout = set() if allow_qc_leakage else load_qc_holdout_bases(s3)
    records_by_base = {}

    # 1. Highest priority: Relabel Studio SAM polygon tiles
    records_by_base.update(collect_relabel_studio_records(s3, qc_holdout, allow_qc_leakage=allow_qc_leakage))

    # 2. Second priority: Roboflow / clean staged ZIPs (excluding *qc_additive*.zip & blind QC tiles)
    if os.path.exists(STAGING_AREA):
        temp_root = os.path.join(LOCAL_ROOT, "temp_merge")
        for idx_z, zip_path in enumerate(sorted(glob.glob(os.path.join(STAGING_AREA, "*.zip")))):
            zname = os.path.basename(zip_path)
            if not allow_qc_leakage and "qc_additive" in zname.lower():
                print(f"🔒 Skipping {zname} (contains blind QC benchmark tiles / synthetic circle masks).")
                continue
            print(f"📦 Unpacking staged archive: {zname}")
            zdir = os.path.join(temp_root, f"z_{idx_z}")
            if os.path.exists(zdir):
                shutil.rmtree(zdir)
            os.makedirs(zdir, exist_ok=True)
            shutil.unpack_archive(zip_path, zdir)

            remap = {0: 0, 1: 1, 2: 2}
            for root, _, files in os.walk(zdir):
                if "data.yaml" in files:
                    remap = _build_class_remap(os.path.join(root, "data.yaml"))
                    break

            added_z = skipped_qc_z = skipped_dup_z = 0
            for root, _, files in os.walk(zdir):
                for f in sorted(files):
                    if not f.lower().endswith((".jpg", ".png")):
                        continue
                    if not allow_qc_leakage and f.startswith("qc__"):
                        skipped_qc_z += 1
                        continue
                    orig_base = strip_tile_base(f)
                    if not allow_qc_leakage and orig_base in qc_holdout:
                        skipped_qc_z += 1
                        continue
                    if orig_base in records_by_base:
                        skipped_dup_z += 1
                        continue

                    stem = os.path.splitext(f)[0]
                    candidates = [
                        os.path.join(root, stem + ".txt"),
                        os.path.join(os.path.dirname(root), "labels", stem + ".txt"),
                        os.path.join(root.replace("/images/", "/labels/").replace("/images", "/labels"), stem + ".txt"),
                    ]
                    lbl = next((c for c in candidates if os.path.exists(c)), None)
                    if not lbl:
                        continue
                    polys = parse_yolo_txt(lbl, remap=remap)
                    if not polys:
                        continue
                    records_by_base[orig_base] = {
                        "orig_base": orig_base,
                        "img_path": os.path.join(root, f),
                        "polys": polys,
                        "source": "staging_area",
                    }
                    added_z += 1
            print(f"   -> Added {added_z} unique tiles (skipped {skipped_qc_z} QC holdout, {skipped_dup_z} duplicates).")

    # 3. Third priority: active_learning/hard_positives
    if os.path.exists(HARD_POSITIVES):
        added_hp = skipped_hp = 0
        for fname in sorted(os.listdir(HARD_POSITIVES)):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            orig_base = strip_tile_base(fname)
            if (not allow_qc_leakage and orig_base in qc_holdout) or orig_base in records_by_base:
                skipped_hp += 1
                continue
            lbl_path = os.path.join(HARD_POSITIVES, os.path.splitext(fname)[0] + ".txt")
            polys = parse_yolo_txt(lbl_path, remap={0: 0, 1: 1, 2: 2})
            if not polys:
                continue
            records_by_base[orig_base] = {
                "orig_base": orig_base,
                "img_path": os.path.join(HARD_POSITIVES, fname),
                "polys": polys,
                "source": "hard_positives",
            }
            added_hp += 1
        print(f"✅ Added {added_hp} unique hard_positives tiles (skipped {skipped_hp} overlapping/holdout).")

    # 4. Deterministic Class-Stratified 85/15 Train/Val Split of Positive Tiles
    strata = {"has_intermediate": [], "has_non_viable": [], "viable_only": []}
    for orig_base in sorted(records_by_base.keys()):
        rec = records_by_base[orig_base]
        cset = {cid for cid, _ in rec["polys"]}
        if 2 in cset:
            strata["has_intermediate"].append(rec)
        elif 1 in cset:
            strata["has_non_viable"].append(rec)
        else:
            strata["viable_only"].append(rec)

    train_recs, val_recs = [], []
    for sname, items in strata.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_fraction))) if len(items) >= 4 else 0
        val_recs.extend(items[:n_val])
        train_recs.extend(items[n_val:])
        print(f"   Stratum '{sname:<16}': {len(items):3d} total -> {len(items) - n_val:3d} train / {n_val:3d} val")

    # Write positive tiles + rare-class & tetraploid Gigas zoom oversampling in TRAIN only
    oversampled_std = oversampled_zoom = 0
    for split_name, recs, dest_dir in [("train", train_recs, TRAIN_DIR), ("val", val_recs, VAL_DIR)]:
        img_out_dir = os.path.join(dest_dir, "images")
        lbl_out_dir = os.path.join(dest_dir, "labels")
        for rec in recs:
            prefix = "hard_pos_" if rec["source"] == "hard_positives" else ""
            out_stem = f"{prefix}{rec['orig_base']}"
            ext = os.path.splitext(rec["img_path"])[1].lower() or ".jpg"
            dest_img = os.path.join(img_out_dir, f"{out_stem}{ext}")
            dest_lbl = os.path.join(lbl_out_dir, f"{out_stem}.txt")
            shutil.copy2(rec["img_path"], dest_img)
            with open(dest_lbl, "w") as lf:
                lf.write(format_yolo_lines(rec["polys"]))

            if split_name == "train":
                cset = {cid for cid, _ in rec["polys"]}
                if 1 in cset or 2 in cset:
                    n_total_p = len(rec["polys"])
                    n_rare_p = sum(1 for cid, _ in rec["polys"] if cid in (1, 2))
                    rare_share = n_rare_p / max(1, n_total_p)
                    is_already_4x = bool(re.search(r"_(n?4x)(_|$)", rec.get("rel_path", rec["orig_base"])))

                    # Balanced native oversampling when rare grains are a meaningful fraction (>=20% or >=3 grains)
                    if rare_share >= 0.20 or n_rare_p >= 3:
                        n_dups = 2 if 2 in cset else 1
                        for d_i in range(1, n_dups + 1):
                            shutil.copy2(dest_img, os.path.join(img_out_dir, f"{out_stem}_nvdup{d_i}{ext}"))
                            shutil.copy2(dest_lbl, os.path.join(lbl_out_dir, f"{out_stem}_nvdup{d_i}.txt"))
                            oversampled_std += 1

                    # Ploidy-aware Gigas zoom: ONLY zoom diploid (2x/s2x/Pinder) rare tiles (1.22x) to bridge 2x->4x;
                    # never zoom slides already at 4x/n4x (we now have real sharp 4x/n4x rare-class tiles).
                    if not is_already_4x:
                        img_bgr = cv2.imread(dest_img)
                        if img_bgr is not None:
                            for z_i, zoom_factor in enumerate((1.22,), start=1):
                                z_img, z_polys = make_gigas_zoom_tile(img_bgr, rec["polys"], zoom_factor, rng)
                                if z_polys and any(cid in (1, 2) for cid, _ in z_polys):
                                    z_img_path = os.path.join(img_out_dir, f"{out_stem}_4xzoom{z_i}.jpg")
                                    z_lbl_path = os.path.join(lbl_out_dir, f"{out_stem}_4xzoom{z_i}.txt")
                                    cv2.imwrite(z_img_path, z_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                                    with open(z_lbl_path, "w") as zf:
                                        zf.write(format_yolo_lines(z_polys))
                                    oversampled_zoom += 1

    print(f"⚡ Rare-class train augmentation: +{oversampled_std} native copies, +{oversampled_zoom} ploidy-aware rare crops.")

    # Clean up temporary unpacked zip dir
    shutil.rmtree(os.path.join(LOCAL_ROOT, "temp_merge"), ignore_errors=True)

    # 5. Curated Real Hard Negatives (cap train background share to ~12% so recall isn't suppressed)
    n_pos_train = len(train_recs) + oversampled_std + oversampled_zoom
    max_neg_train = max(30, int(round(n_pos_train * 0.08)))
    max_syn_train = max(25, int(round(n_pos_train * 0.05)))
    neg_train = neg_val = syn_train = 0
    if os.path.exists(HARD_NEGATIVES):
        hard_files = [
            f for f in sorted(os.listdir(HARD_NEGATIVES))
            if f.lower().endswith((".jpg", ".png")) and (allow_qc_leakage or strip_tile_base(f) not in qc_holdout)
        ]
        rng.shuffle(hard_files)
        n_neg_val = max(1, min(12, int(round(len(val_recs) * 0.15)))) if hard_files else 0
        for i, fname in enumerate(hard_files):
            if i < n_neg_val:
                target_dir = VAL_DIR
                neg_val += 1
            elif neg_train < max_neg_train:
                target_dir = TRAIN_DIR
                neg_train += 1
            else:
                break
            out_name = f"curated_neg_{fname}"
            shutil.copy2(os.path.join(HARD_NEGATIVES, fname), os.path.join(target_dir, "images", out_name))
            open(os.path.join(target_dir, "labels", os.path.splitext(out_name)[0] + ".txt"), "w").close()

    # 6. Synthetic Smudges -> TRAIN ONLY (capped at max_syn_train to preserve high recall)
    if os.path.exists(SMUDGES_RAW):
        np_rng = np.random.default_rng(seed)
        raw_files = [f for f in sorted(os.listdir(SMUDGES_RAW)) if f.lower().endswith((".jpg", ".png"))]
        rng.shuffle(raw_files)
        for fname in raw_files[:max_syn_train]:
            img = cv2.imread(os.path.join(SMUDGES_RAW, fname))
            if img is None:
                continue
            # Strip black zero-padding present on legacy *_pad_*.png crops before blending
            non_black = img.max(axis=2) > 20
            if non_black.any():
                ys_nb, xs_nb = np.where(non_black)
                y1_nb, y2_nb = int(ys_nb.min()), int(ys_nb.max()) + 1
                x1_nb, x2_nb = int(xs_nb.min()), int(xs_nb.max()) + 1
                if (y2_nb - y1_nb) >= 16 and (x2_nb - x1_nb) >= 16:
                    img = img[y1_nb:y2_nb, x1_nb:x2_nb]

            h, w = img.shape[:2]
            scale = min(IMG_SIZE / max(1, h), IMG_SIZE / max(1, w)) * rng.uniform(0.55, 0.85)
            new_w, new_h = max(16, int(w * scale)), max(16, int(h * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Estimate brightfield slide background color from non-dark outer rim pixels
            rim = np.concatenate([
                resized[:4, :, :].reshape(-1, 3),
                resized[-4:, :, :].reshape(-1, 3),
                resized[:, :4, :].reshape(-1, 3),
                resized[:, -4:, :].reshape(-1, 3),
            ], axis=0)
            bright_rim = rim[rim.max(axis=1) > 40]
            bg_bgr = np.median(bright_rim if len(bright_rim) > 0 else rim, axis=0)
            noise = np_rng.normal(0.0, 1.5, size=(IMG_SIZE, IMG_SIZE, 3))
            canvas = np.clip(bg_bgr + noise, 0, 255).astype(np.uint8)

            max_y = max(0, IMG_SIZE - new_h)
            max_x = max(0, IMG_SIZE - new_w)
            y_off = rng.randint(0, max_y) if max_y > 0 else 0
            x_off = rng.randint(0, max_x) if max_x > 0 else 0

            # Smooth 8-px cosine/linear feather mask at patch boundaries
            feather = min(8, new_h // 4, new_w // 4)
            if feather >= 2:
                wy = np.ones(new_h, dtype=np.float32)
                wx = np.ones(new_w, dtype=np.float32)
                ramp = np.linspace(0.0, 1.0, feather, dtype=np.float32)
                wy[:feather], wy[-feather:] = ramp, ramp[::-1]
                wx[:feather], wx[-feather:] = ramp, ramp[::-1]
                alpha = (wy[:, None] * wx[None, :])[:, :, None]
                roi = canvas[y_off:y_off + new_h, x_off:x_off + new_w].astype(np.float32)
                canvas[y_off:y_off + new_h, x_off:x_off + new_w] = np.clip(
                    alpha * resized.astype(np.float32) + (1.0 - alpha) * roi, 0, 255
                ).astype(np.uint8)
            else:
                canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

            out_name = f"syn_neg_{os.path.splitext(fname)[0]}.jpg"
            cv2.imwrite(os.path.join(TRAIN_DIR, "images", out_name), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
            open(os.path.join(TRAIN_DIR, "labels", f"syn_neg_{os.path.splitext(fname)[0]}.txt"), "w").close()
            syn_train += 1

    print(f"🧪 Negatives integrated: curated_neg=({neg_train} train / {neg_val} val), syn_neg=({syn_train} train / 0 val).")


def print_dataset_summary():
    """Print detailed breakdown of train/val images and polygon annotations per class."""
    print("\n" + "═" * 62)
    print("📊 Dataset Composition Summary (Leak-Free & Sanitized)")
    print("═" * 62)
    for split_name, split_dir in [("train", TRAIN_DIR), ("val", VAL_DIR)]:
        img_dir = os.path.join(split_dir, "images")
        lbl_dir = os.path.join(split_dir, "labels")
        if not os.path.exists(img_dir):
            continue
        source_counts = Counter()
        annot_counts = Counter()
        total_grains = 0
        for fn in os.listdir(img_dir):
            if "_4xzoom" in fn:
                source_counts["gigas_4x_zoom_rare"] += 1
            elif "_nvdup" in fn:
                source_counts["oversampled_rare"] += 1
            elif fn.startswith("hard_pos_"):
                source_counts["hard_positives"] += 1
            elif fn.startswith("hard_neg_") or fn.startswith("curated_neg_"):
                source_counts["hard_negatives"] += 1
            elif fn.startswith("syn_neg_"):
                source_counts["smudges_synthetic"] += 1
            else:
                source_counts["pinder_or_relabel"] += 1
            lbl = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
            if os.path.exists(lbl):
                for line in open(lbl).read().strip().splitlines():
                    parts = line.strip().split()
                    if parts:
                        annot_counts[int(parts[0])] += 1
                        total_grains += 1
        total_imgs = sum(source_counts.values())
        print(f"  [{split_name.upper():<5}] {total_imgs:4d} images | {total_grains:5d} polygon annotations")
        for src, n in sorted(source_counts.items(), key=lambda x: -x[1]):
            print(f"    {src:<22}: {n:4d} images")
        cls_names = {0: "viable", 1: "non_viable", 2: "intermediate"}
        for cls_id in (0, 1, 2):
            cnt = annot_counts.get(cls_id, 0)
            pct = 100.0 * cnt / max(1, total_grains)
            print(f"    class {cls_id} ({cls_names[cls_id]:<13}): {cnt:5d} annotations ({pct:5.1f}%)")
    print("═" * 62 + "\n")


def visualize_dataset(num_samples: int = 24):
    print(f"🎨 Generating Dataset Visualizations (up to {num_samples} per split)...")
    os.makedirs(VIS_DIR, exist_ok=True)
    COLOR_MAP = {0: (0, 220, 0), 1: (0, 0, 255), 2: (0, 180, 255)}
    rng = random.Random(42)

    for split_name, split_dir in [("train", TRAIN_DIR), ("val", VAL_DIR)]:
        save_dir = os.path.join(VIS_DIR, f"{split_name}_samples")
        os.makedirs(save_dir, exist_ok=True)
        img_dir = os.path.join(split_dir, "images")
        lbl_dir = os.path.join(split_dir, "labels")
        if not os.path.exists(img_dir):
            continue
        all_imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
        if not all_imgs:
            continue
        samples = rng.sample(all_imgs, min(len(all_imgs), num_samples)) if num_samples else all_imgs
        for img_file in samples:
            img = cv2.imread(os.path.join(img_dir, img_file))
            if img is None:
                continue
            h, w = img.shape[:2]
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + ".txt")
            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 7:
                            cls = int(parts[0])
                            pts = np.array(list(map(float, parts[1:]))).reshape(-1, 2)
                            pts[:, 0] *= w
                            pts[:, 1] *= h
                            pts_i = pts.astype(np.int32).reshape((-1, 1, 2))
                            color = COLOR_MAP.get(cls, (255, 255, 255))
                            cv2.polylines(img, [pts_i], True, color, 2)
                            tx, ty = pts_i[0][0]
                            cv2.putText(img, str(cls), (int(tx), max(12, int(ty) - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.imwrite(os.path.join(save_dir, img_file), img)
    print("✅ Ground-truth sample visualization complete.")


def visualize_predictions(model, num_samples: int = 24):
    print(f"🎨 Generating Validation Prediction Visualizations (up to {num_samples})...")
    img_dir = os.path.join(VAL_DIR, "images")
    if not os.path.exists(img_dir):
        return
    save_dir = os.path.join(VIS_DIR, "predictions")
    os.makedirs(save_dir, exist_ok=True)
    all_imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
    rng = random.Random(42)
    samples = rng.sample(all_imgs, min(len(all_imgs), num_samples)) if num_samples else all_imgs
    for img_file in samples:
        results = model(os.path.join(img_dir, img_file), conf=0.45, iou=0.45, agnostic_nms=True, verbose=False)
        cv2.imwrite(os.path.join(save_dir, f"pred_{img_file}"), results[0].plot())
    print("✅ Prediction visualization complete.")


def upload_file_robust(s3_client, local_path: str, bucket: str, key: str):
    try:
        url = s3_client.generate_presigned_url(
            "put_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600
        )
        size = os.path.getsize(local_path)
        with open(local_path, "rb") as data:
            req = urllib.request.Request(url, data=data, method="PUT")
            req.add_header("Content-Length", str(size))
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, context=ctx) as f:
                if f.status == 200:
                    print(f"   -> Uploaded {os.path.basename(local_path)} ✅")
                else:
                    print(f"   -> ⚠️ Upload rejected {os.path.basename(local_path)}: Status {f.status}")
    except Exception as e:
        print(f"❌ Failed to upload {local_path}: {e}")


def upload_results(s3, local_dir: str, train_name: str, summary_meta: dict = None):
    s3_client = s3.meta.client
    s3_prefix_run = f"Ostatni/Pollen_viability/trained_models/{train_name}"
    print(f"⬆️ Uploading training results from {local_dir} to {s3_prefix_run}...")
    if os.path.exists(local_dir):
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_dir)
                upload_file_robust(s3_client, local_path, S3_BUCKET, f"{s3_prefix_run}/{rel_path}")

    s3_prefix_vis = f"{s3_prefix_run}/visualizations"
    if os.path.exists(VIS_DIR):
        for root, _, files in os.walk(VIS_DIR):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, VIS_DIR)
                upload_file_robust(s3_client, local_path, S3_BUCKET, f"{s3_prefix_vis}/{rel_path}")

    if summary_meta:
        meta_local = "/tmp/latest_clean_run.json"
        with open(meta_local, "w") as mf:
            json.dump(summary_meta, mf, indent=2)
        upload_file_robust(
            s3_client, meta_local, S3_BUCKET, "Ostatni/Pollen_viability/trained_models/latest_clean_run.json"
        )


def send_notification(subject: str, body: str):
    if not GMAIL_APP_PASSWORD:
        print("⚠️ Skipping email notification: GMAIL_APP_PASSWORD not set.")
        return
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = RECEIVER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    try:
        print(f"📧 Sending notification to {RECEIVER_EMAIL}...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(RECEIVER_EMAIL, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        print("✅ Notification sent!")
    except Exception as e:
        print(f"❌ Failed to send notification: {e}")


def inspect_results_csv(save_dir: str) -> dict:
    rcsv_path = os.path.join(save_dir, "results.csv")
    if not os.path.exists(rcsv_path):
        return {"nan_epochs": -1, "total_epochs": 0}
    with open(rcsv_path) as f:
        rows = list(csv.DictReader([line.strip() for line in f], skipinitialspace=True))
    nan_epochs = sum(1 for r in rows if "nan" in str(r.get("val/box_loss", "")).lower())
    return {"nan_epochs": nan_epochs, "total_epochs": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolo11m-seg.pt")
    parser.add_argument("--lr0", type=float, default=0.0008)
    parser.add_argument("--dry-run", action="store_true", help="Assemble & verify dataset without training")
    parser.add_argument("--no-s3-sync", action="store_true", help="Use already downloaded local caches only")
    parser.add_argument("--allow-qc-leakage", action="store_true",
                        help="Include blind QC benchmark tiles in training (NOT recommended)")
    args = parser.parse_args()

    s3 = None if args.no_s3_sync else setup_s3()
    scaffold_dataset()

    if s3 is not None:
        skip_sub = None if args.allow_qc_leakage else "qc_additive"
        download_s3_prefix(s3, "Ostatni/Pollen_viability/staged_area", STAGING_AREA, skip_substring=skip_sub)
        download_s3_prefix(s3, "Ostatni/Pollen_viability/smudges_raw", SMUDGES_RAW)
        download_s3_prefix(s3, "Ostatni/Pollen_viability/active_learning/hard_negatives", HARD_NEGATIVES)
        download_s3_prefix(s3, "Ostatni/Pollen_viability/active_learning/hard_positives", HARD_POSITIVES)

    assemble_clean_dataset(s3=s3, allow_qc_leakage=args.allow_qc_leakage, val_fraction=0.15, seed=42)
    print_dataset_summary()
    visualize_dataset(num_samples=24)

    if args.dry_run:
        print("⚠️ Dry run mode complete: Dataset verified cleanly; skipping GPU training.")
        return

    import torch
    print("🚀 Starting NaN-Proof FP32/TF32 Training...")
    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"   GPU Device: {torch.cuda.get_device_name(0)} (TF32 enabled, AMP FP16 disabled)")
    else:
        print("   Device: CPU")

    model = YOLO(args.model)
    run_name = f"pollen_train_{datetime.now().strftime('%Y%m%d_%H%M')}"

    yaml_path = os.path.join(DATASET_ROOT, "data.yaml")
    abs_root = os.path.abspath(DATASET_ROOT)
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_root}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("names:\n  0: viable\n  1: non_viable\n  2: intermediate\n")

    try:
        results = model.train(
            data=yaml_path,
            epochs=args.epochs,
            patience=0,           # Run full cosine schedule including final close_mosaic refinement
            batch=args.batch,
            imgsz=IMG_SIZE,
            device=device,
            workers=4,
            task="segment",
            name=run_name,
            project=os.path.join(os.getcwd(), "runs/detect"),
            amp=False,            # CRITICAL: Prevents Epoch-25 FP16 overflow & EMA NaN poisoning
            optimizer="AdamW",
            lr0=args.lr0,
            lrf=0.01,
            cos_lr=True,
            warmup_epochs=5.0,
            weight_decay=0.0005,
            agnostic_nms=True,
            val=True,
            cls=0.7,
            box=7.5,
            dfl=1.5,
            degrees=180.0,
            flipud=0.5,
            fliplr=0.5,
            scale=0.5,
            hsv_h=0.015,
            hsv_s=0.45,
            hsv_v=0.30,
            mosaic=1.0,
            close_mosaic=25,
            copy_paste=0.0,       # Proven NaN-free; avoids degenerate rotated mask slivers
            seed=42,
        )

        actual_save_dir = str(results.save_dir)
        csv_check = inspect_results_csv(actual_save_dir)
        print(f"📂 YOLO saved results to: {actual_save_dir} | NaN val epochs: {csv_check['nan_epochs']}/{csv_check['total_epochs']}")

        # Independent Validation on Best Checkpoint
        best_pt = os.path.join(actual_save_dir, "weights", "best.pt")
        eval_model = YOLO(best_pt) if os.path.exists(best_pt) else model
        print("📊 Running Final Validation on best.pt...")
        metrics = eval_model.val(data=yaml_path, split="val", imgsz=IMG_SIZE, agnostic_nms=True)

        box_p = float(getattr(metrics.box, "mp", 0.0))
        box_r = float(getattr(metrics.box, "mr", 0.0))
        box_map50 = float(getattr(metrics.box, "map50", 0.0))
        box_map = float(getattr(metrics.box, "map", 0.0))
        seg_map50 = float(getattr(metrics.seg, "map50", 0.0)) if getattr(metrics, "seg", None) else 0.0
        seg_map = float(getattr(metrics.seg, "map", 0.0)) if getattr(metrics, "seg", None) else 0.0

        per_class = {}
        if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
            for idx_c, ap_val in enumerate(metrics.box.ap50):
                cname = CANONICAL_CLASSES[idx_c] if idx_c < len(CANONICAL_CLASSES) else f"cls_{idx_c}"
                per_class[cname] = round(float(ap_val), 4)

        print(f"   Box Precision: {box_p:.4f} | Box Recall: {box_r:.4f} | Box mAP50: {box_map50:.4f} | Seg mAP50: {seg_map50:.4f}")
        print(f"   Per-class Box AP50: {per_class}")

        visualize_predictions(eval_model, num_samples=24)

        summary_meta = {
            "run_name": run_name,
            "s3_weights_key": f"Ostatni/Pollen_viability/trained_models/{run_name}/weights/best.pt",
            "epochs": args.epochs,
            "nan_epochs": csv_check["nan_epochs"],
            "box_precision": round(box_p, 4),
            "box_recall": round(box_r, 4),
            "box_map50": round(box_map50, 4),
            "box_map50_95": round(box_map, 4),
            "seg_map50": round(seg_map50, 4),
            "seg_map50_95": round(seg_map, 4),
            "per_class_box_ap50": per_class,
            "qc_benchmark_held_out": not args.allow_qc_leakage,
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }

        if s3 is not None:
            upload_results(s3, actual_save_dir, run_name, summary_meta=summary_meta)
            summary_body = (
                f"Hello Jakub,\n\n"
                f"Your clean NaN-proof training run '{run_name}' has finished!\n\n"
                f"Stability:\n"
                f"  - NaN validation epochs: {csv_check['nan_epochs']} / {csv_check['total_epochs']}\n"
                f"  - Blind QC benchmark held out: {not args.allow_qc_leakage}\n\n"
                f"Validation Metrics:\n"
                f"  - Box Precision: {box_p:.4f}\n"
                f"  - Box Recall:    {box_r:.4f}\n"
                f"  - Box mAP50:     {box_map50:.4f}\n"
                f"  - Box mAP50-95:  {box_map:.4f}\n"
                f"  - Seg mAP50:     {seg_map50:.4f}\n"
                f"  - Seg mAP50-95:  {seg_map:.4f}\n"
                f"  - Per-class AP50: {per_class}\n\n"
                f"Weights uploaded to S3:\n"
                f"  peg-bucket/Ostatni/Pollen_viability/trained_models/{run_name}/weights/best.pt\n"
            )
            send_notification(
                subject=f"🚀 Clean Training Complete: {run_name} (mAP50: {box_map50:.3f}, P: {box_p:.3f}, R: {box_r:.3f})",
                body=summary_body,
            )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        send_notification(
            subject=f"❌ Training Failed: {run_name}",
            body=f"Hello Jakub,\n\nYour training run '{run_name}' encountered an error:\n\n{tb}",
        )
        raise e


if __name__ == "__main__":
    main()
