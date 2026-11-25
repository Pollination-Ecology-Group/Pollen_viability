#!/usr/bin/env python3
"""Pollen Viability Labeling App (SAM magic wand, grain-edge snapping, polygon smoothing & YOLO export).

Ported & unified with Pollen_latitude's contour refinement toolkit:
  * SAM predictor at `imgsz=1024` with cgroup-aware CPU threading and LRU feature prefetching.
  * `clean_polygon` + area-preserving Chaikin `round_polygon` to eliminate 4-px YOLO prototype stair-steps.
  * Fast optical contrast grain-edge clipping (`clip_polygon_to_grain_edge`) applied automatically to coarse
    model pre-labels on tile load, plus interactive SAM grain-edge snapping (`/api/snap/<rel>` [E] and
    `/api/snap_all/<rel>` [Shift+E]).
  * Unified Active Learning queue combining all leak-free `Relabel/Prelabels/` tiles (prioritized first so
    annotations feed directly into `train_model.py` without blind-QC leakage) and QC-flagged tiles.
"""

import io
import json
import math
import os
import sys
import threading
import time
import zipfile
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_file

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Resolve imports
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent.parent if APP_DIR.name != "app" else APP_DIR
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(ROOT) != "/" and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import BUCKET, CFG, PREFIX, get_json, key, list_keys, put_json, s3_client  # noqa: E402

UM_PX = CFG.get("imaging", {}).get("um_per_px", 0.87724195)
CLASSES = CFG.get("classes", ["viable", "non_viable", "intermediate", "debris", "foreign_pollen"])
CACHE = Path(os.environ.get("APP_CACHE", ROOT / ".cache" / "label_app"))
CACHE.mkdir(parents=True, exist_ok=True)
LABELER = os.environ.get("LABELER", "jakub")
PASSWORD = os.environ.get("APP_PASSWORD")
TILE = 640
SNAP_MIN_IOU = 0.52

app = Flask(__name__, static_folder=None)
s3 = s3_client()


def _cpus():
    """Respect container cgroup CPU quota so PyTorch/SAM threads are not throttled on Kubernetes."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)


torch.set_num_threads(max(1, _cpus() - 1))


def auth(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if PASSWORD:
            a_ = request.authorization
            if not a_ or a_.password != PASSWORD:
                return Response("login required", 401, {"WWW-Authenticate": 'Basic realm="pollen-viability-label"'})
        return fn(*a, **kw)
    return wrapper


# ---------- S3 caching ----------

def cached(k):
    p = CACHE / k
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, k, str(p))
    return p


def tile_image(rel):
    for candidate in [key("tiles", rel + ".jpg"), f"Ostatni/Pollen_viability/tiles_640/{rel}.jpg"]:
        try:
            img = cv2.imread(str(cached(candidate)), cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass
    raise FileNotFoundError(f"Tile image {rel} not found in S3")


# ---------- Polygon Cleaning, Smoothing & Optical Edge Clipping ----------

def poly_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def round_polygon(pts, iters: int = 2, tile_size: int = TILE) -> list:
    """Area-preserving Chaikin corner-cutting (ported from Pollen_latitude):
    eliminates stair-steps and squarish YOLO prototype corners without changing grain size,
    while keeping true tile-border vertices pinned to the border."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return p.round(1).tolist()
    a0 = poly_area(p)
    if a0 < 4.0:
        return p.round(1).tolist()

    def on_border(pt):
        return pt[0] <= 1.5 or pt[1] <= 1.5 or pt[0] >= tile_size - 1.5 or pt[1] >= tile_size - 1.5

    for _ in range(iters):
        q = []
        n = len(p)
        for i in range(n):
            a, b = p[i], p[(i + 1) % n]
            if on_border(a) and on_border(b):
                q.append(a.copy())
                q.append(b.copy())
            else:
                q.append(0.75 * a + 0.25 * b)
                q.append(0.25 * a + 0.75 * b)
        p = np.asarray(q, dtype=np.float64)

    a1 = poly_area(p)
    if a1 > 1e-6:
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        scale = math.sqrt(a0 / a1)
        for i in range(len(p)):
            if not on_border(p[i]):
                p[i, 0] = min(float(tile_size), max(0.0, cx + (p[i, 0] - cx) * scale))
                p[i, 1] = min(float(tile_size), max(0.0, cy + (p[i, 1] - cy) * scale))

    # Thin to ~1.6 px vertex spacing so vertices stay easy to drag in the UI
    out = []
    for pt in p:
        if not out or math.hypot(pt[0] - out[-1][0], pt[1] - out[-1][1]) >= 1.6:
            out.append([round(float(pt[0]), 1), round(float(pt[1]), 1)])
    return out if len(out) >= 3 else p.round(1).tolist()


def clean_polygon(pts, smooth_iters: int = 2) -> list:
    """Subpixel-supersampled morphological opening (cuts 1-2 px spikes/bridges from masks2segments)
    followed by area-preserving Chaikin smoothing."""
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(arr) < 3:
        return arr.round(1).tolist()
    ss = 2  # 2x supersampling for subpixel contour fidelity
    pts_ss = np.round(arr * ss).astype(np.int32)
    x0, y0 = pts_ss.min(axis=0) - 6
    w, h = pts_ss.max(axis=0) - np.array([x0, y0]) + 12
    if w <= 4 or h <= 4 or w > 1600 or h > 1600:
        return round_polygon(arr, iters=smooth_iters)
    m = np.zeros((int(h), int(w)), np.uint8)
    cv2.fillPoly(m, [pts_ss - np.array([x0, y0])], 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return round_polygon(arr, iters=smooth_iters)
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 16:
        return round_polygon(arr, iters=smooth_iters)
    approx = (cv2.approxPolyDP(c, 1.1, True)[:, 0, :].astype(np.float64) + np.array([x0, y0])) / float(ss)
    return round_polygon(approx, iters=smooth_iters)


def clip_polygon_to_grain_edge(img_bgr: np.ndarray, pts, max_pad: int = 8) -> list:
    """Fast (< 0.4 ms) optical contrast edge snapper for coarse/squarish YOLO pre-labels.
    Recovers parts of a grain clipped by a tight YOLO bounding box (flat top/left walls) and clips
    loose background margins tightly onto the Alexander-stained exine boundary."""
    cleaned = clean_polygon(pts, smooth_iters=2)
    if img_bgr is None or len(cleaned) < 3:
        return cleaned

    arr = np.asarray(cleaned, dtype=np.float32).reshape(-1, 2)
    h_img, w_img = img_bgr.shape[:2]
    bx0, by0 = arr.min(axis=0)
    bx1, by1 = arr.max(axis=0)
    bw, bh = bx1 - bx0, by1 - by0
    if bw < 8 or bh < 8 or bw > 140 or bh > 140:
        return cleaned

    pad = int(min(12, max(max_pad, round(0.24 * max(bw, bh)))))
    x0 = max(0, int(math.floor(bx0)) - pad)
    y0 = max(0, int(math.floor(by0)) - pad)
    x1 = min(w_img, int(math.ceil(bx1)) + pad)
    y1 = min(h_img, int(math.ceil(by1)) + pad)
    crop = img_bgr[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch < 12 or cw < 12:
        return cleaned

    # Estimate local slide background color from the outer 2-px rim of the expanded crop
    rim = np.concatenate([
        crop[:2, :, :].reshape(-1, 3),
        crop[-2:, :, :].reshape(-1, 3),
        crop[:, :2, :].reshape(-1, 3),
        crop[:, -2:, :].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    bg_bgr = np.median(rim, axis=0)

    # Optical contrast against brightfield background works for magenta viable, pink intermediate, & green non-viable
    diff = np.linalg.norm(crop.astype(np.float32) - bg_bgr[None, None, :], axis=2)
    diff_blur = cv2.GaussianBlur(diff, (5, 5), 1.0)
    d_max = float(diff_blur.max())
    if d_max < 14.0:
        return cleaned  # Very faint contrast -> keep cleaned & smoothed outline

    norm_u8 = np.clip(diff_blur * (255.0 / d_max), 0, 255).astype(np.uint8)
    _, otsu = cv2.threshold(norm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Prevent merging with an adjacent touching grain via distance-transform watershed seeded at grain center
    ref_mask = np.zeros((ch, cw), np.uint8)
    local_pts = np.round(arr - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(ref_mask, [local_pts], 1)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_bin = cv2.morphologyEx((otsu > 0).astype(np.uint8), cv2.MORPH_OPEN, kern)
    fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_CLOSE, kern)

    # Connected component containing the polygon centroid (or watershed if touching a neighbor)
    cx_loc = int(np.clip(round(float(arr[:, 0].mean()) - x0), 0, cw - 1))
    cy_loc = int(np.clip(round(float(arr[:, 1].mean()) - y0), 0, ch - 1))

    dist = cv2.distanceTransform(fg_bin, cv2.DIST_L2, 5)
    # Check if there is a second local peak outside ref_mask (adjacent grain like in user's screenshot!)
    outside_eroded = cv2.erode((fg_bin & (1 - cv2.dilate(ref_mask, kern, iterations=1))).astype(np.uint8), kern, iterations=1)
    if outside_eroded.sum() >= 12 and dist[cy_loc, cx_loc] > 2.0:
        markers = np.zeros((ch, cw), dtype=np.int32)
        markers[fg_bin == 0] = 1  # background
        # Seed 2: interior of target grain
        inner_seed = cv2.erode(ref_mask, kern, iterations=2)
        if inner_seed.sum() > 0:
            markers[inner_seed > 0] = 2
        else:
            markers[cy_loc, cx_loc] = 2
        # Seed 3: adjacent grain(s) outside target polygon
        markers[outside_eroded > 0] = 3
        ws_img = cv2.cvtColor(norm_u8, cv2.COLOR_GRAY2BGR)
        cv2.watershed(ws_img, markers)
        cand_mask = (markers == 2).astype(np.uint8)
    else:
        num_lbl, lbl_im = cv2.connectedComponents(fg_bin)
        target_lbl = int(lbl_im[cy_loc, cx_loc])
        if target_lbl == 0:
            # Find label with max overlap with ref_mask
            overlaps = [int(((lbl_im == lid) & (ref_mask == 1)).sum()) for lid in range(1, num_lbl)]
            target_lbl = (1 + int(np.argmax(overlaps))) if overlaps and max(overlaps) > 0 else 0
        if target_lbl == 0:
            return cleaned
        cand_mask = (lbl_im == target_lbl).astype(np.uint8)

    inter = int((cand_mask & ref_mask).sum())
    union = int((cand_mask | ref_mask).sum())
    iou = inter / max(1, union)
    area_ref = max(1, int(ref_mask.sum()))
    area_cand = int(cand_mask.sum())
    ratio = area_cand / area_ref

    # Guardrails: accept optical edge snap only when it clearly corresponds to this single grain
    if iou < 0.48 or not (0.68 <= ratio <= 1.65):
        return cleaned

    cnts, _ = cv2.findContours(cand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return cleaned
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    solidity = cv2.contourArea(c) / max(1.0, cv2.contourArea(hull))
    if solidity < 0.84:
        return cleaned

    snapped_pts = c[:, 0, :].astype(np.float64) + np.array([x0, y0], dtype=np.float64)
    return clean_polygon(snapped_pts, smooth_iters=2)


# ---------- SAM Predictor with LRU Embedding Cache ----------

class Sam:
    """SAM predictor at imgsz=1024 (matching Pollen_latitude) with LRU embedding cache & prefetching."""

    def __init__(self, weights):
        w_path = weights
        if not os.path.exists(w_path):
            candidates = [
                ROOT / "sam_b.pt",
                ROOT / "FastSAM-s.pt",
                Path(os.environ.get("HOME", "/tmp")) / "sam_b.pt",
                Path("/tmp/sam_b.pt"),
                Path("sam_b.pt"),
            ]
            for c in candidates:
                if c.exists():
                    w_path = str(c)
                    break
            else:
                w_path = "sam_b.pt"
        self.w_path = w_path
        self.lock = threading.Lock()
        self.feats = OrderedDict()
        self.pending = set()
        self.cur_rel = None
        self.pred = None
        self._init_predictor()

    def _init_predictor(self):
        try:
            if "ultralytics" in sys.modules and not hasattr(sys.modules["ultralytics"], "__file__"):
                sys.modules.pop("ultralytics", None)
            from ultralytics.models.sam import Predictor
            print(f"📦 Initializing SAM (imgsz=1024) with weights: {self.w_path}")
            self.pred = Predictor(overrides=dict(
                conf=0.0, task="segment", mode="predict", imgsz=1024,
                model=self.w_path, save=False, verbose=False, device="cpu"
            ))
            with self.lock:
                with torch.inference_mode():
                    self.pred.set_image(np.zeros((TILE, TILE, 3), np.uint8))
                    self.pred.reset_image()
            print("✅ SAM initialized successfully")
        except Exception as e:
            print(f"⚠️ SAM init deferred or error: {e}")

    def _ensure_tile_loaded(self, rel):
        if self.pred is None:
            self._init_predictor()
            if self.pred is None:
                raise RuntimeError("SAM predictor is still loading or unavailable")
        if self.cur_rel == rel and self.pred.features is not None:
            return
        img = tile_image(rel)
        if rel in self.feats:
            self.pred.setup_source(img)
            for batch in self.pred.dataset:
                self.pred.im = self.pred.preprocess(batch[1])
                break
            self.pred.features = self.feats[rel]
        else:
            k = key("prelabels", rel + ".emb.npy")
            try:
                f = torch.from_numpy(np.load(cached(k)).astype(np.float32))
                self.pred.setup_source(img)
                for batch in self.pred.dataset:
                    self.pred.im = self.pred.preprocess(batch[1])
                    break
                self.pred.features = f
                self.feats[rel] = f
            except Exception:
                with torch.inference_mode():
                    self.pred.set_image(img)
                    self.feats[rel] = self.pred.features.clone()
            while len(self.feats) > 16:
                self.feats.popitem(last=False)
        self.cur_rel = rel

    def prefetch(self, rel):
        if self.pred is None or rel in self.feats or rel in self.pending:
            return
        self.pending.add(rel)

        def run():
            try:
                k = key("prelabels", rel + ".emb.npy")
                try:
                    f = torch.from_numpy(np.load(cached(k)).astype(np.float32))
                    self.feats[rel] = f
                except Exception:
                    img = tile_image(rel)
                    with self.lock:
                        if self.cur_rel == rel and self.pred.features is not None:
                            return
                        with torch.inference_mode():
                            self.pred.set_image(img)
                            f = self.pred.features.clone()
                            self.feats[rel] = f
                            self.cur_rel = rel
                while len(self.feats) > 16:
                    self.feats.popitem(last=False)
            except Exception as e:
                print(f"Prefetch error for {rel}: {e}")
            finally:
                self.pending.discard(rel)

        threading.Thread(target=run, daemon=True).start()

    def segment(self, rel, points, labels):
        t0 = time.time()
        while rel in self.pending and time.time() - t0 < 60:
            time.sleep(0.04)
        with self.lock:
            self._ensure_tile_loaded(rel)
            with torch.inference_mode():
                r = self.pred(points=[points], labels=[labels])[0]
        if r.masks is None:
            return None
        m = r.masks.data[0].cpu().numpy().astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 10:
            return None
        smoothed = clean_polygon(c[:, 0, :], smooth_iters=2)
        sm_area = poly_area(np.asarray(smoothed, dtype=np.float64))
        return dict(
            polygon=smoothed,
            eq_diam_um=round(2 * math.sqrt(max(1.0, sm_area) / math.pi) * UM_PX, 1),
            score=round(float(r.boxes.conf[0]) if r.boxes is not None and len(r.boxes.conf) else 1.0, 3)
        )


def snap_choice(masks, polygon, min_iou=SNAP_MIN_IOU):
    """Select SAM's candidate mask that best overlaps the outline and return a smoothed grain contour."""
    poly = np.asarray(polygon, np.float32).reshape(-1, 2)
    ref = np.zeros(masks[0].shape, np.uint8)
    cv2.fillPoly(ref, [np.round(poly).astype(np.int32)], 1)
    best, best_iou = None, 0.0
    for m in masks:
        inter, union = int((m & ref).sum()), int((m | ref).sum())
        iou = inter / max(union, 1)
        if iou > best_iou:
            best, best_iou = m, iou
    if best is None or best_iou < min_iou:
        return None, round(best_iou, 3)
    cnts, _ = cv2.findContours(best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, round(best_iou, 3)
    c = max(cnts, key=cv2.contourArea)
    smoothed = clean_polygon(c[:, 0, :], smooth_iters=2)
    return np.asarray(smoothed, dtype=np.float32), round(best_iou, 3)


def sam_snap(sam, rel, polygon):
    """Snap a coarse/squarish or manually drawn outline tightly to SAM's grain edge (ported from Pollen_latitude,
    with adaptive box padding so flat box-clipped walls expand onto the full pollen grain rim)."""
    poly = np.asarray(polygon, np.float32).reshape(-1, 2)
    if len(poly) < 3:
        return None, 0.0
    bx0, by0 = poly.min(axis=0)
    bx1, by1 = poly.max(axis=0)
    pad = max(6.0, 0.24 * float(max(bx1 - bx0, by1 - by0)))
    box = [float(max(bx0 - pad, 0)), float(max(by0 - pad, 0)),
           float(min(bx1 + pad, TILE - 1)), float(min(by1 + pad, TILE - 1))]
    cx, cy = poly.mean(axis=0)

    t = time.time()
    while rel in sam.pending and time.time() - t < 60:
        time.sleep(0.04)

    masks = []
    with sam.lock:
        sam._ensure_tile_loaded(rel)
        with torch.inference_mode():
            for kw in (
                dict(bboxes=[box], points=[[float(cx), float(cy)]], labels=[1]),
                dict(bboxes=[box]),
            ):
                r = sam.pred(multimask_output=True, **kw)[0]
                if r.masks is not None:
                    masks += [m.astype(np.uint8) for m in r.masks.data.cpu().numpy()]
    if not masks:
        return None, 0.0
    c, iou = snap_choice(masks, poly)
    if c is None or len(c) < 3:
        return None, iou
    new_area = poly_area(c.astype(np.float64))
    old_area = max(1.0, poly_area(poly.astype(np.float64)))
    ratio = new_area / old_area
    if not (0.65 <= ratio <= 1.75):
        return None, iou
    return dict(
        polygon=c.round(1).tolist(),
        eq_diam_um=round(2 * math.sqrt(max(1.0, new_area) / math.pi) * UM_PX, 1),
    ), iou


SAM = Sam(os.environ.get("SAM_WEIGHTS", str(ROOT / "sam_b.pt")))


# ---------- Tile discovery & Active Learning Queue ----------

_tiles = {"t": 0, "rows": []}


def cytotype_of(scan_name):
    s = scan_name.lower()
    for cyto in ["nat_2x", "ory_2x", "neo_4x", "nat_4x"]:
        if cyto in s:
            return cyto
    if "_n4x_" in s or s.endswith("_n4x"):
        return "neo_4x"
    if "_s2x_" in s or s.endswith("_s2x"):
        return "ory_2x"
    if "2x" in s:
        return "nat_2x"
    if "4x" in s:
        return "nat_4x"
    return "unknown"


def box_iou_and_containment(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0, 0.0
    a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return (inter / union if union > 0 else 0.0), (inter / min(a1, a2))


def deduplicate_detections(candidates, iou_thresh=0.40, cont_thresh=0.75):
    candidates = sorted(candidates, key=lambda c: c["conf"], reverse=True)
    kept = []
    for cand in candidates:
        if not any(
            box_iou_and_containment(cand["box"], k["box"])[0] > iou_thresh
            or box_iou_and_containment(cand["box"], k["box"])[1] > cont_thresh
            for k in kept
        ):
            kept.append(cand)
    return kept


_manifest_items = {}
_manifest_by_id = {}
_qc_decisions = {}


def tile_rows(force=False):
    if force or time.time() - _tiles["t"] > 60:
        pre_keys = list_keys(s3, key("prelabels"), ".json")
        tile_only_keys = list_keys(s3, key("tiles"), ".jpg")
        done_keys = set(list_keys(s3, key("labels"), ".json"))
        done_ids = {k[len(key("labels")):-5] for k in done_keys}

        # Load QC review decisions across all reviewers
        _qc_decisions.clear()
        try:
            qc_pfx = CFG["s3"].get("review_prefix", "Ostatni/Pollen_viability/Review/").rstrip("/") + "/decisions/"
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=qc_pfx):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".json"):
                        try:
                            dec = json.loads(s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read())
                            parts = obj["Key"][len(qc_pfx):].split("/")
                            if len(parts) >= 2:
                                item_id = parts[1]
                                rev = parts[2].replace(".json", "") if len(parts) >= 3 else "rev"
                                _qc_decisions.setdefault(item_id, {})[rev] = dec
                        except Exception:
                            pass
        except Exception as e:
            print(f"QC decisions load notice: {e}")

        # Load Review manifest.json
        manifest_obj = get_json(s3, "Ostatni/Pollen_viability/Review/manifest.json")
        raw_pfx = CFG["s3"].get("tiles_raw", "Ostatni/Pollen_viability/tiles_640/")
        _manifest_items.clear()
        _manifest_by_id.clear()

        if manifest_obj and "items" in manifest_obj:
            for it in manifest_obj["items"]:
                tk = it.get("tile_key", "")
                rel = tk[len(raw_pfx):] if tk.startswith(raw_pfx) else tk
                if rel.endswith(".jpg"):
                    rel = rel[:-4]
                _manifest_items[rel] = it
                _manifest_by_id[it["id"]] = it

        rows_by_id = {}

        # 1. Include all staged Relabel/Tiles/, Relabel/Prelabels/, AND all saved Relabel/Labels/ (done_ids)
        staged_rels = set(done_ids)
        for k in pre_keys:
            staged_rels.add(k[len(key("prelabels")):-5])
        for k in tile_only_keys:
            rel_c = k[len(key("tiles")):-4]
            if not rel_c.endswith("/overview"):
                staged_rels.add(rel_c)

        for rel in sorted(staged_rels):
            parts = rel.split("/")
            scan = parts[-2] if len(parts) >= 2 else "scan"
            it = _manifest_items.get(rel, {})
            cyto = it.get("cytotype") or cytotype_of(scan)
            rows_by_id[rel] = dict(
                id=rel,
                item_id=it.get("id", ""),
                scan=scan,
                cytotype=cyto,
                labeled=rel in done_ids,
                in_qc_holdout=bool(it),
                density=it.get("density", "medium"),
                selection=it.get("selection", "active_learning"),
                flag_reason=it.get("flag_reason", ""),
                n_pred=it.get("n_pred", 0),
            )

        # 2. Also include QC-manifest tiles (marked in_qc_holdout=True) so View-in-Labeller links & QC flags work
        for rel, it in _manifest_items.items():
            if rel in rows_by_id:
                continue
            scan = it.get("slide_name") or (rel.split("/")[-2] if "/" in rel else "scan")
            cyto = it.get("cytotype") or cytotype_of(scan)
            rows_by_id[rel] = dict(
                id=rel,
                item_id=it["id"],
                scan=scan,
                cytotype=cyto,
                labeled=rel in done_ids,
                in_qc_holdout=True,
                density=it.get("density", "medium"),
                selection=it.get("selection", "qc_benchmark"),
                flag_reason=it.get("flag_reason", ""),
                n_pred=it.get("n_pred", 0),
            )

        rows = list(rows_by_id.values())

        done_by_cyto = defaultdict(int)
        total_by_cyto = defaultdict(int)
        for r in rows:
            if not r["in_qc_holdout"]:
                c_key = r["cytotype"]
                total_by_cyto[c_key] += 1
                if r["labeled"]:
                    done_by_cyto[c_key] += 1

        for r in rows:
            c_key = r["cytotype"]
            r["sp_done"] = done_by_cyto[c_key]
            r["sp_total"] = total_by_cyto[c_key]

            item_id = r.get("item_id", "")
            tile_decs = _qc_decisions.get(item_id, {})
            score = 0
            reasons = []

            if not r["in_qc_holdout"]:
                # Leak-free training queue prioritized FIRST so every labeled tile enters train_model.py!
                score += 25000
                reasons.append("🎯 Leak-Free Retrain")
                # Balance cytotypes dynamically (fewest completed first)
                score += max(0, (25 - done_by_cyto[c_key])) * 250

            has_relabel = any(d.get("relabel") for d in tile_decs.values())
            has_poor = any(d.get("verdict") == "poor" for d in tile_decs.values())
            has_minor = any(d.get("verdict") == "minor" for d in tile_decs.values())
            n_fps = sum(sum(1 for v in d.get("grains", {}).values() if v == "fp") for d in tile_decs.values())
            n_class_errs = sum(sum(1 for v in d.get("grains", {}).values() if str(v).startswith("to_")) for d in tile_decs.values())
            n_missed = sum(len(d.get("missed", [])) for d in tile_decs.values())

            if has_relabel:
                score += 15000
                reasons.append("🏷️ QC Relabel Flag")
            elif has_poor:
                score += 10000
                reasons.append("⚠️ Poor QC Verdict")
            elif has_minor:
                score += 4000
                reasons.append("QC Minor Issues")

            if n_fps > 0:
                score += n_fps * 2000
                reasons.append(f"{n_fps} False Positives")
            if n_class_errs > 0:
                score += n_class_errs * 2500
                reasons.append(f"{n_class_errs} Misclassified")
            if n_missed > 0:
                score += n_missed * 3000
                reasons.append(f"{n_missed} Missed Grains")

            if r.get("cytotype") in ("neo_4x", "nat_4x", "ory_2x"):
                score += 600

            score += (hash(r["id"]) % 100)

            r["flagged"] = has_relabel or (not r["in_qc_holdout"])
            r["improvement_score"] = score
            r["improvement_reasons"] = reasons

        unlabeled = [r for r in rows if not r["labeled"]]
        labeled = [r for r in rows if r["labeled"]]
        unlabeled.sort(key=lambda r: -r["improvement_score"])
        labeled.sort(key=lambda r: -r["improvement_score"])

        for idx, r in enumerate(unlabeled):
            r["priority_rank"] = idx + 1
        for idx, r in enumerate(labeled):
            r["priority_rank"] = 10000 + idx + 1

        rows = unlabeled + labeled
        _tiles.update(t=time.time(), rows=rows)
    return _tiles["rows"]


# ---------- Routes ----------

@app.get("/")
@auth
def index():
    return send_file(Path(__file__).with_name("index.html"))


@app.get("/api/tiles")
@auth
def api_tiles():
    return jsonify(tile_rows(force=request.args.get("refresh") == "1"))


@app.get("/api/image/<path:rel>")
@auth
def api_image(rel):
    for candidate in [key("tiles", rel + ".jpg"), f"Ostatni/Pollen_viability/tiles_640/{rel}.jpg"]:
        try:
            return send_file(cached(candidate), mimetype="image/jpeg", max_age=86400)
        except Exception:
            pass
    return Response("Image not found", 404)


@app.get("/api/tile/<path:rel>")
@auth
def api_tile(rel):
    SAM.prefetch(rel)
    pre = get_json(s3, key("prelabels", rel + ".json"), {"masks": []})
    lab = get_json(s3, key("labels", rel + ".json"))
    scan = rel.split("/")[-2] if "/" in rel else "scan"
    cyto = cytotype_of(scan)

    it_meta = _manifest_items.get(rel, {})
    item_id = it_meta.get("id", "")
    row_info = next((r for r in _tiles["rows"] if r["id"] == rel), {})
    imp_score = row_info.get("improvement_score", 0)
    imp_reasons = row_info.get("improvement_reasons", [])

    try:
        img_bgr = tile_image(rel) if not lab else None
    except Exception:
        img_bgr = None

    masks = []
    raw_pre_masks = pre.get("masks", [])
    if raw_pre_masks and not lab:
        # Clean, smooth & optically clip pre-staged model masks so squarish boxes become smooth grain contours
        for idx, m_in in enumerate(raw_pre_masks):
            poly_in = m_in.get("polygon", [])
            if len(poly_in) < 3:
                continue
            refined = clip_polygon_to_grain_edge(img_bgr, poly_in)
            area = poly_area(np.asarray(refined, dtype=np.float64))
            eq_diam = round(2.0 * math.sqrt(max(1.0, area) / math.pi) * UM_PX, 1)
            masks.append(dict(
                m_in,
                id=m_in.get("id", idx),
                polygon=refined,
                eq_diam_um=eq_diam,
            ))
    elif not lab:
        det_key = it_meta.get("det_key") or f"Ostatni/Pollen_viability/tiles_640/{rel}_det.json"
        raw_det = get_json(s3, det_key) or get_json(s3, f"Ostatni/Pollen_viability/tiles_640/{rel}_det.json")

        if raw_det:
            if "boxes" in raw_det:
                boxes = raw_det.get("boxes", [])
                polys = raw_det.get("masks_xyn", [])
                candidates = []
                for idx, b in enumerate(boxes):
                    if len(b) > 4 and b[4] < 0.30:
                        continue
                    c_idx = int(b[5]) if len(b) > 5 else 0
                    cls_name = CLASSES[c_idx] if c_idx < len(CLASSES) else "viable"
                    poly = polys[idx] if idx < len(polys) and polys[idx] and len(polys[idx]) >= 3 else None
                    candidates.append({
                        "i": idx, "box": b[:4], "conf": float(b[4]) if len(b) > 4 else 0.8,
                        "cls": cls_name, "poly": poly
                    })

                kept = deduplicate_detections(candidates, iou_thresh=0.40, cont_thresh=0.75)
                qc_corrections = _qc_decisions.get(item_id, {})
                active_dec = next(iter(qc_corrections.values()), {}) if qc_corrections else {}
                marks = active_dec.get("grains", {})

                for idx, cand in enumerate(kept):
                    mark = marks.get(str(cand["i"]), "ok")
                    if mark == "fp":
                        continue
                    cls_name = cand["cls"]
                    if mark.startswith("to_") and mark[3:] in CLASSES:
                        cls_name = mark[3:]

                    if cand["poly"]:
                        poly_px = [[float(pt[0]) * 640.0, float(pt[1]) * 640.0] for pt in cand["poly"]]
                    else:
                        b = cand["box"]
                        poly_px = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]

                    refined = clip_polygon_to_grain_edge(img_bgr, poly_px)
                    area = poly_area(np.asarray(refined, dtype=np.float64))
                    eq_diam = round(2.0 * math.sqrt(max(1.0, area) / math.pi) * UM_PX, 1)

                    masks.append(dict(
                        id=idx, polygon=refined, cls=cls_name,
                        score=round(cand["conf"], 3), sources=["model+edge"],
                        eq_diam_um=eq_diam
                    ))

                for m_idx, mk in enumerate(active_dec.get("missed", [])):
                    mx, my = float(mk.get("x", 320)), float(mk.get("y", 320))
                    r_px = 14.0
                    circle = [[mx + r_px * math.cos(a), my + r_px * math.sin(a)]
                              for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
                    refined = clip_polygon_to_grain_edge(img_bgr, circle)
                    area = poly_area(np.asarray(refined, dtype=np.float64))
                    masks.append(dict(
                        id=1000 + m_idx, polygon=refined, cls=mk.get("cls", "viable"),
                        score=1.0, sources=["qc_missed"],
                        eq_diam_um=round(2.0 * math.sqrt(max(1.0, area) / math.pi) * UM_PX, 1)
                    ))

            elif "grains" in raw_det:
                for idx, g in enumerate(raw_det["grains"]):
                    poly = g.get("polygon")
                    if not poly and "bbox" in g:
                        b = g["bbox"]
                        poly = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
                    if poly:
                        refined = clip_polygon_to_grain_edge(img_bgr, poly)
                        area = poly_area(np.asarray(refined, dtype=np.float64))
                        masks.append(dict(
                            id=idx, polygon=refined, cls=g.get("class", "viable"),
                            score=g.get("confidence", 0.8),
                            sources=["model+edge"],
                            eq_diam_um=round(2.0 * math.sqrt(max(1.0, area) / math.pi) * UM_PX, 1)
                        ))

    return jsonify(
        id=rel, item_id=item_id, scan=scan, cytotype=cyto, species=cyto,
        in_qc_holdout=bool(it_meta),
        prelabels=masks, label=lab,
        improvement_score=imp_score, improvement_reasons=imp_reasons,
        um_per_px=UM_PX, classes=CLASSES
    )


@app.get("/api/tile_by_id/<item_id>")
@auth
def api_tile_by_id(item_id):
    tile_rows()
    it = _manifest_by_id.get(item_id)
    if not it:
        return Response("Not found", 404)
    raw_pfx = CFG["s3"].get("tiles_raw", "Ostatni/Pollen_viability/tiles_640/")
    rel = it.get("tile_key", "")
    if rel.startswith(raw_pfx):
        rel = rel[len(raw_pfx):]
    if rel.endswith(".jpg"):
        rel = rel[:-4]
    return jsonify(rel=rel, item_id=item_id)


@app.post("/api/prefetch/<path:rel>")
@auth
def api_prefetch(rel):
    SAM.prefetch(rel)
    return jsonify(ok=True)


@app.post("/api/sam/<path:rel>")
@auth
def api_sam(rel):
    d = request.get_json()
    t0 = time.time()
    out = SAM.segment(rel, d["points"], d["labels"])
    return jsonify(mask=out, ms=int(1000 * (time.time() - t0)))


@app.post("/api/snap/<path:rel>")
@auth
def api_snap(rel):
    """Snap a single outline to SAM's grain boundary (hotkey E / automatic after manual Draw Poly)."""
    d = request.get_json()
    t0 = time.time()
    out, iou = sam_snap(SAM, rel, d["polygon"])
    return jsonify(mask=out, iou=iou, ms=int(1000 * (time.time() - t0)))


@app.post("/api/snap_all/<path:rel>")
@auth
def api_snap_all(rel):
    """Snap every outline on the tile to SAM's grain boundary in one batch using cached tile features."""
    d = request.get_json()
    polygons = d.get("polygons", [])
    t0 = time.time()
    results = []
    n_snapped = 0
    for poly in polygons:
        out, iou = sam_snap(SAM, rel, poly)
        if out is not None:
            results.append(out)
            n_snapped += 1
        else:
            cleaned = round_polygon(poly, iters=2)
            a = poly_area(np.asarray(cleaned, dtype=np.float64))
            results.append(dict(
                polygon=cleaned,
                eq_diam_um=round(2 * math.sqrt(max(1.0, a) / math.pi) * UM_PX, 1),
            ))
    return jsonify(masks=results, n_snapped=n_snapped, total=len(polygons), ms=int(1000 * (time.time() - t0)))


@app.post("/api/label/<path:rel>")
@auth
def api_label(rel):
    d = request.get_json()
    objs = [o for o in d["objects"] if o.get("cls") in CLASSES and len(o.get("polygon", [])) >= 3]
    scan = rel.split("/")[-2] if "/" in rel else "scan"
    cyto = cytotype_of(scan)

    old = get_json(s3, key("labels", rel + ".json"))
    if old:
        put_json(s3, key("review", "label_history", rel, old.get("saved_at", "").replace(":", "") + ".json"), old)

    record = dict(
        tile=rel, scan=scan, cytotype=cyto, species=cyto, labeler=LABELER,
        saved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status=d.get("status", "done"), objects=objs,
        n_prelabels_rejected=d.get("n_rejected", 0),
        seconds_spent=d.get("seconds_spent", 0)
    )
    put_json(s3, key("labels", rel + ".json"), record)

    for r in _tiles["rows"]:
        if r["id"] == rel:
            r["labeled"] = True
    _tiles["t"] = 0
    _labels.pop(rel, None)
    return jsonify(ok=True, n=len(objs))


_labels = {}


@app.get("/api/labels")
@auth
def api_labels():
    root = key("labels")
    rels = [k[len(root):-5] for k in list_keys(s3, root, ".json")]
    missing = [r for r in rels if r not in _labels]
    with ThreadPoolExecutor(16) as ex:
        for r, d in zip(missing, ex.map(lambda r: get_json(s3, key("labels", r + ".json")), missing)):
            _labels[r] = d
    out = []
    for r in rels:
        d = _labels.get(r) or {}
        scan = r.split("/")[-2] if "/" in r else "scan"
        out.append(dict(
            id=r, scan=scan, cytotype=d.get("cytotype", cytotype_of(scan)), species=d.get("cytotype", cytotype_of(scan)),
            status=d.get("status", "done"), saved_at=d.get("saved_at", ""),
            labeler=d.get("labeler", ""), seconds=d.get("seconds_spent"),
            objects=[dict(cls=o["cls"], polygon=o["polygon"]) for o in d.get("objects", [])]
        ))
    out.sort(key=lambda x: x.get("saved_at") or "", reverse=True)
    return jsonify(out)


@app.get("/api/overview/<path:scan_rel>")
@auth
def api_overview(scan_rel):
    try:
        return send_file(cached(key("tiles", scan_rel + "/overview.jpg")), mimetype="image/jpeg")
    except Exception:
        return Response("Not found", 404)


@app.get("/api/export_yolo")
@auth
def api_export_yolo():
    """Dynamically bundles leak-free saved ground-truth polygon labels into a YOLO segmentation dataset."""
    root = key("labels")
    rels = [k[len(root):-5] for k in list_keys(s3, root, ".json")]
    if not rels:
        return jsonify(error="No labeled tiles yet"), 400

    cls_map = {"viable": 0, "non_viable": 1, "intermediate": 2, "debris": 3}
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        data_yaml = (
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "nc: 3\n"
            "names:\n"
            "  0: viable\n"
            "  1: non_viable\n"
            "  2: intermediate\n"
        )
        z.writestr("dataset/data.yaml", data_yaml)

        np.random.seed(42)
        shuffled = list(rels)
        np.random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * 0.2))
        val_set = set(shuffled[:n_val])

        for rel in rels:
            d = get_json(s3, key("labels", rel + ".json"))
            if not d:
                continue
            split = "val" if rel in val_set else "train"
            stem = rel.replace("/", "__")

            try:
                img_path = cached(key("tiles", rel + ".jpg"))
                z.write(str(img_path), f"dataset/images/{split}/{stem}.jpg")
            except Exception:
                try:
                    img_path = cached(f"Ostatni/Pollen_viability/tiles_640/{rel}.jpg")
                    z.write(str(img_path), f"dataset/images/{split}/{stem}.jpg")
                except Exception:
                    continue

            lines = []
            for o in d.get("objects", []):
                c = o.get("cls")
                if c not in cls_map or c == "debris":
                    continue
                cid = cls_map[c]
                coords = [f"{pt[0] / 640.0:.6f} {pt[1] / 640.0:.6f}" for pt in o.get("polygon", [])]
                if len(coords) >= 3:
                    lines.append(f"{cid} " + " ".join(coords))
            z.writestr(f"dataset/labels/{split}/{stem}.txt", "\n".join(lines) + "\n")

    buf.seek(0)
    return Response(
        buf.getvalue(), mimetype="application/zip",
        headers={"Content-Disposition": f"attachment; filename=yolo_dataset_{len(rels)}_tiles.zip"}
    )


@app.get("/healthz")
def healthz():
    return "ok"


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8050))
    print(f"🚀 Starting Pollen Viability Labeling App on http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)
