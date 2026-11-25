#!/usr/bin/env python3
"""
precompute_sam_embeddings.py — GPU batch pre-computation of SAM ViT-B (imgsz=1024) image embeddings
(.emb.npy) AND exact SAM grain-edge snapping for all unlabeled Prelabels in Relabel/.

Why this makes the Labeling App instant:
  * Computing ViT-B 1024x1024 image features on CPU takes ~3-11 seconds per tile, whereas loading a
    2.1 MB float16 `.emb.npy` tensor takes ~15 ms and SAM's lightweight mask decoder runs in ~25 ms on CPU!
  * On a K8s GPU, encoding + snapping all ~90 Relabel + QC-flagged tiles takes < 30 seconds total.
"""

import io
import json
import math
import os
import time

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo_config")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import boto3
import cv2
import numpy as np
import torch
from botocore.client import Config
from ultralytics.models.sam import Predictor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
S3_BUCKET = os.environ.get("S3_BUCKET", "peg-bucket")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

RELABEL_TILES = "Ostatni/Pollen_viability/Relabel/Tiles/"
RELABEL_PRE = "Ostatni/Pollen_viability/Relabel/Prelabels/"
RELABEL_LBL = "Ostatni/Pollen_viability/Relabel/Labels/"
RAW_TILES = "Ostatni/Pollen_viability/tiles_640/"
UM_PX = 0.87724195
TILE = 640


def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def list_keys(s3, prefix: str, suffix: str = "") -> list:
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            if o["Key"].endswith(suffix):
                out.append(o["Key"])
    return out


def poly_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def clean_and_round_polygon(pts, iters: int = 2) -> list:
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(arr) < 3:
        return arr.round(1).tolist()
    ss = 2
    pts_ss = np.round(arr * ss).astype(np.int32)
    x0, y0 = pts_ss.min(axis=0) - 6
    w, h = pts_ss.max(axis=0) - np.array([x0, y0]) + 12
    px = arr.astype(np.float64)
    if 4 < w <= 1600 and 4 < h <= 1600:
        m = np.zeros((int(h), int(w)), np.uint8)
        cv2.fillPoly(m, [pts_ss - np.array([x0, y0])], 1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) >= 16:
                px = (cv2.approxPolyDP(c, 1.1, True)[:, 0, :].astype(np.float64) + np.array([x0, y0])) / float(ss)

    def on_border(pt):
        return pt[0] <= 1.5 or pt[1] <= 1.5 or pt[0] >= TILE - 1.5 or pt[1] >= TILE - 1.5

    a0 = poly_area(px)
    p = px
    if a0 >= 4.0 and len(p) >= 3:
        for _ in range(iters):
            q = []
            for i in range(len(p)):
                a, b = p[i], p[(i + 1) % len(p)]
                if on_border(a) and on_border(b):
                    q.extend([a.copy(), b.copy()])
                else:
                    q.extend([0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b])
            p = np.asarray(q, dtype=np.float64)
        a1 = poly_area(p)
        if a1 > 1e-6:
            cx, cy = p[:, 0].mean(), p[:, 1].mean()
            sc = math.sqrt(a0 / a1)
            for i in range(len(p)):
                if not on_border(p[i]):
                    p[i, 0] = min(float(TILE), max(0.0, cx + (p[i, 0] - cx) * sc))
                    p[i, 1] = min(float(TILE), max(0.0, cy + (p[i, 1] - cy) * sc))

    out = []
    for pt in p:
        if not out or math.hypot(pt[0] - out[-1][0], pt[1] - out[-1][1]) >= 1.6:
            out.append([round(float(pt[0]), 1), round(float(pt[1]), 1)])
    return out if len(out) >= 3 else p.round(1).tolist()


def snap_poly_with_loaded_sam(pred, polygon):
    poly = np.asarray(polygon, np.float32).reshape(-1, 2)
    if len(poly) < 3:
        return None
    bx0, by0 = poly.min(axis=0)
    bx1, by1 = poly.max(axis=0)
    pad = max(6.0, 0.24 * float(max(bx1 - bx0, by1 - by0)))
    box = [float(max(bx0 - pad, 0)), float(max(by0 - pad, 0)),
           float(min(bx1 + pad, TILE - 1)), float(min(by1 + pad, TILE - 1))]
    cx, cy = poly.mean(axis=0)

    masks = []
    for kw in (dict(bboxes=[box], points=[[float(cx), float(cy)]], labels=[1]), dict(bboxes=[box])):
        r = pred(multimask_output=True, **kw)[0]
        if r.masks is not None:
            masks += [m.astype(np.uint8) for m in r.masks.data.cpu().numpy()]
    if not masks:
        return None

    ref = np.zeros(masks[0].shape, np.uint8)
    cv2.fillPoly(ref, [np.round(poly).astype(np.int32)], 1)
    best, best_iou = None, 0.0
    for m in masks:
        inter, union = int((m & ref).sum()), int((m | ref).sum())
        iou = inter / max(union, 1)
        if iou > best_iou:
            best, best_iou = m, iou
    if best is None or best_iou < 0.52:
        return None

    cnts, _ = cv2.findContours(best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    smoothed = clean_and_round_polygon(c[:, 0, :], iters=2)
    new_area = poly_area(np.asarray(smoothed, dtype=np.float64))
    old_area = max(1.0, poly_area(poly.astype(np.float64)))
    if not (0.65 <= new_area / old_area <= 1.75):
        return None
    return smoothed, round(2.0 * math.sqrt(max(1.0, new_area) / math.pi) * UM_PX, 1)


def main():
    s3 = make_s3()
    device = "cpu"
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda:0")
            device = 0
        except Exception as e:
            print(f"⚠️ CUDA device unavailable ({e}); falling back to fast multi-core CPU.")
            device = "cpu"
    sam_weights = os.environ.get("SAM_WEIGHTS", "sam_b.pt")
    print(f"🚀 Precomputing SAM (imgsz=1024) embeddings & snapping Prelabels on device={device}...")

    pred = Predictor(overrides=dict(
        conf=0.0, task="segment", mode="predict", imgsz=1024,
        model=sam_weights, save=False, verbose=False, device=device
    ))

    # Discover all Relabel tiles + QC-flagged tiles
    tile_rels = set()
    for k in list_keys(s3, RELABEL_TILES, ".jpg"):
        rel = k[len(RELABEL_TILES):-4]
        if not rel.endswith("/overview"):
            tile_rels.add(rel)
    for k in list_keys(s3, RELABEL_PRE, ".json"):
        tile_rels.add(k[len(RELABEL_PRE):-5])

    # Also include QC-flagged / poor tiles so opening them in Labeler is instant too
    try:
        manifest = json.loads(s3.get_object(Bucket=S3_BUCKET, Key="Ostatni/Pollen_viability/Review/manifest.json")["Body"].read())
        by_id = {it["id"]: it for it in manifest.get("items", [])}
        set_id = manifest.get("set_id", "final_20261026_1614")
        for k in list_keys(s3, f"Ostatni/Pollen_viability/Review/decisions/{set_id}/", ".json"):
            d = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=k)["Body"].read())
            if d.get("relabel") or d.get("verdict") in ("poor", "minor"):
                it = by_id.get(d.get("item_id"), {})
                tk = it.get("tile_key", "")
                if tk.startswith(RAW_TILES) and tk.endswith(".jpg"):
                    tile_rels.add(tk[len(RAW_TILES):-4])
    except Exception as e:
        print(f"Notice loading QC manifest: {e}")

    existing_embs = {k[len(RELABEL_PRE):-8] for k in list_keys(s3, RELABEL_PRE, ".emb.npy")}
    labeled_rels = {k[len(RELABEL_LBL):-5] for k in list_keys(s3, RELABEL_LBL, ".json")}

    rels_sorted = sorted(tile_rels)
    print(f"📋 Found {len(rels_sorted)} target tiles ({len(existing_embs)} already have .emb.npy).")

    t0 = time.time()
    n_emb_saved = 0
    n_pre_snapped = 0

    for idx, rel in enumerate(rels_sorted, 1):
        need_emb = rel not in existing_embs
        need_snap = rel not in labeled_rels

        # Check if prelabel already snapped
        pre_obj = None
        if need_snap:
            try:
                pre_obj = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=f"{RELABEL_PRE}{rel}.json")["Body"].read())
                if pre_obj.get("sam_snapped"):
                    need_snap = False
            except Exception:
                pre_obj = None
                need_snap = False

        if not need_emb and not need_snap:
            continue

        # Download tile image
        img_bytes = None
        for cand in (f"{RELABEL_TILES}{rel}.jpg", f"{RAW_TILES}{rel}.jpg"):
            try:
                img_bytes = s3.get_object(Bucket=S3_BUCKET, Key=cand)["Body"].read()
                break
            except Exception:
                pass
        if not img_bytes:
            continue
        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        with torch.inference_mode():
            pred.set_image(img_bgr)
            if need_emb:
                emb_f16 = pred.features.detach().cpu().numpy().astype(np.float16)
                buf = io.BytesIO()
                np.save(buf, emb_f16)
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=f"{RELABEL_PRE}{rel}.emb.npy",
                    Body=buf.getvalue(),
                    ContentType="application/octet-stream",
                )
                n_emb_saved += 1

            if need_snap and pre_obj and pre_obj.get("masks"):
                snapped_cnt = 0
                new_masks = []
                for m in pre_obj["masks"]:
                    poly = m.get("polygon", [])
                    res_snap = snap_poly_with_loaded_sam(pred, poly)
                    if res_snap is not None:
                        s_poly, s_diam = res_snap
                        new_masks.append(dict(m, polygon=s_poly, eq_diam_um=s_diam, sources=["model+sam_snap"]))
                        snapped_cnt += 1
                    else:
                        c_poly = clean_and_round_polygon(poly, iters=2)
                        c_area = poly_area(np.asarray(c_poly, dtype=np.float64))
                        new_masks.append(dict(
                            m,
                            polygon=c_poly,
                            eq_diam_um=round(2.0 * math.sqrt(max(1.0, c_area) / math.pi) * UM_PX, 1),
                        ))
                pre_obj["masks"] = new_masks
                pre_obj["sam_snapped"] = True
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=f"{RELABEL_PRE}{rel}.json",
                    Body=json.dumps(pre_obj).encode(),
                    ContentType="application/json",
                )
                n_pre_snapped += 1

        if idx % 10 == 0 or idx == len(rels_sorted):
            print(f"  [{idx}/{len(rels_sorted)}] saved {n_emb_saved} .emb.npy, SAM-snapped {n_pre_snapped} prelabel tiles ({time.time()-t0:.1f}s)", flush=True)

    print(f"✅ Done in {time.time()-t0:.1f}s! Uploaded {n_emb_saved} .emb.npy files and SAM-snapped {n_pre_snapped} prelabel tiles.")


if __name__ == "__main__":
    main()

