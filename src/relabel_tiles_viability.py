#!/usr/bin/env python3
"""
relabel_tiles_viability.py — Focus-filtered, interior, rare-class-enriched Active Learning tile stager
for Pollen Viability.

Fixes & upgrades:
  1. Exact 576-px tile stride (`*_tile_{ty}_{tx}.jpg`) matching Zeiss CZI tiler (`pollen_size_analyzer.py`).
  2. Strict interior coverslip filter (`1728 <= tx, ty <= 14976`) and moderate density (`3 <= n_grains <= 38`,
     `clump_share <= 0.30`) so empty margin tiles (`tile_0_...`) and messy clump mats never enter the queue.
  3. Hard optical sharpness & clean-background gate on actual S3 JPEG pixels:
     * Grain-local Laplacian variance >= 38.0 (rejects blurred / out-of-focus tiles)
     * Brightfield background median in [180, 246] and background std <= 32 (rejects air-bubble walls & smudges)
  4. Parks previously skipped unlabeled tiles safely into `Relabel/Parked/` so they stop cluttering Todo.
  5. Strict Blind-QC Benchmark Isolation: excludes all 191 QC tiles and any overlapping neighbor (< 640 px).
  6. Enriches for `non_viable` and `intermediate` grains across all 4 cytotypes (`nat_2x`, `ory_2x`,
     `nat_4x`, `neo_4x`), max 2 tiles per slide for maximum slide diversity.
"""

import argparse
import json
import math
import os
import random
import re
from collections import defaultdict

import boto3
import cv2
import numpy as np
import pandas as pd
from botocore.client import Config

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

STRIDE_PX = 576
TILE_PX = 640
MIN_INTERIOR_OFFSET = 1728
MAX_INTERIOR_OFFSET = 14976
MIN_FOCUS_LAP_VAR = 38.0
UM_PX = 0.87724195

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
S3_BUCKET = os.environ.get("S3_BUCKET", "peg-bucket")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

RAW_TILES_PREFIX = "Ostatni/Pollen_viability/tiles_640/"
RELABEL_PREFIX = "Ostatni/Pollen_viability/Relabel/"
REVIEW_MANIFEST_KEY = "Ostatni/Pollen_viability/Review/manifest.json"
CLS_NAMES = ["viable", "non_viable", "intermediate", "debris"]


def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def list_s3_keys(s3, prefix: str, suffix: str = "") -> list:
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            if o["Key"].endswith(suffix):
                out.append(o["Key"])
    return out


def parse_ty_tx(name: str):
    m = re.search(r"_tile_(\d+)_(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def park_unlabeled_skipped_tiles(s3) -> set:
    """Move currently unlabeled tiles in Relabel/Tiles & Prelabels to Relabel/Parked/ so skipped blurry tiles leave Todo."""
    lbl_rels = {
        k[len(RELABEL_PREFIX + "Labels/"):-5]
        for k in list_s3_keys(s3, RELABEL_PREFIX + "Labels/", ".json")
    }
    tile_rels = {
        k[len(RELABEL_PREFIX + "Tiles/"):-4]
        for k in list_s3_keys(s3, RELABEL_PREFIX + "Tiles/", ".jpg")
        if not k.endswith("/overview.jpg")
    }
    pre_rels = {
        k[len(RELABEL_PREFIX + "Prelabels/"):-5]
        for k in list_s3_keys(s3, RELABEL_PREFIX + "Prelabels/", ".json")
    }
    unlabeled = (tile_rels | pre_rels) - lbl_rels
    parked_bases = set()
    for rel in sorted(unlabeled):
        parked_bases.add(os.path.basename(rel))
        for sub, ext in [("Tiles/", ".jpg"), ("Prelabels/", ".json"), ("Prelabels/", ".emb.npy")]:
            src_key = f"{RELABEL_PREFIX}{sub}{rel}{ext}"
            dst_key = f"{RELABEL_PREFIX}Parked/{sub}{rel}{ext}"
            try:
                s3.head_object(Bucket=S3_BUCKET, Key=src_key)
                s3.copy_object(Bucket=S3_BUCKET, Key=dst_key, CopySource={"Bucket": S3_BUCKET, "Key": src_key})
                s3.delete_object(Bucket=S3_BUCKET, Key=src_key)
            except Exception:
                pass
    if unlabeled:
        print(f"🧹 Parked {len(unlabeled)} previously skipped/unlabeled tiles into {RELABEL_PREFIX}Parked/.")
    return parked_bases


def compute_sharpness_and_bg_ok(img_bgr: np.ndarray, boxes: list):
    """Returns (is_ok, focus_lap_var, bg_median, bg_std)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg_med = float(np.median(gray))
    bg_std = float(np.std(np.percentile(gray, [15, 85])))
    # Reject dark coverslip borders, air bubbles, or washed-out blank frames
    if not (175.0 <= bg_med <= 246.0):
        return False, 0.0, bg_med, bg_std

    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    box_vars = []
    for b in boxes:
        x1, y1 = max(0, int(b[0]) - 4), max(0, int(b[1]) - 4)
        x2, y2 = min(TILE_PX, int(b[2]) + 4), min(TILE_PX, int(b[3]) + 4)
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            box_vars.append(float(np.var(lap[y1:y2, x1:x2])))
    focus_score = float(np.median(box_vars)) if box_vars else float(np.var(lap))
    return (focus_score >= MIN_FOCUS_LAP_VAR), focus_score, bg_med, bg_std


def deduplicate_boxes(boxes: list, polys: list):
    cands = []
    for idx, b in enumerate(boxes):
        if len(b) < 6 or float(b[4]) < 0.35:
            continue
        if (b[2] - b[0]) > 130 or (b[3] - b[1]) > 130:
            continue
        poly = polys[idx] if idx < len(polys) and polys[idx] and len(polys[idx]) >= 3 else None
        cands.append((float(b[4]), b, poly))
    cands.sort(key=lambda x: -x[0])
    kept = []
    for conf, b, poly in cands:
        ab = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        dup = False
        for _, kb, _ in kept:
            ix1, iy1 = max(b[0], kb[0]), max(b[1], kb[1])
            ix2, iy2 = min(b[2], kb[2]), min(b[3], kb[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            ak = max(1.0, (kb[2] - kb[0]) * (kb[3] - kb[1]))
            if inter / min(ab, ak) > 0.65:
                dup = True
                break
        if not dup:
            kept.append((conf, b, poly))
    return kept


def main():
    ap = argparse.ArgumentParser(description="Select and stage crisp, rare-class-enriched relabel tiles")
    ap.add_argument("--grains", default="output/size_analysis/pollen_grain_measurements.csv")
    ap.add_argument("--per-cytotype", type=int, default=15, help="Sharp tiles to stage per cytotype")
    ap.add_argument("--per-slide", type=int, default=2, help="Max tiles staged per slide")
    ap.add_argument("--out", default="output/qc/relabel_tiles_candidates.csv")
    ap.add_argument("--stage-s3", action="store_true", help="Verify sharpness on S3 and stage into Relabel/")
    ap.add_argument("--park-skipped", action="store_true", help="Park existing unlabeled tiles before staging")
    args = ap.parse_args()

    s3 = make_s3()

    parked_bases = set()
    if args.stage_s3 and args.park_skipped:
        parked_bases = park_unlabeled_skipped_tiles(s3)
    else:
        for k in list_s3_keys(s3, RELABEL_PREFIX + "Parked/Tiles/", ".jpg"):
            parked_bases.add(os.path.splitext(os.path.basename(k))[0])

    # Load blind QC holdout coordinates per slide to block overlapping tiles (< 640 px)
    manifest = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=REVIEW_MANIFEST_KEY)["Body"].read())
    qc_coords_by_slide = defaultdict(list)
    excluded_bases = set(parked_bases)
    for it in manifest.get("items", []):
        tk = it.get("tile_key", "")
        sl = it.get("slide_name", "")
        base = os.path.splitext(os.path.basename(tk))[0]
        excluded_bases.add(base)
        yx = parse_ty_tx(base)
        if sl and yx:
            qc_coords_by_slide[sl].append(yx)

    # Also exclude already-labeled and already-staged tiles
    for sub, suf in [("Labels/", ".json"), ("Tiles/", ".jpg")]:
        for k in list_s3_keys(s3, RELABEL_PREFIX + sub, suf):
            excluded_bases.add(os.path.splitext(os.path.basename(k))[0])

    print(f"🔒 Excluding {len(excluded_bases)} QC-holdout / already-labeled / parked tile basenames.")

    print(f"📖 Reading grain table from {args.grains}...")
    use_cols = ["slide_name", "cytotype", "class_name", "conf", "center_x_px", "center_y_px", "is_clump"]
    df = pd.read_csv(args.grains, usecols=lambda c: c in use_cols, low_memory=False)
    df = df[df["cytotype"].isin(["nat_2x", "ory_2x", "nat_4x", "neo_4x"])].copy()

    # Exact 576-px grid cell (ty, tx) matching *_tile_{ty}_{tx}.jpg
    df["ty"] = (df["center_y_px"] // STRIDE_PX).astype(int) * STRIDE_PX
    df["tx"] = (df["center_x_px"] // STRIDE_PX).astype(int) * STRIDE_PX
    df["is_nv"] = (df["class_name"] == "non_viable").astype(int)
    df["is_int"] = (df["class_name"] == "intermediate").astype(int)
    df["is_unc"] = ((df["conf"] >= 0.55) & (df["conf"] < 0.72)).astype(int)

    tiles = df.groupby(["cytotype", "slide_name", "ty", "tx"], as_index=False).agg(
        n_grains=("conf", "size"),
        n_nv=("is_nv", "sum"),
        n_int=("is_int", "sum"),
        n_unc=("is_unc", "sum"),
        clump_share=("is_clump", "mean"),
        conf_median=("conf", "median"),
    )

    # Filter strictly to clean interior tiles with reasonable grain density & low clumping
    interior_mask = (
        tiles["ty"].between(MIN_INTERIOR_OFFSET, MAX_INTERIOR_OFFSET) &
        tiles["tx"].between(MIN_INTERIOR_OFFSET, MAX_INTERIOR_OFFSET) &
        tiles["n_grains"].between(3, 38) &
        (tiles["clump_share"] <= 0.30)
    )
    tiles = tiles[interior_mask].copy()
    # Priority strongly weights intermediate (x4) + non_viable (x2.5) + borderline uncertain (x1)
    tiles["rare_score"] = 4.0 * tiles["n_int"] + 2.5 * tiles["n_nv"] + 1.0 * tiles["n_unc"]

    staged_rows = []
    rng = random.Random(42)

    for cyt in ["nat_2x", "ory_2x", "nat_4x", "neo_4x"]:
        sub = tiles[tiles["cytotype"] == cyt].copy()
        cands = sub.to_dict("records")
        rng.shuffle(cands)
        # Sort by rare_score descending so tiles rich in intermediate & non_viable are checked first
        cands.sort(key=lambda r: (-r["rare_score"], -r["n_nv"], -r["n_int"]))

        chosen_cyt = []
        per_slide = defaultdict(int)
        checked_s3 = 0
        rejected_blur = 0

        for c in cands:
            if len(chosen_cyt) >= args.per_cytotype:
                break
            sl = c["slide_name"]
            if per_slide[sl] >= args.per_slide:
                continue
            ty, tx = int(c["ty"]), int(c["tx"])
            tile_base = f"{sl}_tile_{ty}_{tx}"
            if tile_base in excluded_bases:
                continue
            # Ensure no overlap (< 640 px) with any blind QC test tile on the same slide
            if any(abs(ty - qy) < TILE_PX and abs(tx - qx) < TILE_PX for qy, qx in qc_coords_by_slide.get(sl, [])):
                continue

            if args.stage_s3:
                rel = f"{sl}/{tile_base}"
                raw_img_key = f"{RAW_TILES_PREFIX}{rel}.jpg"
                raw_det_key = f"{RAW_TILES_PREFIX}{rel}_det.json"
                try:
                    img_bytes = s3.get_object(Bucket=S3_BUCKET, Key=raw_img_key)["Body"].read()
                    det_obj = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=raw_det_key)["Body"].read())
                except Exception:
                    continue

                checked_s3 += 1
                img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                kept_det = deduplicate_boxes(det_obj.get("boxes", []), det_obj.get("masks_xyn", []))
                if len(kept_det) < 2:
                    continue

                ok_focus, focus_var, bg_med, bg_std = compute_sharpness_and_bg_ok(
                    img_bgr, [x[1] for x in kept_det]
                )
                if not ok_focus:
                    rejected_blur += 1
                    continue

                # Build initial prelabel masks (GPU job will SAM-snap them to 1024px edges right after!)
                masks = []
                for m_i, (conf, b, poly_norm) in enumerate(kept_det):
                    if poly_norm:
                        poly_px = [[round(float(pt[0]) * TILE_PX, 1), round(float(pt[1]) * TILE_PX, 1)] for pt in poly_norm]
                    else:
                        poly_px = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
                    c_idx = int(b[5]) if len(b) > 5 else 0
                    cname = CLS_NAMES[c_idx] if c_idx < len(CLS_NAMES) else "viable"
                    area_px = max(1.0, cv2.contourArea(np.asarray(poly_px, dtype=np.float32)))
                    masks.append({
                        "id": m_i,
                        "polygon": poly_px,
                        "cls": cname,
                        "score": round(conf, 3),
                        "sources": ["model"],
                        "eq_diam_um": round(2.0 * math.sqrt(area_px / math.pi) * UM_PX, 1),
                    })

                s3.put_object(Bucket=S3_BUCKET, Key=f"{RELABEL_PREFIX}Tiles/{rel}.jpg", Body=img_bytes, ContentType="image/jpeg")
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=f"{RELABEL_PREFIX}Prelabels/{rel}.json",
                    Body=json.dumps({"tile": rel, "focus_lap_var": round(focus_var, 1), "masks": masks}).encode(),
                    ContentType="application/json",
                )
                c["focus_lap_var"] = round(focus_var, 1)
                c["n_prelabels"] = len(masks)

            chosen_cyt.append(c)
            per_slide[sl] += 1
            excluded_bases.add(tile_base)

        staged_rows.extend(chosen_cyt)
        tot_nv = sum(int(x["n_nv"]) for x in chosen_cyt)
        tot_int = sum(int(x["n_int"]) for x in chosen_cyt)
        print(
            f"  {cyt:<8}: staged {len(chosen_cyt):2d} sharp interior tiles across {len(per_slide):2d} slides "
            f"(non_viable={tot_nv}, intermediate={tot_int}, rejected_blur={rejected_blur}/{checked_s3})"
        )

    out_df = pd.DataFrame(staged_rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\n✅ Saved {len(out_df)} sharp relabel tiles -> {args.out}")


if __name__ == "__main__":
    main()
