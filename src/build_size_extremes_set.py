#!/usr/bin/env python3
"""
build_size_extremes_set.py — build a blinded single-grain QC audit of extreme pollen sizes
═══════════════════════════════════════════════════════════════════════════════════════════
Samples grains from the exact calibrated single-grain morphometry dataset (`grains_clean`)
across all 4 cytotypes (nat_2x, ory_2x, neo_4x, nat_4x) × 2 viability classes (viable,
non_viable) × 3 size-tail tiers:
  1. high_p99    (>= P99 area within cytotype × class; includes extreme >= P99.8 spikes)
  2. high_p95_99 (P95 – P99 upper whisker)
  3. low_p0_2    (<= P2 lower tail above class_min_diam)

Writes:
  s3://peg-bucket/Ostatni/Pollen_viability/Review/extremes_manifest.json
  output/qc/extremes_manifest.json

Does NOT touch `Review/manifest.json`, keeping the main 100-tile blind viability review
completely undisturbed.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pollen_size_analyzer import make_s3, S3_BUCKET, TILE_PREFIX  # noqa: E402
from train_model import strip_tile_base  # noqa: E402

REVIEW_PREFIX = os.environ.get("REVIEW_PREFIX", "Ostatni/Pollen_viability/Review").rstrip("/")
CLASSES = ["viable", "non_viable", "intermediate"]
CYTOTYPES = ["nat_2x", "ory_2x", "neo_4x", "nat_4x"]
CONF = 0.50


def load_training_excluded_bases(s3) -> set:
    excl = set()
    for pfx in (
        "Ostatni/Pollen_viability/Relabel/Labels/",
        "Ostatni/Pollen_viability/active_learning/hard_positives/",
        "Ostatni/Pollen_viability/active_learning/hard_negatives/",
    ):
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=pfx):
            for o in page.get("Contents", []):
                if not o["Key"].endswith("/"):
                    excl.add(strip_tile_base(o["Key"]))
    for root, _, files in os.walk("Pollen_viability"):
        if "datasets" in root:
            continue
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".txt")):
                excl.add(strip_tile_base(f))
    return excl


def box_iou_and_containment(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0, 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return (inter / union if union > 0 else 0.0), (inter / min(a1, a2) if min(a1, a2) > 0 else 0.0)


def deduplicate_detections(candidates, iou_thresh=0.40, cont_thresh=0.75):
    candidates = sorted(candidates, key=lambda c: c["conf"], reverse=True)
    kept = []
    for cand in candidates:
        if not any(
            iou > iou_thresh or cont > cont_thresh
            for k in kept
            for iou, cont in [box_iou_and_containment(cand["box"], k["box"])]
        ):
            kept.append(cand)
    return kept


def list_slide_tiles(s3, slide_name: str):
    """Return list of (ox, oy, img_key, det_key) for one slide."""
    prefix = f"{TILE_PREFIX.rstrip('/')}/{slide_name}/"
    by_stem = defaultdict(dict)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            k = o["Key"]
            if k.endswith("_det.json"):
                by_stem[k[:-9]]["det"] = k
            elif k.endswith(".jpg"):
                by_stem[k[:-4]]["img"] = k
    out = []
    for stem, d in by_stem.items():
        if "img" in d and "det" in d and "_tile_" in stem:
            try:
                y_s, x_s = stem.rsplit("_tile_", 1)[1].split("_", 1)
                out.append((int(x_s), int(y_s), d["img"], d["det"]))
            except Exception:
                pass
    return out


def anon_slide_code(slide_name: str) -> str:
    h = hashlib.sha1(f"ext_salt::{slide_name}".encode()).hexdigest()[:6].upper()
    return f"EX{h[:4]}"


def resolve_candidate(s3, row, slide_tiles, excluded_bases):
    """Find an unexcluded single 640x640 tile containing this stitched grain cleanly."""
    cx, cy = float(row["center_x_px"]), float(row["center_y_px"])
    matches = []
    for ox, oy, img_key, det_key in slide_tiles:
        if strip_tile_base(img_key) in excluded_bases:
            continue
        lx, ly = cx - ox, cy - oy
        if 28.0 <= lx <= 612.0 and 28.0 <= ly <= 612.0:
            dist2 = (lx - 320.0) ** 2 + (ly - 320.0) ** 2
            matches.append((dist2, ox, oy, img_key, det_key, lx, ly))
    matches.sort(key=lambda t: t[0])

    for _, ox, oy, img_key, det_key, lx, ly in matches[:2]:
        try:
            det = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=det_key)["Body"].read())
            boxes, polys = det.get("boxes", []), det.get("masks_xyn", [])
            cands = []
            for i, b in enumerate(boxes):
                if float(b[4]) < CONF:
                    continue
                cls_name = CLASSES[int(b[5])] if int(b[5]) < 3 else "intermediate"
                poly = polys[i] if i < len(polys) else None
                cands.append({"box": [float(v) for v in b[:4]], "conf": float(b[4]), "cls": cls_name, "poly": poly})
            kept = deduplicate_detections(cands)

            best_idx, best_d = -1, 1e9
            for idx, g in enumerate(kept):
                gx = 0.5 * (g["box"][0] + g["box"][2])
                gy = 0.5 * (g["box"][1] + g["box"][3])
                d = math.hypot(gx - lx, gy - ly)
                if g["cls"] == row["class_name"] and d < best_d:
                    best_d = d
                    best_idx = idx
            if best_idx < 0 or best_d > 10.0:
                continue

            tgt = kept[best_idx]
            # Verify the local grain area matches the stitched particle area closely (not split across seam)
            if tgt["poly"] and len(tgt["poly"]) >= 3:
                pts = np.array(tgt["poly"], dtype=np.float32) * 640.0
                local_area_um2 = float(cv2.contourArea(pts)) * (0.8772419513050967 ** 2)
            else:
                w = tgt["box"][2] - tgt["box"][0]
                h = tgt["box"][3] - tgt["box"][1]
                local_area_um2 = float(w * h * math.pi / 4.0) * (0.8772419513050967 ** 2)

            rel_diff = abs(local_area_um2 - float(row["area_um2"])) / max(1.0, float(row["area_um2"]))
            if rel_diff > 0.15:
                continue

            # Measure grain-local Laplacian sharpness on tile image
            img_bytes = s3.get_object(Bucket=S3_BUCKET, Key=img_key)["Body"].read()
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
            x1 = max(0, int(tgt["box"][0]) - 6)
            y1 = max(0, int(tgt["box"][1]) - 6)
            x2 = min(640, int(tgt["box"][2]) + 6)
            y2 = min(640, int(tgt["box"][3]) + 6)
            focus_lap = float(np.var(lap[y1:y2, x1:x2])) if (x2 > x1 and y2 > y1) else 0.0

            return {
                "tile_key": img_key,
                "det_key": det_key,
                "target_grain_index": best_idx,
                "target": {
                    "i": best_idx,
                    "lx": round(0.5 * (tgt["box"][0] + tgt["box"][2]), 2),
                    "ly": round(0.5 * (tgt["box"][1] + tgt["box"][3]), 2),
                    "box": [round(v, 2) for v in tgt["box"]],
                    "cls": tgt["cls"],
                    "conf": round(tgt["conf"], 4),
                    "area_um2": round(float(row["area_um2"]), 2),
                    "diameter_um": round(float(row["diameter_um"]), 2),
                    "solidity": round(float(row["solidity"]), 4),
                    "aspect_ratio": round(float(row["aspect_ratio"]), 3),
                },
                "grain_focus_lap": round(focus_lap, 2),
            }
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    s3 = make_s3()
    excluded_bases = load_training_excluded_bases(s3)
    print(f"🔒 Excluding {len(excluded_bases)} training/relabel tiles.")

    slides_df = pd.read_csv("outcomes/pollen_viability_slides_final.csv")
    md_df = pd.read_csv("output/qc/class_min_diam.csv")[["cytotype", "class_name", "min_diam_um"]]
    ct_df = pd.read_csv("output/qc/class_conf_threshold.csv")[["cytotype", "class_name", "conf_threshold"]]
    grains_df = pd.read_csv("output/size_analysis/pollen_grain_measurements.csv")

    g = grains_df.merge(md_df, on=["cytotype", "class_name"], how="left").merge(
        ct_df, on=["cytotype", "class_name"], how="left"
    )
    g = g[
        (g["valid_for_size"] == 1)
        & (g["slide_name"].isin(set(slides_df["slide_name"])))
        & (g["class_name"].isin(["viable", "non_viable"]))
        & (g["min_diam_um"].isna() | (g["diameter_um"] >= g["min_diam_um"]))
        & (g["conf_threshold"].isna() | (g["conf"] >= g["conf_threshold"]))
    ].copy()

    print(f"📊 Loaded {len(g):,} clean measurable grains across {g['slide_name'].nunique()} QC-passing slides.")

    # Candidate pools per (cytotype, class_name, tail_bucket)
    # Quotas per stratum: 5 in high_p99 (2 from >=P99.7 spike + 3 from P99-P99.7), 4 in high_p95_99, 3 in low_p0_2 = 12 * 8 = 96
    pools = []
    needed_slides = set()
    for cyt in CYTOTYPES:
        for cls in ["viable", "non_viable"]:
            sub = g[(g["cytotype"] == cyt) & (g["class_name"] == cls)].copy()
            sub["pct"] = sub["area_um2"].rank(pct=True) * 100.0

            p_spike = sub[sub["pct"] >= 99.7].sample(frac=1.0, random_state=args.seed)
            p_p99 = sub[(sub["pct"] >= 99.0) & (sub["pct"] < 99.7)].sample(frac=1.0, random_state=args.seed)
            p_p95 = sub[(sub["pct"] >= 95.0) & (sub["pct"] < 99.0)].sample(frac=1.0, random_state=args.seed)
            p_low = sub[sub["pct"] <= 2.0].sample(frac=1.0, random_state=args.seed)

            # Diversify slides inside each pool (max 1 per slide initially)
            def diversify(df, limit=30):
                first_pass = df.drop_duplicates(subset=["slide_name"]).head(limit)
                if len(first_pass) < limit:
                    rest = df.drop(first_pass.index).head(limit - len(first_pass))
                    return pd.concat([first_pass, rest])
                return first_pass

            c_high99 = pd.concat([diversify(p_spike, 12), diversify(p_p99, 18)])
            c_high95 = diversify(p_p95, 20)
            c_low = diversify(p_low, 20)

            pools.append((cyt, cls, "high_p99", 5, c_high99))
            pools.append((cyt, cls, "high_p95_99", 4, c_high95))
            pools.append((cyt, cls, "low_p0_2", 3, c_low))

            for df_pool in (c_high99, c_high95, c_low):
                needed_slides.update(df_pool["slide_name"].head(12).tolist())

    print(f"🔎 Indexing tiles on S3 for {len(needed_slides)} candidate slides…")
    slide_tile_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futs = {ex.submit(list_slide_tiles, s3, sname): sname for sname in needed_slides}
        for fut in concurrent.futures.as_completed(futs):
            slide_tile_map[futs[fut]] = fut.result()

    selected = []
    used_slide_tail = defaultdict(int)
    used_grain_keys = set()

    for cyt, cls, tail_bucket, quota, cand_df in pools:
        got = 0
        # Two passes: pass 0 strictly max 1 grain per slide in this bucket; pass 1 allows up to 2 if needed
        for max_per_slide in (1, 2):
            if got >= quota:
                break
            # For high_p99, ensure first 2 come from >=99.7 spike and remaining 3 from 99.0-99.7
            for _, row in cand_df.iterrows():
                if got >= quota:
                    break
                gk = (row["slide_name"], int(row["grain_id"]))
                if gk in used_grain_keys:
                    continue
                if used_slide_tail[(row["slide_name"], tail_bucket)] >= max_per_slide:
                    continue
                sname = row["slide_name"]
                if sname not in slide_tile_map:
                    slide_tile_map[sname] = list_slide_tiles(s3, sname)
                res = resolve_candidate(s3, row, slide_tile_map[sname], excluded_bases)
                if res is None:
                    continue
                used_grain_keys.add(gk)
                used_slide_tail[(sname, tail_bucket)] += 1
                got += 1
                selected.append({
                    "cytotype": cyt,
                    "class_name": cls,
                    "tail_bucket": tail_bucket,
                    "cytotype_pct": round(float(row["pct"]), 2),
                    "slide_name": sname,
                    "slide_code": anon_slide_code(sname),
                    "plant_id": str(row["plant_id"]),
                    "population": str(row["population"]),
                    "cultivation_run": str(row["cultivation_run"]),
                    "grain_id": int(row["grain_id"]),
                    **res,
                })
        print(f"  ✓ {cyt:7s} | {cls:10s} | {tail_bucket:11s} -> {got}/{quota} grains")

    # Shuffle completely so cytotypes, classes, and tails are interleaved blindly
    rng.shuffle(selected)
    for idx, it in enumerate(selected, start=1):
        h = hashlib.sha1(f"{it['tile_key']}::{it['grain_id']}".encode()).hexdigest()[:8]
        it["id"] = f"ext_{idx:03d}_{h}"
        it["order"] = idx

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    set_id = f"extremes_r3_{ts}"
    manifest = {
        "set_id": set_id,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_items": len(selected),
        "reviewers": ["jakub", "eliska"],
        "items": selected,
    }

    os.makedirs("output/qc", exist_ok=True)
    local_out = "output/qc/extremes_manifest.json"
    with open(local_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"💾 Wrote local manifest: {local_out} ({len(selected)} items, set_id={set_id})")

    if not args.dry_run:
        s3_key = f"{REVIEW_PREFIX}/extremes_manifest.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(manifest).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"☁️  Uploaded to s3://{S3_BUCKET}/{s3_key}")


if __name__ == "__main__":
    main()

