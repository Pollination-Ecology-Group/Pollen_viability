#!/usr/bin/env python3
"""Export ground-truth polygon labels from label_app + human QC review decisions into a YOLO
segmentation dataset, and optionally upload a timestamped additive training ZIP to S3 staged_area/.

Combines:
1. All full polygon-relabeled tiles in S3 `Ostatni/Pollen_viability/Relabel/Labels/`
2. All human-verified non-blurred QC tiles in S3 `Ostatni/Pollen_viability/Review/decisions/<set_id>/`
   * Excludes blurred / out-of-focus tiles (`image_issue == True` or grain-local Laplacian variance `< 12.0`).
   * Includes `verdict in ("good", "minor")` tiles and `verdict == "poor"` tiles that were explicitly
     corrected or serve as verified hard-negative background controls (all detections marked `fp`).
   * Soft-masks any `fp` detection polygon with the tile's median background color so unannotated
     stain/debris does not confuse training, and omits it from `.txt`.
   * Applies human class reclassifications (`to_viable` -> 0, `to_non_viable` -> 1, `to_intermediate` -> 2).
   * Synthesizes circular/contour polygon masks around clicked `missed` grains (`{'x', 'y', 'cls'}`).

Usage:
    python src/export_relabel_dataset.py --upload-staged
"""

import argparse
import datetime
import io
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import boto3
import cv2
import numpy as np
import yaml
from botocore.client import Config

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
S3_BUCKET = os.environ.get("S3_BUCKET", "peg-bucket")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "1Y920BKC0SAWPNDE8RD6")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD")

LABELS_PREFIX = "Ostatni/Pollen_viability/Relabel/Labels/"
TILES_PREFIX = "Ostatni/Pollen_viability/Relabel/Tiles/"
RAW_TILES_PREFIX = "Ostatni/Pollen_viability/tiles_640/"
REVIEW_MANIFEST_KEY = "Ostatni/Pollen_viability/Review/manifest.json"
REVIEW_DECISIONS_PREFIX = "Ostatni/Pollen_viability/Review/decisions/"
STAGED_AREA_PREFIX = "Ostatni/Pollen_viability/staged_area/"

CLASS_MAP = {
    "viable": 0,
    "non_viable": 1,
    "intermediate": 2,
    "debris": 3,
}
RECLASS_MAP = {
    "viable": 0,
    "to_viable": 0,
    "non_viable": 1,
    "to_non_viable": 1,
    "intermediate": 2,
    "to_intermediate": 2,
}
CLASS_NAMES = {0: "viable", 1: "non_viable", 2: "intermediate", 3: "debris"}
MIN_BOX_LAPLACIAN_VAR = 12.0


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def list_s3_keys(s3, prefix, suffix=""):
    out = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            if o["Key"].endswith(suffix):
                out.append(o["Key"])
    return out


def download_image_bytes(s3, candidates):
    for key in candidates:
        if not key:
            continue
        try:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
            return resp["Body"].read()
        except Exception:
            pass
    return None


def compute_grain_focus_score(img_bgr: np.ndarray, boxes: list) -> float:
    """Computes median grain-local Laplacian variance (or full-tile if no boxes)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    box_laps = []
    for b in boxes:
        x1 = max(0, int(b[0]) - 4)
        y1 = max(0, int(b[1]) - 4)
        x2 = min(img_bgr.shape[1], int(b[2]) + 4)
        y2 = min(img_bgr.shape[0], int(b[3]) + 4)
        if x2 > x1 and y2 > y1:
            box_laps.append(float(np.var(lap[y1:y2, x1:x2])))
    if box_laps:
        return float(np.median(box_laps))
    return float(np.var(lap))


def make_circle_polygon_norm(cx: float, cy: float, radius_px: float, w: int = 640, h: int = 640, n_pts: int = 16) -> list:
    """Generates a normalized [x1, y1, ...] polygon around (cx, cy)."""
    coords = []
    for i in range(n_pts):
        ang = 2.0 * math.pi * i / n_pts
        px = min(1.0, max(0.0, (cx + radius_px * math.cos(ang)) / float(w)))
        py = min(1.0, max(0.0, (cy + radius_px * math.sin(ang)) / float(h)))
        coords.append(f"{px:.6f}")
        coords.append(f"{py:.6f}")
    return coords


def main():
    parser = argparse.ArgumentParser(description="Export Relabel + QC review labels to YOLO segmentation dataset")
    parser.add_argument("--output-dir", default=str(ROOT / "datasets" / "pollen_retrain"),
                        help="Local output directory for dataset")
    parser.add_argument("--val-split", type=float, default=0.20, help="Validation set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include-debris", action="store_true", help="Include debris as a 4th class")
    parser.add_argument("--include-qc-benchmark", action="store_true",
                        help="Include blind QC benchmark tiles in training export (default: False to prevent test leakage)")
    parser.add_argument("--upload-staged", action="store_true",
                        help="Package exported dataset as a ZIP and upload to S3 staged_area/")
    parser.add_argument("--blur-report-out", default=str(ROOT / "output" / "qc" / "blurred_tiles_filter.csv"),
                        help="Path to write tile focus & blur QC table")
    args = parser.parse_args()

    random.seed(args.seed)
    s3 = get_s3()

    print(f"🔍 Loading QC benchmark manifest: s3://{S3_BUCKET}/{REVIEW_MANIFEST_KEY}...")
    manifest = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=REVIEW_MANIFEST_KEY)["Body"].read())
    items_by_id = {it["id"]: it for it in manifest.get("items", [])}
    qc_benchmark_bases = {os.path.splitext(os.path.basename(it["tile_key"]))[0] for it in manifest.get("items", [])}
    if not args.include_qc_benchmark:
        print(f"🔒 Holding out {len(qc_benchmark_bases)} blind QC benchmark tiles from export.")

    out_root = Path(args.output_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ["train", "val"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── 1. Collect Relabel Studio full polygon labels ─────────────────────────
    print(f"🔍 Scanning S3: s3://{S3_BUCKET}/{LABELS_PREFIX} for Relabel Studio labels...")
    label_keys = list_s3_keys(s3, LABELS_PREFIX, ".json")
    print(f"📋 Found {len(label_keys)} Relabel Studio tiles.")

    exported_records = []  # list of dicts: {stem, img_bgr, label_lines, source, slide_name, cytotype}
    covered_tile_basenames = set()

    for lk in label_keys:
        rel = lk[len(LABELS_PREFIX):-5]
        tile_base = os.path.basename(rel)
        if not args.include_qc_benchmark and tile_base in qc_benchmark_bases:
            continue
        stem = rel.replace("/", "__")

        try:
            data = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=lk)["Body"].read())
        except Exception as e:
            print(f"⚠️ Failed to read {lk}: {e}")
            continue

        objects = data.get("objects", [])
        if not objects:
            continue

        raw_bytes = download_image_bytes(s3, [f"{TILES_PREFIX}{rel}.jpg", f"{RAW_TILES_PREFIX}{rel}.jpg"])
        if not raw_bytes:
            print(f"⚠️ Image not found for {rel}")
            continue

        img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        label_lines = []
        for o in objects:
            cls_name = o.get("cls")
            if cls_name not in CLASS_MAP:
                continue
            if cls_name == "debris" and not args.include_debris:
                continue
            cid = CLASS_MAP[cls_name]
            poly = o.get("polygon", [])
            if len(poly) < 3:
                continue
            coords = []
            for pt in poly:
                coords.append(f"{min(1.0, max(0.0, pt[0] / 640.0)):.6f}")
                coords.append(f"{min(1.0, max(0.0, pt[1] / 640.0)):.6f}")
            label_lines.append((cid, f"{cid} " + " ".join(coords)))

        if label_lines:
            exported_records.append({
                "stem": stem,
                "tile_base": tile_base,
                "img_bgr": img_bgr,
                "label_lines": label_lines,
                "source": "relabel_studio",
            })
            covered_tile_basenames.add(tile_base)

    # ── 2. Collect Human-Verified QC Review Decisions ─────────────────────────
    print(f"🔍 Scanning S3: s3://{S3_BUCKET}/{REVIEW_MANIFEST_KEY} and decisions...")
    manifest = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=REVIEW_MANIFEST_KEY)["Body"].read())
    items_by_id = {it["id"]: it for it in manifest.get("items", [])}
    set_id = manifest.get("set_id", "final_20261026_1614")
    dec_keys = list_s3_keys(s3, f"{REVIEW_DECISIONS_PREFIX}{set_id}/", ".json")
    print(f"📋 Found {len(dec_keys)} QC review decisions in set '{set_id}'.")

    blur_rows = []
    qc_added_pos = 0
    qc_added_neg = 0
    qc_skipped_blur = 0
    qc_skipped_uncorrected_poor = 0

    for dk in sorted(dec_keys):
        d = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=dk)["Body"].read())
        tid = d.get("item_id")
        it = items_by_id.get(tid)
        if not it:
            continue

        tile_key = it["tile_key"]
        det_key = it["det_key"]
        tile_base = os.path.splitext(os.path.basename(tile_key))[0]
        slide_name = it.get("slide_name", "")
        cytotype = it.get("cytotype", "unknown")
        verdict = d.get("verdict", "good")
        image_issue = bool(d.get("image_issue", False))

        raw_bytes = download_image_bytes(s3, [tile_key])
        if not raw_bytes:
            continue
        img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        try:
            det = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=det_key)["Body"].read())
        except Exception:
            det = {"boxes": [], "masks_xyn": []}

        boxes = det.get("boxes", [])
        masks_xyn = det.get("masks_xyn", [])
        focus_score = compute_grain_focus_score(img_bgr, boxes)
        is_blurred = image_issue or (len(boxes) > 1 and focus_score < MIN_BOX_LAPLACIAN_VAR and verdict == "poor")

        blur_rows.append({
            "item_id": tid,
            "slide_name": slide_name,
            "tile_base": tile_base,
            "cytotype": cytotype,
            "verdict": verdict,
            "image_issue": image_issue,
            "grain_focus_lap_var": round(focus_score, 3),
            "is_blurred_filtered": is_blurred,
            "n_pred": len(boxes),
            "comment": d.get("comment", ""),
        })

        if is_blurred:
            qc_skipped_blur += 1
            continue

        if not args.include_qc_benchmark:
            continue

        if tile_base in covered_tile_basenames:
            continue

        grains_override = d.get("grains", {})
        missed_list = d.get("missed", [])
        custom_list = d.get("custom_grains", [])

        # If verdict == 'poor' and the user did not mark individual grain overrides/missed/custom,
        # skip it so unedited errors on poor tiles aren't exported as ground truth.
        if verdict == "poor" and not grains_override and not missed_list and not custom_list:
            qc_skipped_uncorrected_poor += 1
            continue

        h, w = img_bgr.shape[:2]
        bg_color = tuple(int(x) for x in np.median(img_bgr.reshape(-1, 3), axis=0))
        img_clean = img_bgr.copy()
        label_lines = []

        for idx, b in enumerate(boxes):
            x1, y1, x2, y2, conf, cls_id = b[0], b[1], b[2], b[3], float(b[4]), int(b[5])
            act = grains_override.get(str(idx), "ok")
            poly_norm = masks_xyn[idx] if idx < len(masks_xyn) else []

            if act == "fp":
                # Soft-mask FP region with tile median background color
                if len(poly_norm) >= 3:
                    pts = np.array([[int(pt[0] * w), int(pt[1] * h)] for pt in poly_norm], dtype=np.int32)
                    cv2.fillPoly(img_clean, [pts], bg_color)
                else:
                    cv2.rectangle(img_clean, (max(0, int(x1)-2), max(0, int(y1)-2)),
                                  (min(w, int(x2)+2), min(h, int(y2)+2)), bg_color, -1)
                continue

            final_cid = RECLASS_MAP.get(act, cls_id)
            if len(poly_norm) >= 3:
                coords = []
                for pt in poly_norm:
                    coords.append(f"{min(1.0, max(0.0, float(pt[0]))):.6f}")
                    coords.append(f"{min(1.0, max(0.0, float(pt[1]))):.6f}")
            else:
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                rad = 0.25 * ((x2 - x1) + (y2 - y1))
                coords = make_circle_polygon_norm(cx, cy, rad, w=w, h=h)

            label_lines.append((final_cid, f"{final_cid} " + " ".join(coords)))

        # Add clicked missed grains (typically non_viable shrunken grains ~9.5 px radius)
        for ms in missed_list:
            mcid = RECLASS_MAP.get(ms.get("cls", "non_viable"), 1)
            mx, my = float(ms.get("x", 320.0)), float(ms.get("y", 320.0))
            rad = 9.5 if mcid == 1 else 12.5
            coords = make_circle_polygon_norm(mx, my, rad, w=w, h=h)
            label_lines.append((mcid, f"{mcid} " + " ".join(coords)))

        # Add custom polygon grains if any
        for cg in custom_list:
            ccid = RECLASS_MAP.get(cg.get("cls", "viable"), 0)
            cpoly = cg.get("polygon", cg.get("points", []))
            if len(cpoly) >= 3:
                coords = []
                for pt in cpoly:
                    px = pt["x"] if isinstance(pt, dict) else pt[0]
                    py = pt["y"] if isinstance(pt, dict) else pt[1]
                    coords.append(f"{min(1.0, max(0.0, float(px) / float(w))):.6f}")
                    coords.append(f"{min(1.0, max(0.0, float(py) / float(h))):.6f}")
                label_lines.append((ccid, f"{ccid} " + " ".join(coords)))

        exported_records.append({
            "stem": f"qc__{slide_name}__{tile_base}",
            "tile_base": tile_base,
            "img_bgr": img_clean,
            "label_lines": label_lines,
            "source": "qc_review",
        })
        covered_tile_basenames.add(tile_base)
        if label_lines:
            qc_added_pos += 1
        else:
            qc_added_neg += 1

    # Save blur report CSV
    if args.blur_report_out and blur_rows:
        import pandas as pd
        os.makedirs(os.path.dirname(args.blur_report_out), exist_ok=True)
        pd.DataFrame(blur_rows).to_csv(args.blur_report_out, index=False)
        print(f"🧹 Saved blur filter report ({qc_skipped_blur} blurred tiles filtered out) → {args.blur_report_out}")

    # ── 3. Shuffle, Split, and Write YOLO Dataset ─────────────────────────────
    random.shuffle(exported_records)
    n_val = max(1, int(len(exported_records) * args.val_split))
    val_records = exported_records[:n_val]
    train_records = exported_records[n_val:]

    class_counts = defaultdict(int)
    for split, recs in [("train", train_records), ("val", val_records)]:
        for rec in recs:
            stem = rec["stem"]
            img_dest = out_root / "images" / split / f"{stem}.jpg"
            lbl_dest = out_root / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(img_dest), rec["img_bgr"], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            lines = [line for _, line in rec["label_lines"]]
            for cid, _ in rec["label_lines"]:
                class_counts[cid] += 1
            lbl_dest.write_text(("\n".join(lines) + "\n") if lines else "")

    # Write data.yaml (Note: train_model.py expects names as a list for _build_class_remap)
    names_list = ["viable", "non_viable", "intermediate"]
    if args.include_debris:
        names_list.append("debris")

    data_cfg = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names_list),
        "names": names_list,
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_cfg, f, sort_keys=False)

    total_grains = sum(class_counts.values())
    saved_tiles = len(exported_records)

    print("\n" + "═" * 64)
    print("  POLLEN VIABILITY ADDITIVE YOLO DATASET EXPORT COMPLETE")
    print("═" * 64)
    print(f"  Output folder           : {out_root.resolve()}")
    print(f"  Total tiles exported    : {saved_tiles} ({len(train_records)} train / {len(val_records)} val)")
    print(f"    - Relabel Studio      : {sum(1 for r in exported_records if r['source'] == 'relabel_studio')}")
    print(f"    - QC Review Positives : {qc_added_pos}")
    print(f"    - QC Review Hard Negs : {qc_added_neg}")
    print(f"    - Blurred Filtered Out: {qc_skipped_blur}")
    print(f"    - Unedited Poor Skips : {qc_skipped_uncorrected_poor}")
    print(f"  Total grain annotations : {total_grains:,}")
    for cid, cnt in sorted(class_counts.items()):
        pct = 100 * cnt / max(1, total_grains)
        print(f"    Class {cid} ({CLASS_NAMES.get(cid, '?'):<12}): {cnt:>6,} ({pct:.1f}%)")
    print("═" * 64)

    if args.upload_staged:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_base = ROOT / "datasets" / f"pollen_viability_qc_additive_{ts}"
        zip_file = shutil.make_archive(str(zip_base), "zip", root_dir=str(out_root))
        s3_key = f"{STAGED_AREA_PREFIX}{ os.path.basename(zip_file) }"
        print(f"☁️ Uploading {os.path.basename(zip_file)} ({os.path.getsize(zip_file):,} bytes) → s3://{S3_BUCKET}/{s3_key}...")
        s3.upload_file(zip_file, S3_BUCKET, s3_key)
        print("✅ Uploaded additive training archive to S3 staged_area/!")


if __name__ == "__main__":
    main()

