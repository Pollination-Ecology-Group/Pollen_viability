#!/usr/bin/env python3
"""
run_detections_s3.py
────────────────────
Run YOLO detection on all tiles stored in S3 and write compact JSON
detection files (one per tile) back to S3.

This is run ONCE locally (or on the K8s cluster with GPU), not on
Streamlit Cloud.  The Streamlit app then just loads the JSON files
and visualises the pre-computed results — no PyTorch needed at runtime.

Detection JSON schema (saved as  <tile_stem>_det.json  next to the tile):
{
  "boxes": [
    [x1, y1, x2, y2, conf, cls_id],   # pixel coords, float/int
    ...
  ],
  "masks_xyn": [
    [[x_norm, y_norm], ...],           # normalised polygon vertices per box
    ...
  ]
}

Usage
─────
  # Activate the project venv that has ultralytics installed:
  source .venv/bin/activate

  # Run with environment variables (or a .env file):
  python src/run_detections_s3.py

  # Override the S3 prefix of tiles to process:
  TILE_PREFIX="Ostatni/Pollen_viability/tiles_640/" python src/run_detections_s3.py

  # Force re-run even if _det.json already exists:
  FORCE_REDETECT=1 python src/run_detections_s3.py

Environment variables (falls back to .env / Streamlit secrets):
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
  S3_ENDPOINT (default: https://s3.cl4.du.cesnet.cz),
  S3_BUCKET   (default: bucket)
  MODEL_PATH  (default: best.pt)
  CONF        (default: 0.25)
  IOU         (default: 0.70)
  MAX_DIM     (default: 130)   – max box side in px; larger = multi-grain false detect
  WORKERS     (default: 4)     – parallel S3 upload workers for JSON results
  FORCE_REDETECT (default: 0) – set to 1 to overwrite existing _det.json files
"""

import os, json, io, concurrent.futures
from pathlib import Path

import boto3
from botocore.client import Config
import cv2
import numpy as np
from PIL import Image

# ── load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── config ────────────────────────────────────────────────────────────────────
S3_ENDPOINT    = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_BUCKET      = os.environ.get("S3_BUCKET", "bucket")
TILE_PREFIX    = os.environ.get("TILE_PREFIX", "Ostatni/Pollen_viability/tiles_640/")
MODEL_PATH     = os.environ.get("MODEL_PATH", "best.pt")
CONF           = float(os.environ.get("CONF", "0.25"))
IOU            = float(os.environ.get("IOU", "0.70"))
MAX_DIM        = int(os.environ.get("MAX_DIM", "130"))
WORKERS        = int(os.environ.get("WORKERS", "4"))
FORCE          = os.environ.get("FORCE_REDETECT", "0") == "1"

VALID_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')

# ── helpers ───────────────────────────────────────────────────────────────────

def make_s3():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        config=Config(signature_version='s3v4')
    )


def list_tile_keys(s3):
    """Return list of all tile S3 keys under TILE_PREFIX."""
    keys = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=TILE_PREFIX):
        for obj in page.get('Contents', []):
            k = obj['Key']
            if k.lower().endswith(VALID_EXT) and not k.endswith('_det.json'):
                keys.append(k)
    return keys


def det_key_for(tile_key: str) -> str:
    """Return the S3 key for the detection JSON of a given tile key."""
    stem = tile_key.rsplit('.', 1)[0]
    return stem + '_det.json'


def key_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False


def download_image(s3, key: str) -> np.ndarray | None:
    """Download a tile from S3 and return it as a BGR numpy array."""
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        data = resp['Body'].read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  ⚠️  Could not download {key}: {e}")
        return None


def build_detection_json(results_list, cv_img: np.ndarray) -> dict:
    """
    Convert YOLO result object into a compact, serialisable dict.
    """
    if not results_list or results_list[0].boxes is None:
        return {"boxes": [], "masks_xyn": []}

    res = results_list[0]
    img_h, img_w = cv_img.shape[:2]

    boxes_out = []
    masks_out = []

    boxes = res.boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
        conf  = float(boxes.conf[i])
        cls_id = int(boxes.cls[i])

        # Skip oversized boxes (multi-grain false detections)
        if (x2 - x1) > MAX_DIM or (y2 - y1) > MAX_DIM:
            continue

        boxes_out.append([round(x1), round(y1), round(x2), round(y2), round(conf, 4), cls_id])

        # Normalised polygon (may be absent for bbox-only models)
        if res.masks is not None and i < len(res.masks.xyn):
            poly = [[round(float(pt[0]), 5), round(float(pt[1]), 5)]
                    for pt in res.masks.xyn[i]]
        else:
            # Derive a rectangle polygon from the box
            nw, nh = img_w or 1, img_h or 1
            poly = [
                [x1 / nw, y1 / nh], [x2 / nw, y1 / nh],
                [x2 / nw, y2 / nh], [x1 / nw, y2 / nh]
            ]
        masks_out.append(poly)

    return {"boxes": boxes_out, "masks_xyn": masks_out}


def upload_json(s3, key: str, data: dict):
    body = json.dumps(data, separators=(',', ':')).encode('utf-8')
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body,
                  ContentType='application/json')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("🌸 Pollen Detection → S3 JSON Writer")
    print(f"   Model    : {MODEL_PATH}")
    print(f"   S3 bucket: {S3_BUCKET}")
    print(f"   Prefix   : {TILE_PREFIX}")
    print(f"   conf={CONF}  iou={IOU}  max_dim={MAX_DIM}px")
    print(f"   Force re-detect: {FORCE}")
    print()

    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise SystemExit("❌ AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set.")

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"❌ Model file not found: {MODEL_PATH}\n"
                         "   Set MODEL_PATH env var to your best.pt location.")

    from ultralytics import YOLO
    import torch

    model = YOLO(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Running inference on: {device}\n")

    s3 = make_s3()

    print("📋 Listing tiles from S3…")
    all_keys = list_tile_keys(s3)
    print(f"   Found {len(all_keys)} tile(s).\n")

    skipped = 0
    processed = 0
    errors = 0

    def process_one(tile_key):
        nonlocal skipped, processed, errors
        dkey = det_key_for(tile_key)

        if not FORCE and key_exists(s3, dkey):
            skipped += 1
            return

        cv_img = download_image(s3, tile_key)
        if cv_img is None:
            errors += 1
            return

        try:
            with torch.no_grad():
                results = model(
                    [cv_img],
                    conf=CONF, iou=IOU,
                    agnostic_nms=False, verbose=False,
                    device=device
                )
        except Exception as e:
            print(f"  ⚠️  Inference failed for {tile_key}: {e}")
            errors += 1
            return

        det = build_detection_json(results, cv_img)
        try:
            upload_json(s3, dkey, det)
        except Exception as e:
            print(f"  ⚠️  Upload failed for {dkey}: {e}")
            errors += 1
            return

        n = len(det["boxes"])
        print(f"  ✅  {os.path.basename(tile_key):50s}  →  {n} grain(s)")
        processed += 1

    # Process sequentially (inference is already parallelised internally by YOLO)
    # Use threads only for S3 I/O overlap; keep inference serial to avoid GPU OOM.
    for i, key in enumerate(all_keys, 1):
        print(f"[{i:4d}/{len(all_keys)}] ", end="", flush=True)
        process_one(key)

    print()
    print("─" * 60)
    print(f"✅ Done.  Processed: {processed}  Skipped: {skipped}  Errors: {errors}")
    print(f"   Detection JSON files saved as  <tile_stem>_det.json  in S3.")


if __name__ == "__main__":
    main()
