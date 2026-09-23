#!/usr/bin/env python3
"""
run_detections_s3.py  — Batch YOLO detection → S3 JSON writer
══════════════════════════════════════════════════════════════
Run ONCE locally (GPU recommended) or on K8s cluster. Writes a
compact _det.json file alongside each tile in S3. The Streamlit app
then loads these JSONs at display time — no PyTorch needed at runtime.

Speed benchmarks (640×640 tiles):
  CPU only  (14 cores)  ~1-2  tiles/s  → 199 k tiles ≈ 30-55 h  ❌ too slow
  GPU T4/V100           ~50-100 tiles/s → 199 k tiles ≈ 30-60 min ✅

Usage
─────
  # With local GPU / local machine:
  source .venv/bin/activate
  MODEL_PATH=best.pt BATCH_SIZE=32 python src/run_detections_s3.py

  # Force re-detect even if _det.json already exists:
  FORCE_REDETECT=1 MODEL_PATH=best.pt python src/run_detections_s3.py

  # On K8s cluster (GPU node): deploy k8s/pollen-detect-job.yaml
  #   → the job downloads best.pt from S3, then runs this script.

Detection JSON schema (saved as  <tile_stem>_det.json  next to the tile):
{
  "boxes":     [[x1, y1, x2, y2, conf, cls_id], ...],  # pixels
  "masks_xyn": [[[xn, yn], ...], ...]                   # normalised polygon per box
}

Environment variables:
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  S3_ENDPOINT   (default: https://s3.cl4.du.cesnet.cz)
  S3_BUCKET     (default: bucket)
  TILE_PREFIX   (default: Ostatni/Pollen_viability/tiles_640/)
  MODEL_PATH    (default: best.pt)
  S3_MODEL_KEY  (optional) — download model from S3 if MODEL_PATH not found
  CONF          (default: 0.25)
  IOU           (default: 0.70)
  MAX_DIM       (default: 130)  — max box side px; larger = multi-grain false det
  BATCH_SIZE    (default: 32)   — images per YOLO inference call
  FETCH_WORKERS (default: 16)   — parallel S3 download threads
  FORCE_REDETECT (default: 0)  — set 1 to overwrite existing _det.json files
"""

import os, json, io, sys, time, concurrent.futures
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
S3_ENDPOINT    = os.environ.get("S3_ENDPOINT",   "https://s3.cl4.du.cesnet.cz")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_BUCKET      = os.environ.get("S3_BUCKET",     "bucket")
TILE_PREFIX    = os.environ.get("TILE_PREFIX",   "Ostatni/Pollen_viability/tiles_640/")
MODEL_PATH     = os.environ.get("MODEL_PATH",    "best.pt")
S3_MODEL_KEY   = os.environ.get("S3_MODEL_KEY",  "Ostatni/Pollen_viability/trained_models/pollen_train_20260923_1457/weights/best.pt")
CONF           = float(os.environ.get("CONF",    "0.25"))
IOU            = float(os.environ.get("IOU",     "0.70"))
MAX_DIM        = int(os.environ.get("MAX_DIM",   "130"))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE","32"))
FETCH_WORKERS  = int(os.environ.get("FETCH_WORKERS", "16"))
FORCE          = os.environ.get("FORCE_REDETECT", "0") == "1"

VALID_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')


# ── helpers ───────────────────────────────────────────────────────────────────

def make_s3():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        config=Config(
            signature_version='s3v4',
            max_pool_connections=FETCH_WORKERS + 4,
        )
    )


def list_tile_keys(s3):
    """Return (tile_keys, existing_det_keys) from the tiles_640 prefix."""
    tile_keys = []
    det_keys  = set()
    paginator = s3.get_paginator('list_objects_v2')
    print("📋 Listing S3 objects (this may take a moment for 200k files)…")
    t0 = time.time()
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=TILE_PREFIX):
        for obj in page.get('Contents', []):
            k = obj['Key']
            if k.endswith('_det.json'):
                det_keys.add(k)
            elif k.lower().endswith(VALID_EXT):
                tile_keys.append(k)
    elapsed = time.time() - t0
    print(f"   Found {len(tile_keys):,} tiles, {len(det_keys):,} existing _det.json  ({elapsed:.1f}s)")
    return tile_keys, det_keys


def det_key_for(tile_key: str) -> str:
    return tile_key.rsplit('.', 1)[0] + '_det.json'


def download_tile(args):
    """Download one tile; return (key, np_bgr) or (key, None) on error."""
    s3, key = args
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        data = resp['Body'].read()
        pil  = Image.open(io.BytesIO(data)).convert("RGB")
        img  = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return key, img
    except Exception as e:
        print(f"  ⚠️  Download failed {key}: {e}", flush=True)
        return key, None


def build_det_json(result, cv_img: np.ndarray) -> dict:
    """Convert one YOLO result → compact dict suitable for JSON serialisation."""
    if result.boxes is None or len(result.boxes) == 0:
        return {"boxes": [], "masks_xyn": []}

    img_h, img_w = cv_img.shape[:2]
    boxes_out = []
    masks_out = []

    for i in range(len(result.boxes)):
        x1, y1, x2, y2 = [float(v) for v in result.boxes.xyxy[i].tolist()]
        conf   = float(result.boxes.conf[i])
        cls_id = int(result.boxes.cls[i])

        # Skip oversized detections (multi-grain false positives)
        if (x2 - x1) > MAX_DIM or (y2 - y1) > MAX_DIM:
            continue

        boxes_out.append([round(x1), round(y1), round(x2), round(y2),
                          round(conf, 4), cls_id])

        # Normalised polygon mask (segmentation model) or rectangle fallback
        if result.masks is not None and i < len(result.masks.xyn):
            poly = [[round(float(pt[0]), 5), round(float(pt[1]), 5)]
                    for pt in result.masks.xyn[i]]
        else:
            nw, nh = img_w or 1, img_h or 1
            poly = [
                [x1/nw, y1/nh], [x2/nw, y1/nh],
                [x2/nw, y2/nh], [x1/nw, y2/nh]
            ]
        masks_out.append(poly)

    return {"boxes": boxes_out, "masks_xyn": masks_out}


def upload_jsons(s3, items):
    """Parallel upload of {key: dict} items to S3."""
    def put_one(kv):
        k, d = kv
        body = json.dumps(d, separators=(',', ':')).encode('utf-8')
        s3.put_object(Bucket=S3_BUCKET, Key=k, Body=body,
                      ContentType='application/json')
    with concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        list(ex.map(put_one, items.items()))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise SystemExit("❌ AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set.")

    # Download model from S3 if not present locally
    if not os.path.exists(MODEL_PATH) and S3_MODEL_KEY:
        print(f"⬇️  Downloading model from S3: {S3_MODEL_KEY}")
        s3_tmp = make_s3()
        s3_tmp.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
        print(f"   Saved to {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1e6:.0f} MB)")

    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"❌ Model not found: {MODEL_PATH}")

    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🌸 Pollen Detection → S3 JSON Writer")
    print(f"   Model    : {MODEL_PATH}  ({os.path.getsize(MODEL_PATH)/1e6:.0f} MB)")
    print(f"   Device   : {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"   Batch    : {BATCH_SIZE}  conf={CONF}  iou={IOU}  max_dim={MAX_DIM}px")
    print(f"   S3 bucket: {S3_BUCKET}")
    print(f"   Prefix   : {TILE_PREFIX}")
    print(f"   Force    : {FORCE}")
    print()

    model = YOLO(MODEL_PATH)

    s3 = make_s3()
    all_tile_keys, existing_det_keys = list_tile_keys(s3)

    # Filter out already-processed tiles (unless FORCE)
    if FORCE:
        to_process = all_tile_keys
    else:
        to_process = [k for k in all_tile_keys if det_key_for(k) not in existing_det_keys]

    total   = len(to_process)
    skipped = len(all_tile_keys) - total
    print(f"\n🚀 Processing {total:,} tiles  (skipping {skipped:,} already done)\n")

    if total == 0:
        print("✅ All tiles already have detection JSONs. Done!")
        return

    processed = 0
    errors    = 0
    t_start   = time.time()

    # Process in chunks: download BATCH_SIZE tiles in parallel, then infer
    for chunk_start in range(0, total, BATCH_SIZE):
        chunk_keys = to_process[chunk_start : chunk_start + BATCH_SIZE]

        # ── Parallel download ────────────────────────────────────────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
            results_dl = list(ex.map(download_tile, [(s3, k) for k in chunk_keys]))

        valid_keys = [k   for k, img in results_dl if img is not None]
        valid_imgs = [img for k, img in results_dl if img is not None]
        errors    += len(chunk_keys) - len(valid_keys)

        if not valid_imgs:
            continue

        # ── Batch YOLO inference ─────────────────────────────────────────────
        try:
            with torch.no_grad():
                batch_results = model(
                    valid_imgs, conf=CONF, iou=IOU,
                    agnostic_nms=False, verbose=False, device=device
                )
        except Exception as e:
            print(f"  ⚠️  Inference failed on chunk [{chunk_start}]: {e}", flush=True)
            errors += len(valid_keys)
            continue

        # ── Build JSON payload ───────────────────────────────────────────────
        jsons_to_upload = {}
        for i, (key, cv_img) in enumerate(zip(valid_keys, valid_imgs)):
            det = build_det_json(batch_results[i], cv_img)
            jsons_to_upload[det_key_for(key)] = det

        # ── Parallel upload ──────────────────────────────────────────────────
        try:
            upload_jsons(s3, jsons_to_upload)
        except Exception as e:
            print(f"  ⚠️  Upload failed on chunk [{chunk_start}]: {e}", flush=True)
            errors += len(valid_keys)
            continue

        processed += len(valid_keys)

        # ── Progress report ──────────────────────────────────────────────────
        elapsed = time.time() - t_start
        rate    = processed / elapsed if elapsed > 0 else 0
        eta_s   = (total - processed) / rate if rate > 0 else 0
        eta_str = f"{eta_s/3600:.1f}h" if eta_s > 3600 else f"{eta_s/60:.0f}min"
        n_grains = sum(len(d["boxes"]) for d in jsons_to_upload.values())
        print(
            f"  [{processed:6,}/{total:,}]  "
            f"{rate:.1f} tiles/s  ETA {eta_str}  "
            f"last batch: {n_grains} grains",
            flush=True
        )

    elapsed_total = time.time() - t_start
    print()
    print("─" * 65)
    print(f"✅ Done.  Processed: {processed:,}  Errors: {errors}  "
          f"Total time: {elapsed_total/60:.0f} min")
    print(f"   Detection JSONs saved as  <tile_stem>_det.json  in S3.")


if __name__ == "__main__":
    main()
