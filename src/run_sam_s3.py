#!/usr/bin/env python3
"""
run_sam_s3.py  — FastSAM grain detection → S3 JSON writer
══════════════════════════════════════════════════════════════
Runs FastSAM (class-agnostic) on a *targeted* subset of tiles and writes
compact _det.json files alongside each tile in S3.  Designed to replace
the poor YOLO detections for high non-viable samples so that Pinder's
Swipe Mode can be used to manually classify each SAM-detected grain crop.

Typical usage
─────────────
  # Target top NV samples (for non-viable grains) + N random for viable:
  source .venv/bin/activate
  python src/run_sam_s3.py \
      --target-samples 1-6-J,7-9-F,6-1-F \
      --viable-limit 200

  # Force overwrite even if _det.json already exists:
  python src/run_sam_s3.py --target-samples 1-6-J --force

Environment variables
─────────────────────
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  S3_ENDPOINT   (default: https://s3.cl4.du.cesnet.cz)
  S3_BUCKET     (default: bucket)
  TILE_PREFIX   (default: Ostatni/Pollen_viability/tiles_640/)
  MODEL_PATH    (default: FastSAM-s.pt)
  CONF          (default: 0.30)
  IOU           (default: 0.50)
  MIN_AREA      (default: 400)   — min grain area in px² (filters noise)
  MAX_DIM       (default: 150)   — max bounding-box side in px (filters clumps)
  BATCH_SIZE    (default: 16)    — images per SAM inference call
  FETCH_WORKERS (default: 12)    — parallel S3 download threads
"""

import os, json, io, sys, time, random, argparse, concurrent.futures
from pathlib import Path

import boto3
from botocore.client import Config
import cv2
import numpy as np
from PIL import Image

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
MODEL_PATH     = os.environ.get("MODEL_PATH",    "FastSAM-s.pt")
CONF           = float(os.environ.get("CONF",    "0.30"))
IOU            = float(os.environ.get("IOU",     "0.50"))
MIN_AREA       = int(os.environ.get("MIN_AREA",  "400"))   # px² — filter specks
MAX_DIM        = int(os.environ.get("MAX_DIM",   "150"))   # px  — filter clumps
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE","16"))
FETCH_WORKERS  = int(os.environ.get("FETCH_WORKERS", "12"))

VALID_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')


# ── helpers ───────────────────────────────────────────────────────────────────

def make_s3():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        config=Config(signature_version='s3v4',
                      max_pool_connections=FETCH_WORKERS + 4),
    )


def folder_matches_sample(folder_prefix: str, sample_ids: list[str]) -> bool:
    """Return True if any sample ID string appears in the folder name."""
    folder_name = folder_prefix.rstrip("/").split("/")[-1]
    return any(sid in folder_name for sid in sample_ids)


def list_tile_folders(s3) -> dict[str, list[str]]:
    """Return {folder_prefix: [tile_keys]} for all subfolders in TILE_PREFIX."""
    print("📋 Listing S3 tile folders …")
    t0 = time.time()
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=TILE_PREFIX, Delimiter="/")
    folders = [p['Prefix'] for p in resp.get('CommonPrefixes', [])]
    print(f"   Found {len(folders)} CZI image folders  ({time.time()-t0:.1f}s)")
    return folders


def fetch_folder_keys(s3, folder_prefix: str) -> list[str]:
    """List all tile image keys in a folder (no pagination needed — folders are small)."""
    keys = []
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder_prefix)
    for obj in resp.get('Contents', []):
        k = obj['Key']
        if k.lower().endswith(VALID_EXT):
            keys.append(k)
    return keys


def det_key_for(tile_key: str) -> str:
    return tile_key.rsplit('.', 1)[0] + '_det.json'


def download_tile(args):
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
    """Convert one SAM result → compact _det.json dict.

    SAM is class-agnostic so all detections get cls_id=0 (placeholder).
    Pinder Swipe Mode will let the user relabel to 0/1/2.
    """
    img_h, img_w = cv_img.shape[:2]

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return {"boxes": [], "masks_xyn": []}

    boxes_out = []
    masks_out = []

    for i in range(len(result.boxes)):
        x1, y1, x2, y2 = [float(v) for v in result.boxes.xyxy[i].tolist()]
        conf   = float(result.boxes.conf[i])
        w_box  = x2 - x1
        h_box  = y2 - y1

        # Filter: too small (noise/specks) or too large (multi-grain clumps)
        area = w_box * h_box
        if area < MIN_AREA:
            continue
        if w_box > MAX_DIM or h_box > MAX_DIM:
            continue

        boxes_out.append([round(x1), round(y1), round(x2), round(y2),
                          round(conf, 4), 0])   # cls_id=0 placeholder

        if i < len(result.masks.xyn):
            poly = [[round(float(pt[0]), 5), round(float(pt[1]), 5)]
                    for pt in result.masks.xyn[i]]
        else:
            nw, nh = img_w or 1, img_h or 1
            poly = [
                [x1/nw, y1/nh], [x2/nw, y1/nh],
                [x2/nw, y2/nh], [x1/nw, y2/nh],
            ]
        masks_out.append(poly)

    return {"boxes": boxes_out, "masks_xyn": masks_out}


def upload_jsons(s3, items: dict):
    def put_one(kv):
        k, d = kv
        body = json.dumps(d, separators=(',', ':')).encode('utf-8')
        s3.put_object(Bucket=S3_BUCKET, Key=k, Body=body,
                      ContentType='application/json')
    with concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        list(ex.map(put_one, items.items()))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run FastSAM on targeted S3 tile subsets → write _det.json")
    parser.add_argument("--target-samples", default="1-6-J,7-9-F,6-1-F",
                        help="Comma-separated sample IDs to target for NV tiles "
                             "(e.g. '1-6-J,7-9-F,6-1-F')")
    parser.add_argument("--viable-limit", type=int, default=200,
                        help="Number of random non-targeted tiles to also process "
                             "(provides viable grain variety)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing _det.json files")
    parser.add_argument("--model", default=MODEL_PATH,
                        help=f"Path to FastSAM weights (default: {MODEL_PATH})")
    args = parser.parse_args()

    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise SystemExit("❌ AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set.")

    if not os.path.exists(args.model):
        raise SystemExit(f"❌ Model not found: {args.model}  "
                         f"(expected FastSAM-s.pt in repo root)")

    target_ids = [s.strip() for s in args.target_samples.split(",") if s.strip()]
    print(f"\n🌸 FastSAM → S3 JSON Writer")
    print(f"   Model         : {args.model}")
    print(f"   Target samples: {target_ids}")
    print(f"   Viable limit  : {args.viable_limit} random tiles")
    print(f"   Force overwrite: {args.force}")
    print(f"   conf={CONF}  iou={IOU}  min_area={MIN_AREA}px²  max_dim={MAX_DIM}px\n")

    import torch
    from ultralytics import FastSAM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}" +
          (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else
           "  ⚠️  No GPU — will be slow for large batches"))

    model = FastSAM(args.model)

    s3 = make_s3()

    # ── Select tile keys ──────────────────────────────────────────────────────
    all_folders = list_tile_folders(s3)
    target_folders = [f for f in all_folders if folder_matches_sample(f, target_ids)]
    other_folders  = [f for f in all_folders if f not in target_folders]

    print(f"\n🎯 Targeted folders  : {len(target_folders)}")
    for f in target_folders:
        print(f"   {f.rstrip('/').split('/')[-1]}")

    # Collect all tile keys from targeted folders
    target_keys = []
    for folder in target_folders:
        target_keys.extend(fetch_folder_keys(s3, folder))
    print(f"\n   → {len(target_keys):,} tile images in targeted folders")

    # Random sample from non-targeted folders for viable variety
    viable_keys = []
    if args.viable_limit > 0:
        random.shuffle(other_folders)
        for folder in other_folders:
            if len(viable_keys) >= args.viable_limit:
                break
            keys = fetch_folder_keys(s3, folder)
            viable_keys.extend(keys[:max(1, args.viable_limit // len(other_folders) + 1)])
        viable_keys = viable_keys[:args.viable_limit]
        print(f"   → {len(viable_keys):,} random viable tiles from {len(other_folders)} other folders")

    all_keys = target_keys + viable_keys

    # Filter out already-processed unless --force
    if not args.force:
        print(f"\n🔍 Checking for existing _det.json …")
        existing = set()
        for folder in target_folders + list({k.rsplit('/', 2)[0] + '/' for k in viable_keys}):
            resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder)
            for obj in resp.get('Contents', []):
                if obj['Key'].endswith('_det.json'):
                    existing.add(obj['Key'])
        to_process = [k for k in all_keys if det_key_for(k) not in existing]
        print(f"   Skipping {len(all_keys) - len(to_process):,} already done  "
              f"→  processing {len(to_process):,}")
    else:
        to_process = all_keys
        print(f"\n🚀 Force mode — processing all {len(to_process):,} tiles")

    if not to_process:
        print("✅ All tiles already have SAM _det.json. Done!")
        return

    # ── Process in batches ────────────────────────────────────────────────────
    total     = len(to_process)
    processed = 0
    grains    = 0
    errors    = 0
    t_start   = time.time()

    for chunk_start in range(0, total, BATCH_SIZE):
        chunk_keys = to_process[chunk_start: chunk_start + BATCH_SIZE]

        # Parallel download
        with concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
            results_dl = list(ex.map(download_tile, [(s3, k) for k in chunk_keys]))

        valid_keys = [k   for k, img in results_dl if img is not None]
        valid_imgs = [img for k, img in results_dl if img is not None]
        errors += len(chunk_keys) - len(valid_keys)

        if not valid_imgs:
            continue

        # FastSAM inference
        try:
            with torch.no_grad():
                batch_results = model(
                    valid_imgs,
                    conf=CONF, iou=IOU,
                    verbose=False, device=device,
                    retina_masks=True,
                )
        except Exception as e:
            print(f"  ⚠️  SAM inference failed on chunk [{chunk_start}]: {e}", flush=True)
            errors += len(valid_keys)
            continue

        # Build & upload JSONs
        jsons_to_upload = {}
        chunk_grains = 0
        for i, (key, cv_img) in enumerate(zip(valid_keys, valid_imgs)):
            det = build_det_json(batch_results[i], cv_img)
            jsons_to_upload[det_key_for(key)] = det
            chunk_grains += len(det["boxes"])

        try:
            upload_jsons(s3, jsons_to_upload)
        except Exception as e:
            print(f"  ⚠️  Upload failed on chunk [{chunk_start}]: {e}", flush=True)
            errors += len(valid_keys)
            continue

        processed  += len(valid_keys)
        grains     += chunk_grains
        elapsed     = time.time() - t_start
        rate        = processed / elapsed if elapsed > 0 else 0
        eta_s       = (total - processed) / rate if rate > 0 else 0
        eta_str     = f"{eta_s/60:.0f}min" if eta_s > 60 else f"{eta_s:.0f}s"
        print(
            f"  [{processed:5,}/{total:,}]  {rate:.1f} tiles/s  ETA {eta_str}  "
            f"last batch: {chunk_grains} grains detected",
            flush=True,
        )

    elapsed_total = time.time() - t_start
    print()
    print("─" * 65)
    print(f"✅ Done.  Processed: {processed:,}  Errors: {errors}  "
          f"Total time: {elapsed_total/60:.1f} min")
    print(f"   Total SAM grains written: {grains:,}")
    print(f"\n👉 Next: open Pinder, use Source Image Filter to select")
    print(f"   [{', '.join(target_ids)}]  and use Swipe Mode to classify grains.")


if __name__ == "__main__":
    main()
