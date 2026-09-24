#!/usr/bin/env python3
"""
infer_on_czi.py — Run the trained pollen model on one CZI file end-to-end
═══════════════════════════════════════════════════════════════════════════
Downloads a CZI from S3, tiles it at 640×640 (same as preprocess_czi.py),
runs the YOLO segmentation model on every tile, and produces:

  output/<slide_name>/
    annotated_tiles/     ← every tile with coloured overlays
    mosaic.jpg           ← all tiles stitched back into the full image
    detections.json      ← per-tile grain count + class + confidence
    summary.txt          ← human-readable summary (viable/non-viable counts)

Usage:
  # Pick a specific CZI by its filename pattern:
  python src/infer_on_czi.py --pattern "27-3-B_AA025_n4x"

  # Or by full S3 key:
  python src/infer_on_czi.py --key "Ostatni/Pollen_viability/Source/20260724_0001_27-3-B_AA025_n4x_Pollen_viability.czi"

  # Adjust confidence (default 0.10 — lower than production because n4x tiles
  # were not in the original training set):
  python src/infer_on_czi.py --pattern "27-3-B_AA025_n4x" --conf 0.08
"""

import os, io, json, argparse, sys, time
import cv2, numpy as np
from pathlib import Path
from collections import defaultdict

import boto3
from botocore.client import Config
from PIL import Image

# ── config ────────────────────────────────────────────────────────────────────
S3_ENDPOINT  = os.environ.get('S3_ENDPOINT',  'https://s3.cl4.du.cesnet.cz')
AWS_ACCESS   = os.environ.get('AWS_ACCESS_KEY_ID',  '1Y920BKC0SAWPNDE8RD6')
AWS_SECRET   = os.environ.get('AWS_SECRET_ACCESS_KEY',
                               'SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD')
S3_BUCKET    = os.environ.get('S3_BUCKET', 'bucket')
S3_CZI_PFX  = 'Ostatni/Pollen_viability/Source'
MODEL_PATH   = os.environ.get('MODEL_PATH',
               str(Path(__file__).parent.parent / 'best.pt'))
S3_MODEL_KEY = 'Ostatni/Pollen_viability/trained_models/pollen_train_20260923_1457/weights/best.pt'

TILE_SIZE    = 640
OVERLAP      = 0.1          # 10% overlap — same as preprocess_czi.py
CLASS_MAP    = {0: ('viable',      (0,   200,  60)),
                1: ('non_viable',  (0,    50, 220)),
                2: ('intermediate',(0,   200, 255))}
STRIDE       = int(TILE_SIZE * (1 - OVERLAP))


def make_s3():
    return boto3.client('s3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS,
        aws_secret_access_key=AWS_SECRET,
        config=Config(signature_version='s3v4'))


def find_czi_key(s3, pattern):
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_CZI_PFX):
        for obj in page.get('Contents', []):
            if pattern.lower() in obj['Key'].lower() and obj['Key'].lower().endswith('.czi'):
                return obj['Key'], obj['Size']
    return None, None


def extract_image_from_czi(czi_path):
    import czifile
    with czifile.CziFile(czi_path) as czi:
        img_data = czi.asarray()
    img_data = np.squeeze(img_data)
    if img_data.ndim == 3 and img_data.shape[0] in [3, 4]:
        img_data = np.transpose(img_data, (1, 2, 0))
    if img_data.dtype == np.uint16:
        lo, hi = img_data.min(), img_data.max()
        img_data = ((img_data - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    if img_data.ndim == 2:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
    elif img_data.ndim == 3 and img_data.shape[2] == 3:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    return img_data


def tile_positions(h, w):
    """Return (y, x) start positions matching preprocess_czi.py logic."""
    y_starts = list(range(0, h - TILE_SIZE, STRIDE))
    if y_starts and y_starts[-1] + TILE_SIZE < h:
        y_starts.append(h - TILE_SIZE)
    if not y_starts:
        y_starts = [0]
    x_starts = list(range(0, w - TILE_SIZE, STRIDE))
    if x_starts and x_starts[-1] + TILE_SIZE < w:
        x_starts.append(w - TILE_SIZE)
    if not x_starts:
        x_starts = [0]
    return y_starts, x_starts


def draw_detections(bgr, boxes, confs, classes, masks=None):
    out = bgr.copy()
    h, w = out.shape[:2]
    for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
        label, color = CLASS_MAP.get(int(cls), (f'cls{cls}', (200, 200, 200)))
        x1, y1, x2, y2 = [int(v) for v in box]
        if masks is not None and i < len(masks):
            pts = (masks[i] * np.array([w, h])).astype(np.int32)
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
            cv2.polylines(out, [pts], True, color, 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"{label} {conf:.2f}"
        cv2.putText(out, tag, (x1 + 2, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def run(args):
    import torch
    from ultralytics import YOLO

    s3 = make_s3()

    # ── 1. Locate CZI ─────────────────────────────────────────────────────────
    if args.key:
        czi_key  = args.key
        czi_size = s3.head_object(Bucket=S3_BUCKET, Key=czi_key)['ContentLength']
    else:
        print(f"🔍 Searching for CZI matching '{args.pattern}'...")
        czi_key, czi_size = find_czi_key(s3, args.pattern)
        if not czi_key:
            sys.exit(f"❌  No CZI found matching pattern '{args.pattern}'")
    slide_name = Path(czi_key).stem
    print(f"📄 CZI: {slide_name}  ({czi_size/1e6:.0f} MB)")

    # ── 2. Download CZI ───────────────────────────────────────────────────────
    czi_local = f'/tmp/{slide_name}.czi'
    if not os.path.exists(czi_local):
        print(f"⬇️  Downloading ({czi_size/1e6:.0f} MB) — this may take a few minutes...")
        t0 = time.time()
        s3.download_file(S3_BUCKET, czi_key, czi_local)
        print(f"   Done in {time.time()-t0:.0f}s")
    else:
        print(f"   Using cached CZI at {czi_local}")

    # ── 3. Extract & tile ─────────────────────────────────────────────────────
    print("🔬 Extracting image from CZI...")
    img = extract_image_from_czi(czi_local)
    H, W = img.shape[:2]
    print(f"   Image size: {W}×{H} px")

    y_starts, x_starts = tile_positions(H, W)
    total_tiles = len(y_starts) * len(x_starts)
    print(f"   Tiling into {total_tiles} tiles ({len(x_starts)}×{len(y_starts)})")

    # ── 4. Load model ─────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"⬇️  Downloading model from S3...")
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    model  = YOLO(MODEL_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🤖 Model loaded  |  device: {device}  |  conf={args.conf}")

    # ── 5. Inference on all tiles ─────────────────────────────────────────────
    out_dir     = Path(f'output/{slide_name}')
    ann_dir     = out_dir / 'annotated_tiles'
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Full-res canvas for mosaic
    mosaic      = img.copy()
    detections  = []
    cls_totals  = defaultdict(int)
    total_grains = 0
    tiles_with_detections = 0

    print(f"\n🚀 Running inference on {total_tiles} tiles...")
    t0 = time.time()

    for tile_i, y in enumerate(y_starts):
        for x in x_starts:
            tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                pad = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                pad[:tile.shape[0], :tile.shape[1]] = tile
                tile = pad

            with torch.no_grad():
                results = model(tile, conf=args.conf, iou=0.5,
                                verbose=False, device=device)
            r = results[0]
            n = len(r.boxes) if r.boxes is not None else 0

            tile_dets = []
            if n > 0:
                tiles_with_detections += 1
                boxes   = r.boxes.xyxy.cpu().numpy()
                confs_v = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                masks_v = r.masks.xyn if r.masks is not None else None

                for i in range(n):
                    cls_id = int(classes[i])
                    cls_totals[cls_id] += 1
                    total_grains += 1
                    tile_dets.append({
                        'class': cls_id,
                        'class_name': CLASS_MAP.get(cls_id, (f'cls{cls_id}',))[0],
                        'confidence': float(confs_v[i]),
                        'bbox_tile': [float(v) for v in boxes[i]]
                    })

                # Draw on tile and paste back into mosaic
                masks_list = [masks_v[i] for i in range(n)] if masks_v is not None else None
                ann_tile   = draw_detections(tile, boxes, confs_v, classes, masks_list)

                # Paste annotated tile into mosaic
                ty2 = min(y + TILE_SIZE, H)
                tx2 = min(x + TILE_SIZE, W)
                mosaic[y:ty2, x:tx2] = ann_tile[:ty2-y, :tx2-x]

                # Save annotated tile
                cv2.imwrite(str(ann_dir / f'tile_{y}_{x}.jpg'), ann_tile)

            detections.append({'tile_y': y, 'tile_x': x, 'detections': tile_dets})

        # Progress
        done = (tile_i + 1) * len(x_starts)
        if done % max(1, total_tiles // 10) == 0 or done == total_tiles:
            elapsed = time.time() - t0
            pct     = done / total_tiles * 100
            eta     = elapsed / done * (total_tiles - done)
            print(f"  {pct:5.1f}%  ({done}/{total_tiles} tiles)  "
                  f"grains={total_grains}  ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0

    # ── 6. Save mosaic (scaled to max 4000px wide) ───────────────────────────
    scale = min(1.0, 4000 / W)
    if scale < 1.0:
        mosaic_small = cv2.resize(mosaic, (int(W*scale), int(H*scale)))
    else:
        mosaic_small = mosaic
    mosaic_path = str(out_dir / 'mosaic.jpg')
    cv2.imwrite(mosaic_path, mosaic_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"\n🖼️  Mosaic saved → {mosaic_path}")

    # ── 7. Save detections JSON ───────────────────────────────────────────────
    json_path = str(out_dir / 'detections.json')
    with open(json_path, 'w') as f:
        json.dump({'slide': slide_name, 'model': S3_MODEL_KEY,
                   'conf_threshold': args.conf,
                   'total_tiles': total_tiles,
                   'tiles_with_detections': tiles_with_detections,
                   'total_grains': total_grains,
                   'class_totals': {CLASS_MAP.get(k,(str(k),))[0]: v
                                    for k, v in cls_totals.items()},
                   'tiles': detections}, f, indent=2)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    summary_lines = [
        f"═══════════════════════════════════════════════",
        f"  Pollen Viability — Inference Summary",
        f"═══════════════════════════════════════════════",
        f"  Slide   : {slide_name}",
        f"  Image   : {W}×{H} px",
        f"  Tiles   : {total_tiles}  (conf≥{args.conf})",
        f"  Elapsed : {elapsed:.0f}s  ({total_tiles/elapsed:.1f} tiles/s)",
        f"",
        f"  Total grains detected : {total_grains}",
    ]
    for cls_id in sorted(cls_totals):
        name = CLASS_MAP.get(cls_id, (f'cls{cls_id}',))[0]
        pct  = cls_totals[cls_id] / total_grains * 100 if total_grains else 0
        summary_lines.append(f"    {name:<15}: {cls_totals[cls_id]:5d}  ({pct:.1f}%)")
    if total_grains > 1:
        viab  = cls_totals.get(0, 0)
        total = cls_totals.get(0, 0) + cls_totals.get(1, 0)
        if total > 0:
            summary_lines.append(f"")
            summary_lines.append(f"  Viability rate : {viab/total*100:.1f}%")
    summary_lines.append(f"═══════════════════════════════════════════════")

    summary = '\n'.join(summary_lines)
    print('\n' + summary)
    (out_dir / 'summary.txt').write_text(summary)
    print(f"\n📁 All output in: {out_dir.resolve()}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pollen model on a single CZI file')
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--pattern', type=str,
                     help='Substring to match against CZI filename on S3 (e.g. "27-3-B_AA025_n4x")')
    grp.add_argument('--key',     type=str,
                     help='Full S3 key of the CZI file')
    parser.add_argument('--conf',  type=float, default=0.10,
                        help='Confidence threshold (default: 0.10)')
    run(parser.parse_args())
