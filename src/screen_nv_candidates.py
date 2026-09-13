#!/usr/bin/env python3
"""
screen_nv_candidates.py — Color-based non-viable pollen tile screener
══════════════════════════════════════════════════════════════════════
No GPU needed. Uses the Alexander-stain color logic already in preprocess_czi.py:
  - Viable pollen = deep magenta/purple (high saturation, hue 135–12)
  - Non-viable    = pale, green, grey, or transparent (low magenta fraction)

Phase 1: Randomly sample N tiles per CZI folder, score folders by NV density.
Phase 2: For top-K folders, screen ALL tiles and write _det.json with
         bounding boxes of NV candidate particles (cls=0 placeholder).

After running, open Pinder → Source Image Filter → select the output folders
→ Swipe Mode → classify each crop as viable / non-viable / skip.

Usage
─────
  source .venv/bin/activate
  python src/screen_nv_candidates.py          # default: top 15 folders
  python src/screen_nv_candidates.py --top 20 --sample-per-folder 8

Environment variables
─────────────────────
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  S3_ENDPOINT, S3_BUCKET
"""

import os, io, json, time, random, argparse, concurrent.futures
from collections import defaultdict

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

S3_ENDPOINT  = os.environ.get("S3_ENDPOINT",  "https://s3.cl4.du.cesnet.cz")
AWS_ACCESS   = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET   = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_BUCKET    = os.environ.get("S3_BUCKET",    "bucket")
TILE_PREFIX  = "Ostatni/Pollen_viability/tiles_640/"
WORKERS      = 24

# ── Alexander-stain color params (from preprocess_czi.py) ─────────────────────
MIN_PARTICLE_AREA  = 250    # px²  — ignore tiny specks
MAX_PARTICLE_AREA  = 25000  # px²  — ignore huge air bubbles
MIN_CIRCULARITY    = 0.22   # filters non-circular debris
MAX_MAGENTA_FRAC   = 0.20   # particle with <20% magenta pixels = NV candidate


def make_s3():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS,
        aws_secret_access_key=AWS_SECRET,
        config=Config(signature_version='s3v4',
                      max_pool_connections=WORKERS + 4),
    )


# ── Core color screen ──────────────────────────────────────────────────────────

def screen_tile(img_bgr: np.ndarray) -> list[dict]:
    """
    Return list of NV candidate detections.
    Each dict: {x1, y1, x2, y2, score}
    score = 1 - magenta_fraction  (higher = more non-viable-looking)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Background mask (white/light-grey slide)
    is_bg = (s < 35) & (v > 195)

    # Deep magenta/purple (viable cytoplasm)
    is_magenta = ((h > 135) | (h < 12)) & (s > 55) & (v < 225)

    # Foreground particle mask
    fg = (~is_bg).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_PARTICLE_AREA < area < MAX_PARTICLE_AREA):
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        if 4 * np.pi * area / (peri * peri) < MIN_CIRCULARITY:
            continue

        c_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(c_mask, [cnt], -1, 255, -1)
        n_px  = np.sum(c_mask > 0)
        if n_px == 0:
            continue
        m_frac = np.sum((c_mask > 0) & is_magenta) / n_px

        if m_frac < MAX_MAGENTA_FRAC:
            x, y, w, h_box = cv2.boundingRect(cnt)
            detections.append({
                'x1': x, 'y1': y,
                'x2': x + w, 'y2': y + h_box,
                'score': round(1.0 - m_frac, 4),
            })
    return detections


def det_json_from_detections(dets: list[dict], img_w: int, img_h: int) -> dict:
    """Convert color-screen detections → _det.json format (cls=0 placeholder)."""
    boxes = []
    masks = []
    for d in dets:
        x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
        boxes.append([x1, y1, x2, y2, round(d['score'], 4), 0])
        # Rectangular pseudo-mask (normalised)
        nw, nh = max(img_w, 1), max(img_h, 1)
        masks.append([
            [x1/nw, y1/nh], [x2/nw, y1/nh],
            [x2/nw, y2/nh], [x1/nw, y2/nh],
        ])
    return {'boxes': boxes, 'masks_xyn': masks}


# ── S3 helpers ─────────────────────────────────────────────────────────────────

def list_tile_folders(s3) -> list[str]:
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=TILE_PREFIX, Delimiter='/')
    return [p['Prefix'] for p in resp.get('CommonPrefixes', [])]


def fetch_folder_keys(s3, folder: str) -> list[str]:
    keys = []
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=folder)
    for obj in resp.get('Contents', []):
        k = obj['Key']
        if k.lower().endswith(('.jpg', '.jpeg', '.png')):
            keys.append(k)
    return keys


def download_tile_bgr(args):
    s3, key = args
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        pil  = Image.open(io.BytesIO(resp['Body'].read())).convert('RGB')
        return key, cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return key, None


# ── Phase 1: Quick folder ranking ─────────────────────────────────────────────

def rank_folders(s3, folders: list[str], sample_n: int) -> list[tuple]:
    """
    Sample sample_n tiles per folder, screen them, return
    list of (folder, avg_nv_per_tile) sorted descending.
    """
    print(f"\n📊 Phase 1 — Sampling {sample_n} tiles × {len(folders)} folders …")
    t0 = time.time()

    # Build sample list
    sample_pairs = []
    folder_tile_counts = {}
    for folder in folders:
        keys = fetch_folder_keys(s3, folder)
        tile_keys = [k for k in keys if not k.endswith('_det.json')]
        folder_tile_counts[folder] = len(tile_keys)
        sampled = random.sample(tile_keys, min(sample_n, len(tile_keys)))
        for k in sampled:
            sample_pairs.append((folder, k))

    print(f"   Downloading {len(sample_pairs)} sample tiles …")
    folder_nv_scores = defaultdict(list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        dl_args = [(s3, pair[1]) for pair in sample_pairs]
        for (folder, _), (key, img) in zip(
                sample_pairs, ex.map(download_tile_bgr, dl_args)):
            if img is None:
                continue
            dets = screen_tile(img)
            folder_nv_scores[folder].append(len(dets))

    # Rank: average NV candidates per sampled tile
    ranked = []
    for folder in folders:
        scores = folder_nv_scores.get(folder, [0])
        avg = sum(scores) / len(scores)
        total_tiles = folder_tile_counts.get(folder, 0)
        ranked.append((folder, avg, total_tiles))

    ranked.sort(key=lambda x: x[1], reverse=True)
    print(f"   Done in {time.time()-t0:.1f}s")
    return ranked


# ── Phase 2: Full detection for top folders ────────────────────────────────────

def process_folder(args):
    s3, folder, force = args
    keys = fetch_folder_keys(s3, folder)
    tile_keys = [k for k in keys if not k.endswith('_det.json')]

    # Check existing det.json keys
    existing_det = {k for k in keys if k.endswith('_det.json')}

    total_nv = 0
    tiles_with_nv = 0
    uploaded = 0

    for key in tile_keys:
        det_key = key.rsplit('.', 1)[0] + '_det.json'
        if not force and det_key in existing_det:
            continue
        try:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
            pil  = Image.open(io.BytesIO(resp['Body'].read())).convert('RGB')
            img  = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            dets = screen_tile(img)
            det  = det_json_from_detections(dets, pil.width, pil.height)
            body = json.dumps(det, separators=(',', ':')).encode('utf-8')
            s3.put_object(Bucket=S3_BUCKET, Key=det_key,
                          Body=body, ContentType='application/json')
            uploaded += 1
            if dets:
                total_nv += len(dets)
                tiles_with_nv += 1
        except Exception as e:
            print(f"  ⚠️  {key}: {e}")

    name = folder.rstrip('/').split('/')[-1]
    print(f"   {name[:60]:60s}  tiles={len(tile_keys):4d}  "
          f"NV_tiles={tiles_with_nv:3d}  NV_grains={total_nv:4d}", flush=True)
    return folder, total_nv


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Color-screen tiles for NV candidates, write _det.json to S3")
    parser.add_argument('--top',               type=int, default=15,
                        help='Number of top folders to fully process (default: 15)')
    parser.add_argument('--sample-per-folder', type=int, default=6,
                        help='Tiles to sample per folder in Phase 1 (default: 6)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing _det.json in targeted folders')
    parser.add_argument('--seed',              type=int, default=42)
    args = parser.parse_args()

    if not AWS_ACCESS or not AWS_SECRET:
        raise SystemExit('❌ AWS credentials not set.')

    random.seed(args.seed)
    s3 = make_s3()

    print('🌸 NV Tile Screener (color-based, no GPU needed)')

    # List all folders
    print(f'\n📋 Listing CZI folders …')
    folders = list_tile_folders(s3)
    print(f'   Found {len(folders)} folders')

    # ── Phase 1: Rank folders ────────────────────────────────────────────────
    ranked = rank_folders(s3, folders, args.sample_per_folder)

    print(f'\n🏆 Top {args.top} folders by NV candidate density:')
    print(f'   {"Folder name":<60}  {"Avg NV/tile":>11}  {"Total tiles":>11}')
    print('   ' + '─' * 86)
    top_folders = []
    for i, (folder, avg, total) in enumerate(ranked[:args.top]):
        name = folder.rstrip('/').split('/')[-1]
        print(f'   {name[:60]:<60}  {avg:>11.2f}  {total:>11}')
        top_folders.append(folder)

    if not top_folders:
        raise SystemExit('No folders found.')

    # ── Phase 2: Full detection for top folders ──────────────────────────────
    print(f'\n🔍 Phase 2 — Full color-screen + _det.json upload for top {args.top} folders …')
    print(f'   (force={args.force})\n')
    t1 = time.time()

    # Process sequentially so progress is readable
    # (parallel folder processing would interleave output)
    total_nv_written = 0
    for folder in top_folders:
        _, nv = process_folder((s3, folder, args.force))
        total_nv_written += nv

    elapsed = time.time() - t1
    print(f'\n✅ Done in {elapsed:.1f}s  |  Total NV candidates written: {total_nv_written}')

    # Output folder names for Pinder Source Image Filter
    short_names = [f.rstrip('/').split('/')[-1] for f in top_folders]
    print(f'\n👉 Open Pinder → Source Image Filter → select these folders:')
    for name in short_names:
        print(f'   • {name}')
    print(f'\n   Then: Swipe Mode → classify pale particle crops as viable / non-viable')


if __name__ == '__main__':
    main()
