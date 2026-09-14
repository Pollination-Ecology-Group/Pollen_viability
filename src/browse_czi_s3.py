#!/usr/bin/env python3
"""
browse_czi_s3.py — Fast CZI Overview Thumbnail Generator + NV Screener
═══════════════════════════════════════════════════════════════════════
Generates a compact visual thumbnail for every CZI sample folder stored in
S3 tiles_640/ WITHOUT downloading the large CZI files.

Strategy
────────
Each CZI has already been tiled into 640×640 JPEGs under:
  Ostatni/Pollen_viability/tiles_640/<folder_name>/<tile>.jpg

This script:
  1. Lists all CZI tile folders on S3.
  2. Downloads a small random sample of tiles per folder (~8 tiles = tiny).
  3. Runs the Alexander-stain NV color screen on each sampled tile.
  4. Composes a labeled mosaic (grid of tiles) as a preview thumbnail.
  5. Uploads thumbnails to S3: Ostatni/Pollen_viability/thumbs/<name>.jpg
  6. Prints a ranked table (highest NV density first) so you know which
     CZI images to prioritize for FastSAM + Swipe Mode curation.

You can then browse the thumbs/ prefix in Cyberduck, rclone, or any S3
browser to visually inspect which slides contain non-viable pollen.

Usage
─────
  source .venv/bin/activate
  python src/browse_czi_s3.py                 # all folders, 8 tiles each
  python src/browse_czi_s3.py --tiles 16      # sample more tiles per folder
  python src/browse_czi_s3.py --no-upload     # dry-run, save locally only
  python src/browse_czi_s3.py --force         # re-generate existing thumbs
  python src/browse_czi_s3.py --top 20        # process only top 20 NV folders

Environment variables
─────────────────────
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  S3_ENDPOINT   (default: https://s3.cl4.du.cesnet.cz)
  S3_BUCKET     (default: bucket)
"""

import os, io, json, time, random, argparse, concurrent.futures
from pathlib import Path

import boto3
from botocore.client import Config
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
S3_ENDPOINT  = os.environ.get("S3_ENDPOINT",  "https://s3.cl4.du.cesnet.cz")
AWS_ACCESS   = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET   = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_BUCKET    = os.environ.get("S3_BUCKET",    "bucket")
TILE_PREFIX  = "Ostatni/Pollen_viability/tiles_640/"
THUMB_PREFIX = "Ostatni/Pollen_viability/thumbs/"
LOCAL_OUT    = Path("data/czi_thumbs")

WORKERS      = 16
VALID_EXT    = ('.jpg', '.jpeg', '.png')

# ── Alexander-stain NV color screen params (same as preprocess_czi.py) ───────
MIN_PARTICLE_AREA = 250    # px²
MAX_PARTICLE_AREA = 25000  # px²
MIN_CIRCULARITY   = 0.22
MAX_MAGENTA_FRAC  = 0.20   # < this → NV candidate


# ── S3 helpers ────────────────────────────────────────────────────────────────

def make_s3():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS,
        aws_secret_access_key=AWS_SECRET,
        config=Config(signature_version='s3v4',
                      max_pool_connections=WORKERS + 4),
    )


def list_tile_folders(s3) -> list:
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=TILE_PREFIX, Delimiter='/')
    return [p['Prefix'] for p in resp.get('CommonPrefixes', [])]


def fetch_tile_keys(s3, folder: str) -> list:
    keys = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=folder):
        for obj in page.get('Contents', []):
            k = obj['Key']
            if k.lower().endswith(VALID_EXT):
                keys.append(k)
    return keys


def thumb_key_for(folder_name: str) -> str:
    return THUMB_PREFIX + folder_name + ".jpg"


def thumb_exists(s3, folder_name: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=thumb_key_for(folder_name))
        return True
    except Exception:
        return False


def download_tile_rgb(args):
    s3, key = args
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        pil  = Image.open(io.BytesIO(resp['Body'].read())).convert('RGB')
        return key, np.array(pil)           # H×W×3  uint8 RGB
    except Exception:
        return key, None


# ── Alexander-stain NV screen ─────────────────────────────────────────────────

def nv_score(img_rgb: np.ndarray):
    """
    Returns (nv_count, nv_rate) for one tile.
    Higher = more non-viable pollen candidates.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    is_bg      = (s < 35) & (v > 195)
    is_magenta = ((h > 135) | (h < 12)) & (s > 55) & (v < 225)

    fg = (~is_bg).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    nv_count    = 0
    total_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_PARTICLE_AREA < area < MAX_PARTICLE_AREA):
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        if 4 * np.pi * area / (peri * peri) < MIN_CIRCULARITY:
            continue
        total_count += 1

        c_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        cv2.drawContours(c_mask, [cnt], -1, 255, -1)
        n_px = np.sum(c_mask > 0)
        if n_px == 0:
            continue
        m_frac = np.sum((c_mask > 0) & is_magenta) / n_px

        if m_frac < MAX_MAGENTA_FRAC:
            nv_count += 1

    rate = nv_count / total_count if total_count > 0 else 0.0
    return nv_count, rate


# ── Mosaic composer ───────────────────────────────────────────────────────────

def _try_load_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_mosaic(tiles_rgb, nv_scores, folder_name, tile_display_size=320, cols=4):
    """
    Compose a labeled grid mosaic from sampled tiles.
    Each tile gets an NV count badge in the corner.
    """
    n      = len(tiles_rgb)
    rows   = (n + cols - 1) // cols
    header = 56
    total_w = cols * tile_display_size
    total_h = header + rows * tile_display_size

    canvas = Image.new('RGB', (total_w, total_h), color=(22, 22, 30))
    draw   = ImageDraw.Draw(canvas)

    font_hdr = _try_load_font(20)
    font_sm  = _try_load_font(14)

    # Header bar
    draw.rectangle([0, 0, total_w, header - 1], fill=(38, 38, 52))

    avg_nv_cnt  = sum(c for c, _ in nv_scores) / len(nv_scores) if nv_scores else 0.0
    avg_nv_rate = sum(r for _, r in nv_scores) / len(nv_scores) if nv_scores else 0.0
    draw.text((8, 8),  folder_name[:60],                         font=font_hdr, fill=(210, 215, 255))
    draw.text((8, 32), f"NV avg: {avg_nv_cnt:.1f}/tile  ({avg_nv_rate*100:.0f}% of grains)",
              font=font_sm, fill=(160, 200, 160))

    # Tiles
    for i, (img_rgb, (cnt, rate)) in enumerate(zip(tiles_rgb, nv_scores)):
        col = i % cols
        row = i // cols
        x0  = col * tile_display_size
        y0  = header + row * tile_display_size

        thumb = Image.fromarray(img_rgb).resize(
            (tile_display_size, tile_display_size), Image.LANCZOS)
        canvas.paste(thumb, (x0, y0))

        # NV badge
        if cnt > 0:
            if rate < 0.10:
                badge_rgb = (60, 179, 60)
            elif rate < 0.30:
                badge_rgb = (230, 160, 50)
            else:
                badge_rgb = (220, 60, 60)
            badge_text = f"NV:{cnt}"
            bw, bh = 58, 22
            bx = x0 + 4
            by = y0 + tile_display_size - bh - 4
            draw.rectangle([bx, by, bx + bw, by + bh], fill=badge_rgb)
            draw.text((bx + 4, by + 3), badge_text, font=font_sm, fill=(255, 255, 255))

        # Thin tile border
        draw.rectangle([x0, y0, x0 + tile_display_size - 1, y0 + tile_display_size - 1],
                        outline=(55, 55, 72), width=1)

    return canvas


# ── Per-folder processing ─────────────────────────────────────────────────────

def process_folder(s3, folder_prefix, n_tiles, force, no_upload):
    """Sample tiles, screen for NV, build mosaic, upload. Returns stats dict."""
    folder_name = folder_prefix.rstrip('/').split('/')[-1]

    if not force and not no_upload and thumb_exists(s3, folder_name):
        return {'folder': folder_name, 'skipped': True, 'reason': 'exists'}

    all_keys = fetch_tile_keys(s3, folder_prefix)
    if not all_keys:
        return {'folder': folder_name, 'skipped': True, 'reason': 'no tiles'}

    sample_keys = random.sample(all_keys, min(n_tiles, len(all_keys)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(WORKERS, len(sample_keys))) as ex:
        dl_results = list(ex.map(download_tile_rgb, [(s3, k) for k in sample_keys]))

    valid = [(k, img) for k, img in dl_results if img is not None]
    if not valid:
        return {'folder': folder_name, 'skipped': True, 'reason': 'download failed'}

    imgs   = [img for _, img in valid]
    scores = [nv_score(img) for img in imgs]

    avg_nv_cnt  = sum(c for c, _ in scores) / len(scores)
    avg_nv_rate = sum(r for _, r in scores) / len(scores)

    cols   = min(4, len(imgs))
    mosaic = make_mosaic(imgs, scores, folder_name, tile_display_size=320, cols=cols)

    # Save locally
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    local_path = LOCAL_OUT / f"{folder_name}.jpg"
    mosaic.save(str(local_path), 'JPEG', quality=88)

    # Upload to S3
    if not no_upload:
        buf = io.BytesIO()
        mosaic.save(buf, 'JPEG', quality=88)
        buf.seek(0)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=thumb_key_for(folder_name),
            Body=buf,
            ContentType='image/jpeg',
        )

    return {
        'folder':      folder_name,
        'skipped':     False,
        'total_tiles': len(all_keys),
        'sampled':     len(imgs),
        'avg_nv_cnt':  avg_nv_cnt,
        'avg_nv_rate': avg_nv_rate,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate NV-ranked CZI overview thumbnails from S3 tiles")
    parser.add_argument('--tiles',     type=int, default=8,
                        help='Tiles to sample per CZI folder (default: 8)')
    parser.add_argument('--top',       type=int, default=0,
                        help='Only process top N folders by NV density (0=all)')
    parser.add_argument('--force',     action='store_true',
                        help='Regenerate even if thumbnail already exists on S3')
    parser.add_argument('--no-upload', action='store_true',
                        help='Save thumbnails locally only, do not upload to S3')
    parser.add_argument('--workers',   type=int, default=6,
                        help='Parallel folder workers (default: 6)')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    if not AWS_ACCESS or not AWS_SECRET:
        raise SystemExit("❌  AWS credentials not set. Source .env first.")

    random.seed(args.seed)
    s3 = make_s3()

    print("\n🌸  CZI Overview Browser — Thumbnail Generator")
    print(f"   Sampling {args.tiles} tiles per folder")
    if args.no_upload:
        print(f"   Upload  : ❌ disabled (--no-upload)")
    else:
        print(f"   Upload  : ✅ s3://{S3_BUCKET}/{THUMB_PREFIX}")
    print(f"   Local   : {LOCAL_OUT.resolve()}\n")

    print("📋  Listing tile folders on S3 …")
    folders = list_tile_folders(s3)
    print(f"   Found {len(folders)} CZI sample folders\n")

    if not folders:
        raise SystemExit("No tile folders found. Has preprocess_czi.py been run yet?")

    # ── Optional quick pre-scan to select top N folders ────────────────────
    if args.top > 0 and len(folders) > args.top:
        print(f"⚡  Quick pre-scan (2 tiles × {len(folders)} folders) to find top {args.top} NV-dense …")
        t0 = time.time()

        def quick_score(fp):
            try:
                keys = fetch_tile_keys(s3, fp)
                sample = random.sample(keys, min(2, len(keys)))
                cnts = []
                for k in sample:
                    _, img = download_tile_rgb((s3, k))
                    if img is not None:
                        cnt, _ = nv_score(img)
                        cnts.append(cnt)
                return (fp, sum(cnts) / len(cnts) if cnts else 0.0)
            except Exception:
                return (fp, 0.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers * 4) as ex:
            quick_results = list(ex.map(quick_score, folders))

        quick_results.sort(key=lambda x: x[1], reverse=True)
        folders = [f for f, _ in quick_results[: args.top]]
        print(f"   Pre-scan done in {time.time()-t0:.1f}s → keeping top {len(folders)} folders\n")

    # ── Full processing ──────────────────────────────────────────────────────
    print(f"🔍  Building thumbnails for {len(folders)} folders (workers={args.workers}) …\n")
    t_start  = time.time()
    results  = []

    def _process(fp):
        return process_folder(s3, fp, args.tiles, args.force, args.no_upload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process, fp): fp for fp in folders}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            done += 1
            r = future.result()
            results.append(r)
            name = r['folder'][:55]
            if not r.get('skipped'):
                nv_s = f"{r['avg_nv_cnt']:.1f} NV/tile  ({r['avg_nv_rate']*100:.0f}%)"
                print(f"  [{done:3d}/{len(folders)}] {name:<55}  {nv_s}", flush=True)
            else:
                print(f"  [{done:3d}/{len(folders)}] {name:<55}  ⏭ {r.get('reason','skipped')}", flush=True)

    elapsed   = time.time() - t_start
    processed = [r for r in results if not r.get('skipped')]
    processed.sort(key=lambda r: r.get('avg_nv_cnt', 0), reverse=True)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"✅  Done in {elapsed:.1f}s  |  {len(processed)} thumbnails generated\n")

    if processed:
        print(f"  {'Rank':>4}  {'CZI Folder':<48}  {'NV/tile':>8}  {'NV%':>5}  {'Tiles':>6}")
        print("  " + "─" * 75)
        for i, r in enumerate(processed[:50], 1):
            bar = "█" * min(int(r['avg_nv_cnt'] * 3), 18)
            print(f"  {i:4d}  {r['folder'][:48]:<48}  {r['avg_nv_cnt']:>8.2f}  "
                  f"{r['avg_nv_rate']*100:>4.0f}%  {r['total_tiles']:>6}  {bar}")

    print()
    if not args.no_upload:
        print(f"👁️  Browse thumbnails on S3:")
        print(f"   Endpoint : {S3_ENDPOINT}")
        print(f"   Path     : s3://{S3_BUCKET}/{THUMB_PREFIX}")
        print(f"   rclone   : rclone copy remote:{S3_BUCKET}/{THUMB_PREFIX} ./data/czi_thumbs/")
    print(f"📂  Local copies : {LOCAL_OUT.resolve()}/\n")

    if processed:
        top5 = [r['folder'] for r in processed[:5]]
        print(f"🎯  Top 5 most NV-rich (pass to run_sam_s3.py):")
        print(f"   python src/run_sam_s3.py --target-samples {','.join(top5)}\n")


if __name__ == '__main__':
    main()
