#!/usr/bin/env python3
"""
export_yolo_dataset.py  — Export Pinder-labeled grains to YOLO training format
════════════════════════════════════════════════════════════════════════════════
Scans S3 tiles_640/ for tiles that have a companion .txt label file
(written by Pinder Swipe Mode) and packages them into a YOLO-ready dataset.

Label format (written by Pinder Swipe Mode):
  Segmentation: {cls_id} x1 y1 x2 y2 ... (polygon, normalised)
  Detection:    {cls_id} xc yc w h       (bbox, normalised)

Output structure:
  {out_dir}/
    images/train/   80% of tiles
    images/val/     20% of tiles
    labels/train/   matching .txt files
    labels/val/
    data.yaml       nc=2, names: [viable, non_viable]

Usage
─────
  source .venv/bin/activate
  python src/export_yolo_dataset.py --out ./dataset/
  python src/export_yolo_dataset.py --out ./dataset/ --val-split 0.15

Environment variables
─────────────────────
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  S3_ENDPOINT, S3_BUCKET
"""

import os, io, argparse, random, shutil, concurrent.futures
from pathlib import Path
from collections import defaultdict

import boto3
from botocore.client import Config
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

S3_ENDPOINT    = os.environ.get("S3_ENDPOINT",   "https://s3.cl4.du.cesnet.cz")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_BUCKET      = os.environ.get("S3_BUCKET",     "bucket")
TILE_PREFIX    = "Ostatni/Pollen_viability/tiles_640/"
VALID_EXT      = ('.jpg', '.jpeg', '.png')


def make_s3():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        config=Config(signature_version='s3v4', max_pool_connections=24),
    )


def find_labeled_tiles(s3) -> list[dict]:
    """Return list of {tile_key, label_key} for tiles with companion .txt labels.

    Scans two sources:
      1. tiles_640/   — Swipe Mode labels (.txt alongside .jpg)
      2. active_learning/hard_positives/ — Grid/batch mode (jpg + segmentation txt)
    """
    AL_PREFIX = "Ostatni/Pollen_viability/active_learning/"
    paginator  = s3.get_paginator('list_objects_v2')
    labeled    = []

    # ── Source 1: tiles_640/ (Swipe Mode) ─────────────────────────────────────
    print("📋 Scanning tiles_640/ for Swipe-Mode labels …")
    all_keys = set()
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=TILE_PREFIX):
        for obj in page.get('Contents', []):
            all_keys.add(obj['Key'])

    for key in all_keys:
        if not key.lower().endswith(VALID_EXT):
            continue
        label_key = key.rsplit('.', 1)[0] + '.txt'
        if label_key in all_keys:
            labeled.append({'tile_key': key, 'label_key': label_key,
                            'source': 'swipe'})
    print(f"   Found {len(labeled)} Swipe-Mode labeled tiles")

    # ── Source 2: active_learning/hard_positives/ (Grid / batch mode) ─────────
    print("📋 Scanning active_learning/hard_positives/ for batch labels …")
    al_keys = set()
    for page in paginator.paginate(Bucket=S3_BUCKET,
                                   Prefix=AL_PREFIX + "hard_positives/"):
        for obj in page.get('Contents', []):
            al_keys.add(obj['Key'])

    al_before = len(labeled)
    for key in al_keys:
        if not key.lower().endswith(VALID_EXT):
            continue
        label_key = key.rsplit('.', 1)[0] + '.txt'
        if label_key in al_keys:
            labeled.append({'tile_key': key, 'label_key': label_key,
                            'source': 'batch'})
    print(f"   Found {len(labeled) - al_before} batch-labeled tiles")

    # Deduplicate by filename stem (same tile may appear in both)
    seen = {}
    unique = []
    for item in labeled:
        stem = os.path.basename(item['tile_key']).rsplit('.', 1)[0]
        if stem not in seen:
            seen[stem] = True
            unique.append(item)

    print(f"   Total unique labeled tiles: {len(unique)}")
    return unique


def download_pair(args):
    s3, item = args
    try:
        img_resp   = s3.get_object(Bucket=S3_BUCKET, Key=item['tile_key'])
        label_resp = s3.get_object(Bucket=S3_BUCKET, Key=item['label_key'])
        img_bytes   = img_resp['Body'].read()
        label_text  = label_resp['Body'].read().decode('utf-8').strip()
        return item, img_bytes, label_text
    except Exception as e:
        print(f"  ⚠️  Failed {item['tile_key']}: {e}")
        return item, None, None


def count_classes(label_text: str) -> dict:
    counts = defaultdict(int)
    for line in label_text.splitlines():
        parts = line.strip().split()
        if parts:
            counts[int(parts[0])] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Export Pinder-labeled tiles to YOLO training dataset")
    parser.add_argument("--out", default="./dataset",
                        help="Output directory (default: ./dataset)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction for validation set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise SystemExit("❌ AWS credentials not set.")

    random.seed(args.seed)
    out = Path(args.out)

    s3 = make_s3()
    labeled = find_labeled_tiles(s3)

    if not labeled:
        raise SystemExit("❌ No labeled tiles found. Use Pinder Swipe Mode to label some tiles first.")

    # Shuffle and split
    random.shuffle(labeled)
    n_val   = max(1, int(len(labeled) * args.val_split))
    n_train = len(labeled) - n_val
    splits  = {'train': labeled[:n_train], 'val': labeled[n_train:]}

    print(f"\n📊 Split: {n_train} train  /  {n_val} val  ({args.val_split:.0%} val)")

    # Create directory structure
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Download and save
    cls_counts = defaultdict(int)
    total_saved = 0

    for split, items in splits.items():
        print(f"\n⬇️  Downloading {len(items)} {split} tiles …")
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            for item, img_bytes, label_text in ex.map(
                    download_pair, [(s3, it) for it in items]):
                if img_bytes is None or label_text is None:
                    continue

                fname = Path(item['tile_key']).name
                stem  = Path(fname).stem

                # Save image (convert to jpg if needed)
                img_path = out / 'images' / split / (stem + '.jpg')
                pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                pil.save(img_path, quality=92)

                # Save label
                label_path = out / 'labels' / split / (stem + '.txt')
                label_path.write_text(label_text)

                # Count classes
                for cls_id, n in count_classes(label_text).items():
                    cls_counts[cls_id] += n

                total_saved += 1

    # Write data.yaml
    yaml_content = f"""# YOLO dataset — Pollen viability
path: {out.resolve()}
train: images/train
val:   images/val

nc: 2
names:
  0: viable
  1: non_viable
"""
    (out / 'data.yaml').write_text(yaml_content)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  EXPORT SUMMARY")
    print(f"{'═'*55}")
    print(f"  Tiles saved      : {total_saved:>8,}")
    print(f"  Train / Val      : {n_train:>6,} / {n_val:,}")
    total_grains = sum(cls_counts.values())
    print(f"  Total grain labels: {total_grains:>7,}")
    cls_names = {0: 'viable', 1: 'non_viable', 2: 'aborted'}
    for cls_id, n in sorted(cls_counts.items()):
        pct = n / max(total_grains, 1) * 100
        print(f"    cls {cls_id} ({cls_names.get(cls_id,'?'):<10}): {n:>6,}  ({pct:.1f}%)")

    ratio = cls_counts.get(0, 0) / max(cls_counts.get(1, 1), 1)
    if ratio > 3:
        print(f"\n  ⚠️  Class imbalance: viable/non_viable ratio = {ratio:.1f}x")
        print(f"      Consider labeling more non-viable tiles from 1-6-J / 7-9-F.")
    elif ratio < 0.33:
        print(f"\n  ⚠️  Class imbalance: non_viable/viable ratio = {1/ratio:.1f}x")
        print(f"      Consider adding more viable-dominant tiles.")
    else:
        print(f"\n  ✅ Class balance looks good (ratio: {ratio:.2f})")

    print(f"{'═'*55}")
    print(f"\n✅ Dataset saved → {out.resolve()}")
    print(f"   data.yaml: {(out / 'data.yaml').resolve()}")
    print(f"\n👉 Train with:")
    print(f"   yolo train data={out.resolve()}/data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
