#!/usr/bin/env python3
"""
export_yolo_dataset.py  — Export Pinder-labeled grains to YOLO format on S3
════════════════════════════════════════════════════════════════════════════════
Scans S3 for labeled tiles and uploads a YOLO-ready dataset back to S3.
Nothing is written to local disk.

S3 output layout:
  {S3_DATASET_PREFIX}/
    images/train/{tile}.jpg
    images/val/{tile}.jpg
    labels/train/{tile}.txt
    labels/val/{tile}.txt
    data.yaml

Label sources:
  1. tiles_640/  — Swipe Mode labels (.txt alongside .jpg)
  2. active_learning/hard_positives/ — Grid/batch mode (jpg + segmentation txt)

Usage
─────
  source .venv/bin/activate
  python src/export_yolo_dataset.py
  python src/export_yolo_dataset.py --val-split 0.15 --prefix my_dataset/

Environment variables
─────────────────────
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  S3_ENDPOINT   (default: https://s3.cl4.du.cesnet.cz)
  S3_BUCKET     (default: bucket)
"""

import os, io, argparse, random, concurrent.futures
from collections import defaultdict

import boto3
from botocore.client import Config
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

S3_ENDPOINT      = os.environ.get("S3_ENDPOINT",   "https://s3.cl4.du.cesnet.cz")
AWS_ACCESS_KEY   = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY   = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_BUCKET        = os.environ.get("S3_BUCKET",     "bucket")
TILE_PREFIX      = "Ostatni/Pollen_viability/tiles_640/"
AL_PREFIX        = "Ostatni/Pollen_viability/active_learning/"
DATASET_PREFIX   = "Ostatni/Pollen_viability/yolo_dataset/"
VALID_EXT        = ('.jpg', '.jpeg', '.png')


def make_s3():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        config=Config(signature_version='s3v4', max_pool_connections=32),
    )


def find_labeled_tiles(s3) -> list[dict]:
    """Scan both label sources and return unique {tile_key, label_key} pairs."""
    paginator = s3.get_paginator('list_objects_v2')
    labeled   = []

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
    print(f"   {len(labeled)} Swipe-Mode labeled tiles")

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
    print(f"   {len(labeled) - al_before} batch-labeled tiles")

    # Deduplicate by filename stem
    seen, unique = {}, []
    for item in labeled:
        stem = os.path.basename(item['tile_key']).rsplit('.', 1)[0]
        if stem not in seen:
            seen[stem] = True
            unique.append(item)

    print(f"   → {len(unique)} unique labeled tiles total")
    return unique


def fetch_and_upload(args):
    """Download one (tile, label) pair from S3, re-upload to dataset prefix."""
    s3, item, split, dataset_prefix = args
    try:
        img_resp   = s3.get_object(Bucket=S3_BUCKET, Key=item['tile_key'])
        label_resp = s3.get_object(Bucket=S3_BUCKET, Key=item['label_key'])
        img_bytes  = img_resp['Body'].read()
        label_text = label_resp['Body'].read().decode('utf-8').strip()

        fname = os.path.basename(item['tile_key'])
        stem  = os.path.splitext(fname)[0]

        # Convert to JPEG in-memory (normalise format)
        buf = io.BytesIO()
        Image.open(io.BytesIO(img_bytes)).convert('RGB').save(buf, 'JPEG', quality=92)
        buf.seek(0)

        img_key   = f"{dataset_prefix}images/{split}/{stem}.jpg"
        label_key = f"{dataset_prefix}labels/{split}/{stem}.txt"

        s3.put_object(Bucket=S3_BUCKET, Key=img_key,
                      Body=buf.read(), ContentType='image/jpeg')
        s3.put_object(Bucket=S3_BUCKET, Key=label_key,
                      Body=label_text.encode('utf-8'),
                      ContentType='text/plain')

        return label_text, True
    except Exception as e:
        print(f"  ⚠️  Failed {item['tile_key']}: {e}")
        return "", False


def count_classes(label_text: str) -> dict:
    counts = defaultdict(int)
    for line in label_text.splitlines():
        parts = line.strip().split()
        if parts:
            counts[int(parts[0])] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Export Pinder-labeled tiles to YOLO dataset on S3")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--prefix",    default=DATASET_PREFIX,
                        help=f"S3 key prefix for dataset (default: {DATASET_PREFIX})")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise SystemExit("❌ AWS credentials not set.")

    random.seed(args.seed)
    prefix = args.prefix.rstrip('/') + '/'

    s3      = make_s3()
    labeled = find_labeled_tiles(s3)

    if not labeled:
        raise SystemExit(
            "❌ No labeled tiles found.\n"
            "   Run SAM first:  python src/run_sam_s3.py --target-samples 1-6-J,7-9-F,6-1-F --force\n"
            "   Then label in Pinder Swipe Mode, then re-run this script."
        )

    # Shuffle and split
    random.shuffle(labeled)
    n_val   = max(1, int(len(labeled) * args.val_split))
    splits  = {
        'train': labeled[n_val:],
        'val':   labeled[:n_val],
    }
    print(f"\n📊 Split: {len(splits['train'])} train / {n_val} val  "
          f"({args.val_split:.0%} val)")
    print(f"📤 Uploading to s3://{S3_BUCKET}/{prefix}\n")

    # Upload all tiles
    cls_counts  = defaultdict(int)
    total_saved = 0

    for split, items in splits.items():
        print(f"⬆️  {split}: {len(items)} tiles …")
        upload_args = [(s3, item, split, prefix) for item in items]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            for label_text, ok in ex.map(fetch_and_upload, upload_args):
                if ok:
                    for cls_id, n in count_classes(label_text).items():
                        cls_counts[cls_id] += n
                    total_saved += 1

    # Write data.yaml to S3
    yaml_content = (
        f"# YOLO dataset — Pollen viability\n"
        f"# Generated by export_yolo_dataset.py\n"
        f"# S3 location: s3://{S3_BUCKET}/{prefix}\n\n"
        f"path: .  # root relative to where you download the dataset\n"
        f"train: images/train\n"
        f"val:   images/val\n\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: viable\n"
        f"  1: non_viable\n"
    )
    s3.put_object(Bucket=S3_BUCKET, Key=prefix + 'data.yaml',
                  Body=yaml_content.encode('utf-8'), ContentType='text/plain')

    # ── Summary ───────────────────────────────────────────────────────────────
    cls_names    = {0: 'viable', 1: 'non_viable', 2: 'aborted'}
    total_grains = sum(cls_counts.values())

    print(f"\n{'═'*60}")
    print(f"  EXPORT SUMMARY")
    print(f"{'═'*60}")
    print(f"  Tiles uploaded   : {total_saved:>8,}")
    print(f"  Train / Val      : {len(splits['train']):>6,} / {n_val}")
    print(f"  Total grain labels: {total_grains:>6,}")
    for cls_id, n in sorted(cls_counts.items()):
        pct = n / max(total_grains, 1) * 100
        print(f"    cls {cls_id} ({cls_names.get(cls_id,'?'):<10}): {n:>6,}  ({pct:.1f}%)")

    ratio = cls_counts.get(0, 1) / max(cls_counts.get(1, 1), 1)
    if ratio > 3:
        print(f"\n  ⚠️  Imbalance: viable/non_viable = {ratio:.1f}×")
        print(f"      Label more NV grains in Pinder (samples: 1-6-J, 7-9-F, 6-1-F)")
    else:
        print(f"\n  ✅ Class balance OK (viable/non_viable = {ratio:.2f}×)")

    print(f"{'═'*60}")
    print(f"\n✅ Dataset on S3: s3://{S3_BUCKET}/{prefix}")
    print(f"   data.yaml key: {prefix}data.yaml")
    print(f"\n👉 To train (on a GPU machine that can pull from S3):")
    print(f"   # 1. Download dataset from S3")
    print(f"   #    aws s3 sync s3://{S3_BUCKET}/{prefix} ./dataset/ --endpoint-url {S3_ENDPOINT}")
    print(f"   # 2. Train")
    print(f"   #    yolo train data=dataset/data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
