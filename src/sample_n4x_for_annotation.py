#!/usr/bin/env python3
"""
sample_n4x_for_annotation.py  — Sample representative n4x tiles for annotation
═══════════════════════════════════════════════════════════════════════════════
The current training dataset contains only S4x/4x slide images (white/pink
background). The inference tiles include n4x slides (AA012 mounting medium,
teal background) that the model has never been trained on.

This script:
  1. Finds all n4x slide folders in S3 tiles_640/
  2. Samples N tiles per slide (prioritising tiles likely to contain pollen)
  3. Downloads them into  annotation_batch/n4x_tiles/
  4. Produces a ZIP ready for upload to Roboflow (or any annotation tool)

Usage:
  python src/sample_n4x_for_annotation.py
  python src/sample_n4x_for_annotation.py --per-slide 5 --max-slides 20
  python src/sample_n4x_for_annotation.py --per-slide 10 --output-dir my_batch

After running:
  1. Open Roboflow → your project → Upload
  2. Drag-and-drop annotation_batch/n4x_tiles_for_annotation.zip
  3. Annotate the grains (class 0=viable, 1=non_viable, 2=intermediate)
  4. Export as YOLO Segmentation format and drop the zip into S3:
       Ostatni/Pollen_viability/staging_area/<name>.zip
  5. Re-run training — train_model.py will auto-merge via merge_staged_data()
"""

import os, io, argparse, zipfile, random, cv2, numpy as np
from pathlib import Path

import boto3
from botocore.client import Config
from PIL import Image

# ── config ────────────────────────────────────────────────────────────────────
S3_ENDPOINT  = os.environ.get("S3_ENDPOINT",  "https://s3.cl4.du.cesnet.cz")
AWS_ACCESS   = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET   = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_BUCKET    = os.environ.get("S3_BUCKET", "bucket")
TILE_PREFIX  = "Ostatni/Pollen_viability/tiles_640/"
VALID_EXT    = ('.jpg', '.jpeg', '.png')


def make_s3():
    return boto3.client("s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS,
        aws_secret_access_key=AWS_SECRET,
        config=Config(signature_version="s3v4", max_pool_connections=16))


def pollen_score(bgr: np.ndarray) -> float:
    """
    Heuristic: score a tile by how much dark-purple pollen content it has.
    High score = likely to contain pollen grains worth annotating.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Dark purple/magenta: hue ~135-180 or 0-10, high saturation, not too bright
    purple = ((h > 130) | (h < 15)) & (s > 60) & (v < 210)
    ratio  = purple.sum() / purple.size
    return float(ratio)


def is_n4x_folder(folder_name: str) -> bool:
    return "n4x" in folder_name.lower()


def list_n4x_folders(s3) -> list[str]:
    folders = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=TILE_PREFIX, Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            folder = p["Prefix"]
            name   = folder.split("/")[-2]
            if is_n4x_folder(name):
                folders.append(folder)
    return folders


def sample_tiles_from_folder(s3, folder_prefix: str,
                              n: int, score_sample: int = 50) -> list[tuple[str, float]]:
    """
    List up to `score_sample` tiles in the folder, score them by pollen content,
    return top `n` keys sorted by score descending.
    """
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=folder_prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith(VALID_EXT) and not k.endswith("_det.json"):
                keys.append(k)
        if len(keys) >= score_sample * 3:
            break

    if not keys:
        return []

    # Sub-sample to avoid downloading hundreds of tiles for scoring
    candidates = random.sample(keys, min(len(keys), score_sample))

    scored = []
    for k in candidates:
        try:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=k)
            data = resp["Body"].read()
            pil  = Image.open(io.BytesIO(data)).convert("RGB")
            bgr  = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            score = pollen_score(bgr)
            scored.append((k, score, bgr))
        except Exception as e:
            print(f"    ⚠️  Could not score {k.split('/')[-1]}: {e}")

    # Sort by pollen score, take top n
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(k, score, bgr) for k, score, bgr in scored[:n]]


def main():
    parser = argparse.ArgumentParser(description="Sample n4x tiles for Roboflow annotation")
    parser.add_argument("--per-slide",   type=int, default=4,
                        help="Tiles to sample per n4x slide folder (default: 4)")
    parser.add_argument("--max-slides",  type=int, default=30,
                        help="Max number of n4x slide folders to sample from (default: 30)")
    parser.add_argument("--output-dir",  type=str, default="annotation_batch/n4x_tiles",
                        help="Local output directory for sampled tiles")
    parser.add_argument("--min-score",   type=float, default=0.005,
                        help="Minimum pollen-content score to keep a tile (default: 0.005)")
    args = parser.parse_args()

    global AWS_ACCESS, AWS_SECRET
    if not AWS_ACCESS or not AWS_SECRET:
        # Try loading from .env
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()
            AWS_ACCESS = os.environ.get("AWS_ACCESS_KEY_ID")
            AWS_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not AWS_ACCESS or not AWS_SECRET:
        raise SystemExit("❌ AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set. Source .env first.")

    s3 = make_s3()

    print("=" * 65)
    print("🌸  n4x Tile Sampler — Annotation Batch Preparation")
    print("=" * 65)
    print(f"   Tiles per slide : {args.per_slide}")
    print(f"   Max slides      : {args.max_slides}")
    print(f"   Output dir      : {args.output_dir}")
    print(f"   Min score       : {args.min_score}")
    print()

    # 1. Find n4x slide folders
    print("🔍 Listing n4x slide folders in S3...")
    all_n4x_folders = list_n4x_folders(s3)
    print(f"   Found {len(all_n4x_folders)} n4x slide folders.")

    # Sample a diverse subset of slides
    if len(all_n4x_folders) > args.max_slides:
        selected_folders = random.sample(all_n4x_folders, args.max_slides)
        print(f"   Sub-sampling {args.max_slides} slides for diversity.")
    else:
        selected_folders = all_n4x_folders

    # 2. Sample tiles from each folder
    os.makedirs(args.output_dir, exist_ok=True)
    saved_tiles = []
    total_skipped = 0

    for i, folder in enumerate(selected_folders):
        slide_name = folder.split("/")[-2]
        print(f"  [{i+1:2d}/{len(selected_folders)}] {slide_name}", flush=True)

        results = sample_tiles_from_folder(s3, folder, n=args.per_slide)

        for key, score, bgr in results:
            if score < args.min_score:
                total_skipped += 1
                print(f"    ⏭️  Skipped (score {score:.4f} < {args.min_score})")
                continue

            # Save with flat filename encoding the slide name
            tile_fname = key.split("/")[-1]
            safe_slide = slide_name.replace(" ", "_")[:60]
            out_name   = f"{safe_slide}__{tile_fname}"
            out_path   = os.path.join(args.output_dir, out_name)

            cv2.imwrite(out_path, bgr)
            saved_tiles.append(out_path)
            print(f"    ✅ {out_name}  (score={score:.4f})")

    # 3. Bundle into ZIP for Roboflow
    zip_path = str(Path(args.output_dir).parent / "n4x_tiles_for_annotation.zip")
    print(f"\n📦 Creating ZIP → {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in saved_tiles:
            zf.write(fp, arcname=os.path.basename(fp))

    print()
    print("─" * 65)
    print(f"✅  Done!")
    print(f"   Saved  : {len(saved_tiles)} tiles  (skipped {total_skipped} low-score)")
    print(f"   Folder : {args.output_dir}/")
    print(f"   ZIP    : {zip_path}")
    print()
    print("📋  Next steps:")
    print("   1. Open your Roboflow project → Upload the ZIP")
    print("      (these are unannotated images — annotate grains there)")
    print("   2. Annotate: class 0=viable  class 1=non_viable  class 2=intermediate")
    print("   3. Export as 'YOLO v8 Segmentation' format → download ZIP")
    print("   4. Upload the exported ZIP to S3:")
    print("      Ostatni/Pollen_viability/staging_area/n4x_batch_v1.zip")
    print("   5. Re-run training:")
    print("      ./deploy_training.sh")
    print("      (train_model.py will auto-merge via merge_staged_data())")


if __name__ == "__main__":
    main()
