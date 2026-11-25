#!/usr/bin/env python3
"""Stage tiles flagged during QC review or uncertain detections for label_app.

Pulls tiles from Review/decisions/ marked with relabel: true, needs_training, or poor verdict,
and stages them in Relabel/Tiles/ and Relabel/Prelabels/ so they are immediately available
with priority in the Labeling App.

Usage:
    python src/prepare_relabel_tiles.py
    python src/prepare_relabel_tiles.py --max-tiles 50
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import boto3
from botocore.client import Config

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
S3_BUCKET = os.environ.get("S3_BUCKET", "peg-bucket")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
REVIEW_PREFIX = os.environ.get("REVIEW_PREFIX", "Ostatni/Pollen_viability/Review").rstrip("/")
RELABEL_PREFIX = "Ostatni/Pollen_viability/Relabel/"
RAW_TILES_PREFIX = "Ostatni/Pollen_viability/tiles_640/"


def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare tiles for label_app")
    parser.add_argument("--max-tiles", type=int, default=100, help="Maximum number of tiles to stage")
    parser.add_argument("--sample-qc", action="store_true", help="Stage representative sample from QC manifest")
    args = parser.parse_args()

    s3 = get_s3()

    print(f"🔍 Reading Review decisions from s3://{S3_BUCKET}/{REVIEW_PREFIX}/decisions/ ...")
    manifest_key = f"{REVIEW_PREFIX}/manifest.json"
    try:
        manifest_obj = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=manifest_key)["Body"].read())
        items_by_id = {it["id"]: it for it in manifest_obj.get("items", [])}
    except Exception as e:
        print(f"⚠️ Could not read QC review manifest: {e}")
        items_by_id = {}

    flagged_tiles = []
    pfx = f"{REVIEW_PREFIX}/decisions/"
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=pfx):
            for o in page.get("Contents", []):
                if o["Key"].endswith(".json"):
                    try:
                        dec = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=o["Key"])["Body"].read())
                        if dec.get("relabel") or dec.get("needs_training") or dec.get("verdict") == "poor":
                            item_id = o["Key"][len(pfx):].split("/")[1]
                            it = items_by_id.get(item_id, {})
                            tile_key = it.get("tile_key", "")
                            if tile_key:
                                flagged_tiles.append((item_id, tile_key, dec, it.get("det_key")))
                    except Exception:
                        pass
    except Exception as e:
        print(f"⚠️ Could not list decisions: {e}")

    print(f"📋 Found {len(flagged_tiles)} tiles flagged for relabeling during QC review.")
    if not flagged_tiles:
        print("💡 No tiles have been flagged in QC review yet (reviewers can press 'R' in review_app).")
        print(f"🚀 Staging representative stratified sample (up to {args.max_tiles} tiles) from QC manifest...")
        for it in manifest_obj.get("items", [])[:args.max_tiles]:
            flagged_tiles.append((it["id"], it["tile_key"], {"relabel": False, "source": "qc_sample"}, it.get("det_key")))

    # Deduplicate by tile key
    seen = set()
    staged = 0
    for item_id, tile_key, dec, det_key_hint in flagged_tiles:
        if tile_key in seen:
            continue
        seen.add(tile_key)

        # Determine relative name
        rel = tile_key
        if rel.startswith(RAW_TILES_PREFIX):
            rel = rel[len(RAW_TILES_PREFIX):]
        if rel.endswith(".jpg"):
            rel = rel[:-4]

        # Copy tile image to Relabel/Tiles/
        target_img_key = f"{RELABEL_PREFIX}Tiles/{rel}.jpg"
        target_pre_key = f"{RELABEL_PREFIX}Prelabels/{rel}.json"

        try:
            s3.copy_object(
                Bucket=S3_BUCKET,
                Key=target_img_key,
                CopySource={"Bucket": S3_BUCKET, "Key": tile_key}
            )

            # Look for existing _det.json
            det_key = det_key_hint or f"{RAW_TILES_PREFIX}{rel}_det.json"
            masks = []
            try:
                raw_det = json.loads(s3.get_object(Bucket=S3_BUCKET, Key=det_key)["Body"].read())
                if "boxes" in raw_det:
                    boxes = raw_det.get("boxes", [])
                    polys = raw_det.get("masks_xyn", [])
                    for idx, b in enumerate(boxes):
                        if len(b) > 4 and b[4] < 0.35:
                            continue
                        poly = None
                        if idx < len(polys) and polys[idx] and len(polys[idx]) >= 3:
                            poly = [[round(float(pt[0]) * 640.0, 1), round(float(pt[1]) * 640.0, 1)] for pt in polys[idx]]
                        elif len(b) >= 4:
                            poly = [[round(float(b[0]), 1), round(float(b[1]), 1)],
                                    [round(float(b[2]), 1), round(float(b[1]), 1)],
                                    [round(float(b[2]), 1), round(float(b[3]), 1)],
                                    [round(float(b[0]), 1), round(float(b[3]), 1)]]
                        if poly:
                            c_idx = int(b[5]) if len(b) > 5 else 0
                            cls_names = ["viable", "non_viable", "intermediate", "debris"]
                            cls_name = cls_names[c_idx] if c_idx < len(cls_names) else "viable"
                            masks.append(dict(
                                id=idx, polygon=poly, cls=cls_name,
                                score=round(float(b[4]) if len(b) > 4 else 0.8, 3),
                                sources=["model"], eq_diam_um=25.0
                            ))
                elif "grains" in raw_det:
                    for idx, g in enumerate(raw_det.get("grains", [])):
                        poly = g.get("polygon")
                        if not poly and "bbox" in g:
                            b = g["bbox"]
                            poly = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
                        if poly:
                            masks.append(dict(
                                id=idx, polygon=poly, cls=g.get("class", "viable"),
                                score=g.get("confidence", 0.75), sources=["model"],
                                eq_diam_um=round(float(g.get("eq_diam_um", 25.0)), 1)
                            ))
            except Exception:
                pass

            s3.put_object(
                Bucket=S3_BUCKET, Key=target_pre_key,
                Body=json.dumps({"masks": masks}, indent=2).encode(),
                ContentType="application/json"
            )
            staged += 1
            print(f"   Staged [{staged}]: {rel} ({len(masks)} pre-labels)")
        except Exception as e:
            print(f"⚠️ Failed to stage {tile_key}: {e}")

        if staged >= args.max_tiles:
            break

    print(f"\n✅ Staged {staged} tiles into s3://{S3_BUCKET}/{RELABEL_PREFIX}Tiles/")
    print(f"   Launch label_app to annotate: python src/label_app/app.py")


if __name__ == "__main__":
    main()
