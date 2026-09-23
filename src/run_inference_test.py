#!/usr/bin/env python3
"""
run_inference_test.py  — Single-image inference test for the new pollen model
═══════════════════════════════════════════════════════════════════════════════
Downloads ONE tile from S3, runs YOLO segmentation inference with the new model,
and uploads two artefacts back to S3:
  - <run_prefix>/test_input.jpg      — the raw input tile
  - <run_prefix>/test_output.jpg     — annotated result (bboxes + masks + labels)
  - <run_prefix>/test_results.json   — structured detection payload

Usage (K8s): deploy via deploy_inference_test.sh
Usage (local):
  MODEL_PATH=/path/to/best.pt python src/run_inference_test.py

Environment variables (all optional, sensible defaults):
  S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET
  S3_MODEL_KEY   — S3 key for best.pt
  MODEL_PATH     — local destination for the model (default: /tmp/best.pt)
  TEST_TILE_KEY  — specific tile key to test (default: auto-select first tile)
  TILE_PREFIX    — prefix to search for tiles (default: Ostatni/Pollen_viability/tiles_640/)
  OUTPUT_PREFIX  — S3 prefix for results   (default: Ostatni/Pollen_viability/inference_tests/<timestamp>/)
  CONF           — confidence threshold     (default: 0.25)
  IOU            — IoU NMS threshold        (default: 0.70)
  MAX_DIM        — max box side px          (default: 130)
"""

import os, io, json, sys, time, ssl, urllib.request
from pathlib import Path
from datetime import datetime

import boto3
from botocore.client import Config
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── config ────────────────────────────────────────────────────────────────────
S3_ENDPOINT  = os.environ.get("S3_ENDPOINT",  "https://s3.cl4.du.cesnet.cz")
AWS_ACCESS   = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET   = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_BUCKET    = os.environ.get("S3_BUCKET",    "bucket")
S3_MODEL_KEY = os.environ.get("S3_MODEL_KEY",
    "Ostatni/Pollen_viability/trained_models/pollen_train_20260923_1457/weights/best.pt")
MODEL_PATH   = os.environ.get("MODEL_PATH",   "/tmp/best.pt")
TEST_TILE_KEY= os.environ.get("TEST_TILE_KEY", "")
TILE_PREFIX  = os.environ.get("TILE_PREFIX",  "Ostatni/Pollen_viability/tiles_640/")
TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PREFIX= os.environ.get("OUTPUT_PREFIX",
    f"Ostatni/Pollen_viability/inference_tests/{TIMESTAMP}/")
CONF         = float(os.environ.get("CONF",  "0.25"))
IOU          = float(os.environ.get("IOU",   "0.70"))
MAX_DIM      = int(os.environ.get("MAX_DIM", "130"))

# Class labels & BGR colours
CLASS_MAP    = {0: ("viable",       (0, 220, 80)),
                1: ("non_viable",   (0, 60, 230)),
                2: ("intermediate", (0, 200, 255))}

VALID_EXT    = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')


# ── helpers ───────────────────────────────────────────────────────────────────

def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS,
        aws_secret_access_key=AWS_SECRET,
        config=Config(signature_version="s3v4", max_pool_connections=8),
    )


def download_model(s3):
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already cached at {MODEL_PATH}")
        return
    print(f"⬇️  Downloading model from S3: {S3_MODEL_KEY}")
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    print(f"   Saved → {MODEL_PATH}  ({size_mb:.1f} MB)")


def pick_tile(s3) -> str:
    """Return TEST_TILE_KEY if set, otherwise find the first real tile in S3."""
    if TEST_TILE_KEY:
        print(f"📌 Using specified tile: {TEST_TILE_KEY}")
        return TEST_TILE_KEY

    print(f"🔍 Auto-selecting a tile from {TILE_PREFIX} …")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=TILE_PREFIX):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith(VALID_EXT) and not k.endswith("_det.json"):
                print(f"   Selected: {k}")
                return k
    raise SystemExit("❌ No tiles found under TILE_PREFIX — nothing to test.")


def load_tile(s3, tile_key: str) -> np.ndarray:
    print(f"⬇️  Downloading tile: {tile_key}")
    resp = s3.get_object(Bucket=S3_BUCKET, Key=tile_key)
    data = resp["Body"].read()
    pil  = Image.open(io.BytesIO(data)).convert("RGB")
    bgr  = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    print(f"   Tile shape: {bgr.shape[1]}×{bgr.shape[0]} px")
    return bgr


def run_inference(model, bgr_img: np.ndarray):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔮 Running inference on device: {device}")
    t0 = time.time()
    with torch.no_grad():
        results = model(bgr_img, conf=CONF, iou=IOU, agnostic_nms=False,
                        verbose=False, device=device)
    elapsed = time.time() - t0
    result  = results[0]
    n_boxes = len(result.boxes) if result.boxes is not None else 0
    print(f"   Inference time : {elapsed*1000:.0f} ms")
    print(f"   Raw detections : {n_boxes}")
    return result


def build_annotated(bgr_img: np.ndarray, result) -> np.ndarray:
    """Draw segmentation masks + bboxes + labels on the image."""
    img_h, img_w = bgr_img.shape[:2]
    out = bgr_img.copy()

    if result.boxes is None or len(result.boxes) == 0:
        print("   ⚠️  No detections — returning clean tile.")
        return out

    kept = 0
    for i in range(len(result.boxes)):
        x1, y1, x2, y2 = [float(v) for v in result.boxes.xyxy[i].tolist()]
        conf   = float(result.boxes.conf[i])
        cls_id = int(result.boxes.cls[i])

        # Filter oversized boxes
        if (x2 - x1) > MAX_DIM or (y2 - y1) > MAX_DIM:
            continue

        label, bgr_col = CLASS_MAP.get(cls_id, (f"cls{cls_id}", (200, 200, 200)))
        kept += 1

        # Draw filled semi-transparent mask if available
        if result.masks is not None and i < len(result.masks.xyn):
            pts_norm = result.masks.xyn[i]
            pts_px   = (pts_norm * np.array([img_w, img_h])).astype(np.int32)
            overlay  = out.copy()
            cv2.fillPoly(overlay, [pts_px], bgr_col)
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
            cv2.polylines(out, [pts_px], True, bgr_col, 2)

        # Draw bounding box
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), bgr_col, 2)

        # Label
        tag  = f"{label} {conf:.2f}"
        tw, th = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        ty   = max(int(y1) - 4, th + 2)
        cv2.rectangle(out, (int(x1), ty - th - 2), (int(x1) + tw + 2, ty + 2), bgr_col, -1)
        cv2.putText(out, tag, (int(x1) + 1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)

    print(f"   Kept (after MAX_DIM filter): {kept}")
    return out


def build_det_json(result, bgr_img: np.ndarray, tile_key: str) -> dict:
    img_h, img_w = bgr_img.shape[:2]
    boxes_out, masks_out, stats = [], [], {}

    if result.boxes is not None:
        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = [float(v) for v in result.boxes.xyxy[i].tolist()]
            conf   = float(result.boxes.conf[i])
            cls_id = int(result.boxes.cls[i])
            label  = CLASS_MAP.get(cls_id, (f"cls{cls_id}", None))[0]

            if (x2 - x1) > MAX_DIM or (y2 - y1) > MAX_DIM:
                continue

            boxes_out.append([round(x1), round(y1), round(x2), round(y2),
                               round(conf, 4), cls_id])

            if result.masks is not None and i < len(result.masks.xyn):
                poly = [[round(float(p[0]), 5), round(float(p[1]), 5)]
                        for p in result.masks.xyn[i]]
            else:
                nw, nh = img_w or 1, img_h or 1
                poly = [[x1/nw, y1/nh], [x2/nw, y1/nh],
                        [x2/nw, y2/nh], [x1/nw, y2/nh]]
            masks_out.append(poly)
            stats[label] = stats.get(label, 0) + 1

    return {
        "model":       os.path.basename(S3_MODEL_KEY),
        "tile_key":    tile_key,
        "conf":        CONF,
        "iou":         IOU,
        "max_dim":     MAX_DIM,
        "n_detections": len(boxes_out),
        "class_counts": stats,
        "boxes":       boxes_out,
        "masks_xyn":   masks_out,
    }


def upload_robust(s3_client, body_bytes: bytes, bucket: str, key: str,
                  content_type: str = "application/octet-stream"):
    """Upload bytes to S3 via presigned PUT (works around CESNET SSL quirks)."""
    try:
        url = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=600,
        )
        req = urllib.request.Request(url, data=body_bytes, method="PUT")
        req.add_header("Content-Length", str(len(body_bytes)))
        req.add_header("Content-Type", content_type)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        with urllib.request.urlopen(req, context=ctx) as resp:
            if resp.status == 200:
                print(f"   ✅ Uploaded → s3://{bucket}/{key}")
            else:
                print(f"   ⚠️  HTTP {resp.status} for {key}")
    except Exception as exc:
        # Fallback: direct boto3 put
        print(f"   ⚠️  Presigned upload failed ({exc}), trying boto3 put_object …")
        s3_client.put_object(Bucket=bucket, Key=key, Body=body_bytes,
                             ContentType=content_type)
        print(f"   ✅ Uploaded → s3://{bucket}/{key}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not AWS_ACCESS or not AWS_SECRET:
        raise SystemExit("❌ AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set.")

    print("=" * 65)
    print("🌸  Pollen Viability — Single-Image Inference Test")
    print("=" * 65)
    print(f"   Model key : {S3_MODEL_KEY}")
    print(f"   Output    : s3://{S3_BUCKET}/{OUTPUT_PREFIX}")
    print(f"   conf={CONF}  iou={IOU}  max_dim={MAX_DIM}px")
    print()

    s3 = make_s3()

    # 1. Download model
    download_model(s3)

    # 2. Load YOLO
    import torch
    from ultralytics import YOLO
    model  = YOLO(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 YOLO model loaded  (device: {device})")
    print()

    # 3. Pick & download tile
    tile_key = pick_tile(s3)
    bgr_img  = load_tile(s3, tile_key)

    # 4. Run inference
    result   = run_inference(model, bgr_img)

    # 5. Build annotated image
    annotated = build_annotated(bgr_img, result)

    # 6. Build JSON payload
    det_json  = build_det_json(result, bgr_img, tile_key)

    # 7. Print summary
    print()
    print("─" * 65)
    print("📊  Detection Summary")
    print(f"   Total kept  : {det_json['n_detections']}")
    for cls_label, count in det_json["class_counts"].items():
        symbol = "🟢" if cls_label == "viable" else ("🔴" if cls_label == "non_viable" else "🟡")
        print(f"   {symbol} {cls_label:<14}: {count}")
    print("─" * 65)
    print()

    # 8. Upload artefacts to S3
    s3_client = s3

    # 8a. Input tile
    resp_tile = s3.get_object(Bucket=S3_BUCKET, Key=tile_key)
    tile_bytes = resp_tile["Body"].read()
    upload_robust(s3_client, tile_bytes, S3_BUCKET,
                  OUTPUT_PREFIX + "test_input.jpg", "image/jpeg")

    # 8b. Annotated output
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        upload_robust(s3_client, buf.tobytes(), S3_BUCKET,
                      OUTPUT_PREFIX + "test_output.jpg", "image/jpeg")

    # 8c. JSON results
    json_bytes = json.dumps(det_json, indent=2).encode("utf-8")
    upload_robust(s3_client, json_bytes, S3_BUCKET,
                  OUTPUT_PREFIX + "test_results.json", "application/json")

    print()
    print("✅  Inference test complete!")
    print(f"   Results in S3: s3://{S3_BUCKET}/{OUTPUT_PREFIX}")
    print(f"   View output  : test_output.jpg  (annotated detections)")
    print(f"   Raw JSON     : test_results.json")


if __name__ == "__main__":
    main()
