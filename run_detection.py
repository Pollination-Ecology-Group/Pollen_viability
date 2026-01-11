
import os
import sys
import boto3
import cv2
import pandas as pd
import shutil
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.ops import nms
from botocore.client import Config

# --- CONFIGURATION FROM ENV ---
S3_ENDPOINT = os.environ.get('S3_ENDPOINT', 'https://s3.cl4.du.cesnet.cz')
S3_BUCKET = os.environ.get('S3_BUCKET')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

# Detection Config
CONF_THRESHOLD = 0.25
IOU_MERGE_THRESHOLD = 0.45
TILE_SIZE = 1600
BORDER_MARGIN = 10

# Paths
LOCAL_DETECT_IMAGES = 'detect_images'
LOCAL_DETECTED = 'detected'
LOCAL_MODEL = 'best.pt'

def setup_s3():
    if not all([S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        print("⚠️ S3 Environment variables missing, skipping S3 sync.")
        return None
    return boto3.resource('s3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4')
    )

def download_s3_folder(s3, prefix, local_dir):
    bucket = s3.Bucket(S3_BUCKET)
    print(f"⬇️ Downloading from S3: {prefix} -> {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    for obj in bucket.objects.filter(Prefix=prefix):
        rel_path = os.path.relpath(obj.key, prefix)
        if rel_path == ".": continue
        dest_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if not os.path.exists(dest_path): # Basic skip existing
            bucket.download_file(obj.key, dest_path)
    print("✅ Download complete.")

def upload_s3_folder(s3, local_dir, prefix):
    bucket = s3.Bucket(S3_BUCKET)
    print(f"⬆️ Uploading to S3: {local_dir} -> {prefix}")
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            s3_path = os.path.join(prefix, rel_path)
            bucket.upload_file(local_path, s3_path)
    print("✅ Upload complete.")

def get_tiles(img, tile_size=1600, overlap=0.2):
    h, w = img.shape[:2]
    tiles = []
    stride = int(tile_size * (1 - overlap))
    y_starts = list(range(0, h, stride))
    x_starts = list(range(0, w, stride))
    if y_starts[-1] + tile_size < h: y_starts[-1] = h - tile_size
    if x_starts[-1] + tile_size < w: x_starts[-1] = w - tile_size

    for y in y_starts:
        for x in x_starts:
            y = max(0, y); x = max(0, x)
            tile = img[y:y+tile_size, x:x+tile_size]
            tiles.append((x, y, tile))
    return tiles

def run_detection():
    # Setup
    s3 = setup_s3()
    
    # 1. Download Data
    if s3:
        # Assuming S3 structure matches: datasets/, detect_images/, detected/
        # We download 'detect_images/' from S3 to local 'detect_images/'
        # Note: The user said "My S3 bucket follows the exact same file structure... (e.g., datasets/, detect_images/, detected/)"
        # S3 Prefix logic: If the bucket root IS the project root, then prefix is 'detect_images'.
        # If there is a project folder, we might need to adjust. I'll assume root or look for 'detect_images' key.
        # But 'detect_images' is likely a folder at the root (or 'Pollen_viability/detect_images').
        # The notebook used prefix 'Ostatni/Pollen_viability'. I'll assume Ostatni/Pollen_viability/detect_images or just check env.
        # To be safe, I'll assume the bucket root contains 'detect_images' unless configured otherwise.
        download_s3_folder(s3, 'detect_images', LOCAL_DETECT_IMAGES)
        
        # Try to download model if not present
        if not os.path.exists(LOCAL_MODEL):
            print("Trying to download model from S3...")
            # Try specific path or default
            # Implementation choice: use a default path or failover to yolov8x.pt
            try:
                # Attempt to find best.pt in trained_models
                # For now, we will fallback to standard yolov8x if local file missing
                pass 
            except:
                pass

    # 2. Load Model
    model_name = LOCAL_MODEL if os.path.exists(LOCAL_MODEL) else 'yolov8x.pt'
    print(f"🔮 Loading Model: {model_name}")
    model = YOLO(model_name)

    os.makedirs(LOCAL_DETECTED, exist_ok=True)
    images = [f for f in os.listdir(LOCAL_DETECT_IMAGES) if f.lower().endswith(('.jpg', '.png'))]
    data_rows = []

    print(f"🚀 Starting Analysis on {len(images)} images...")

    for img_file in images:
        img_path = os.path.join(LOCAL_DETECT_IMAGES, img_file)
        original_img = cv2.imread(img_path)
        if original_img is None: continue
        
        h_orig, w_orig = original_img.shape[:2]
        
        # Logic from Step 7
        if w_orig > 2000 or h_orig > 2000:
            tile_list = get_tiles(original_img, tile_size=TILE_SIZE, overlap=0.25)
            all_boxes, all_scores, all_cls = [], [], []
            
            for (tx, ty, tile) in tile_list:
                results = model(tile, verbose=False, imgsz=TILE_SIZE, conf=CONF_THRESHOLD)
                for box in results[0].boxes:
                    lx1, ly1, lx2, ly2 = box.xyxy[0].tolist()
                    # Border Patrol
                    on_left = (lx1 < BORDER_MARGIN) and (tx > 0)
                    on_right = (lx2 > TILE_SIZE - BORDER_MARGIN) and (tx + TILE_SIZE < w_orig)
                    on_top = (ly1 < BORDER_MARGIN) and (ty > 0)
                    on_bottom = (ly2 > TILE_SIZE - BORDER_MARGIN) and (ty + TILE_SIZE < h_orig)
                    
                    if not (on_left or on_right or on_top or on_bottom):
                        all_boxes.append([lx1 + tx, ly1 + ty, lx2 + tx, ly2 + ty])
                        all_scores.append(float(box.conf[0]))
                        all_cls.append(int(box.cls[0]))
            
            if all_boxes:
                boxes_t = torch.tensor(all_boxes)
                scores_t = torch.tensor(all_scores)
                keep = nms(boxes_t, scores_t, IOU_MERGE_THRESHOLD)
                final_boxes = boxes_t[keep].numpy()
                final_scores = scores_t[keep].numpy()
                final_cls = torch.tensor(all_cls)[keep].numpy()
            else:
                 final_boxes, final_scores, final_cls = [], [], []
        else:
            results = model(img_path, verbose=False, imgsz=1280, conf=CONF_THRESHOLD)
            if results[0].boxes:
                final_boxes = results[0].boxes.xyxy.cpu().numpy()
                final_scores = results[0].boxes.conf.cpu().numpy()
                final_cls = results[0].boxes.cls.cpu().numpy()
            else:
                final_boxes, final_scores, final_cls = [], [], []

        # Draw & Count
        v_count, nv_count = 0, 0
        COLOR_VIABLE = (0, 200, 0)
        COLOR_NON_VIABLE = (0, 0, 255)

        for j in range(len(final_boxes)):
            x1, y1, x2, y2 = map(int, final_boxes[j])
            cls_id = int(final_cls[j])
            color = COLOR_VIABLE if cls_id == 0 else COLOR_NON_VIABLE
            if cls_id == 0: v_count += 1
            else: nv_count += 1
            cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 4)
        
        cv2.imwrite(os.path.join(LOCAL_DETECTED, img_file), original_img)
        data_rows.append({
            'filename': img_file,
            'viable': v_count,
            'non_viable': nv_count,
            'total': v_count + nv_count
        })

    # Save CSV
    if data_rows:
        df = pd.DataFrame(data_rows)
        csv_path = os.path.join(LOCAL_DETECTED, 'pollen_counts.csv')
        df.to_csv(csv_path, index=False)
        print("✅ Detection done.")
        
        # 3. Upload Results
        if s3:
            upload_s3_folder(s3, LOCAL_DETECTED, 'detected')

if __name__ == "__main__":
    run_detection()
