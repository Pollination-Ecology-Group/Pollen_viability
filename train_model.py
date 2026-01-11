
import os
import sys
import boto3
import shutil
import glob
import random
import cv2
import numpy as np
import argparse
from botocore.client import Config
from ultralytics import YOLO
from datetime import datetime

# --- CONFIGURATION FROM ENV ---
S3_ENDPOINT = os.environ.get('S3_ENDPOINT', 'https://s3.cl4.du.cesnet.cz')
S3_BUCKET = os.environ.get('S3_BUCKET')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

# Paths
LOCAL_ROOT = 'Pollen_viability'
DATASET_ROOT = os.path.join(LOCAL_ROOT, 'datasets/pollen_v1')
STAGING_AREA = os.path.join(LOCAL_ROOT, 'staged_area')
SMUDGES_RAW = os.path.join(LOCAL_ROOT, 'smudges_raw')
TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
VAL_DIR = os.path.join(DATASET_ROOT, 'val')
VIS_DIR = 'visualizations'

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

def download_s3_prefix(s3, prefix, local_dir):
    bucket = s3.Bucket(S3_BUCKET)
    print(f"⬇️ Downloading {prefix} -> {local_dir}")
    count = 0
    for obj in bucket.objects.filter(Prefix=prefix):
        rel_path = os.path.relpath(obj.key, prefix)
        if rel_path == "." or rel_path.startswith("_"): continue
        dest_path = os.path.join(local_dir, rel_path)
        if not os.path.exists(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            bucket.download_file(obj.key, dest_path)
            count += 1
            if count % 100 == 0: print(f"   Downloaded {count}...", end='\r')
    print(f"✅ Downloaded {count} new files.")

def merge_staged_data():
    print("🔄 Checking for new data in staging area...")
    if not os.path.exists(STAGING_AREA): return
    
    zips = glob.glob(os.path.join(STAGING_AREA, "*.zip"))
    if not zips:
        print("ℹ️ No new zips found.")
        return

    # Process first zip found
    zip_path = zips[0]
    print(f"1️⃣ Processing: {os.path.basename(zip_path)}")
    
    temp_dir = os.path.join(LOCAL_ROOT, 'temp_merge')
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    shutil.unpack_archive(zip_path, temp_dir)
    
    # Simple recursive finder
    found_pairs = []
    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png')):
                base = os.path.splitext(f)[0]
                label_path = os.path.join(root, base + ".txt")
                if os.path.exists(label_path):
                    found_pairs.append((os.path.join(root, f), label_path))
    
    print(f"   Found {len(found_pairs)} pairs.")
    
    # Merge logic
    random.shuffle(found_pairs)
    split_idx = int(len(found_pairs) * 0.2) # 20% val
    val_batch = found_pairs[:split_idx]
    train_batch = found_pairs[split_idx:]
    
    for batch, dest in [(train_batch, TRAIN_DIR), (val_batch, VAL_DIR)]:
        img_dest = os.path.join(dest, 'images')
        lbl_dest = os.path.join(dest, 'labels')
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)
        for img, lbl in batch:
            shutil.copy2(img, img_dest)
            shutil.copy2(lbl, lbl_dest)
            
    print("✅ Merge complete.")

def generate_synthetic_negatives():
    print("🧪 Generating Synthetic Negatives...")
    if not os.path.exists(SMUDGES_RAW):
        print("⚠️ No raw smudges found.")
        return

    raw_files = [f for f in os.listdir(SMUDGES_RAW) if f.lower().endswith(('.jpg', '.png'))]
    if not raw_files: return

    CANVAS_SIZE = 640
    BG_COLOR = (200, 200, 200)
    
    count = 0
    for fname in raw_files:
        img_path = os.path.join(SMUDGES_RAW, fname)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Resize logic
        h, w = img.shape[:2]
        scale = min(CANVAS_SIZE/h, CANVAS_SIZE/w) * 0.8
        new_w, new_h = int(w*scale), int(h*scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), BG_COLOR, dtype=np.uint8)
        y_off = (CANVAS_SIZE - new_h) // 2
        x_off = (CANVAS_SIZE - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
        # Save
        is_val = random.random() < 0.2
        target_dir = VAL_DIR if is_val else TRAIN_DIR
        
        out_name = f"syn_neg_{fname}"
        cv2.imwrite(os.path.join(target_dir, 'images', out_name), canvas)
        # Empty label
        with open(os.path.join(target_dir, 'labels', os.path.splitext(out_name)[0]+'.txt'), 'w') as f:
            pass
        count += 1
    print(f"✅ Generated {count} synthetic samples.")

def visualize_dataset(num_samples=None):
    print("🎨 Generating Dataset Visualizations (Ground Truth)...")
    os.makedirs(VIS_DIR, exist_ok=True)
    
    # Define colors
    COLOR_MAP = {0: (0, 255, 0), 1: (0, 0, 255)} # Green=Viable, Red=Non-Viable
    
    for split_name, split_dir in [('train', TRAIN_DIR), ('val', VAL_DIR)]:
        save_dir = os.path.join(VIS_DIR, f"{split_name}_samples")
        os.makedirs(save_dir, exist_ok=True)
        
        img_dir = os.path.join(split_dir, 'images')
        lbl_dir = os.path.join(split_dir, 'labels')
        
        if not os.path.exists(img_dir): continue
        
        all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not all_imgs: continue
        
        if num_samples:
             samples = random.sample(all_imgs, min(len(all_imgs), num_samples))
        else:
             samples = all_imgs
        
        for img_file in samples:
            img_path = os.path.join(img_dir, img_file)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + '.txt')
            
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]
            
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                            
                            # Convert YOLO to xyxy
                            x1 = int((cx - bw/2) * w)
                            y1 = int((cy - bh/2) * h)
                            x2 = int((cx + bw/2) * w)
                            y2 = int((cy + bh/2) * h)
                            
                            color = COLOR_MAP.get(cls, (255, 255, 255))
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img, str(cls), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imwrite(os.path.join(save_dir, img_file), img)
    print("✅ Visualization complete.")

def visualize_predictions(model, split='val', num_samples=None):
    print("🎨 Generating Prediction Visualizations...")
    # Using validation set as proxy for test if test doesn't exist standardly in this structure
    target_dir = VAL_DIR # Could verify if test exists
    img_dir = os.path.join(target_dir, 'images')
    
    if not os.path.exists(img_dir): return

    save_dir = os.path.join(VIS_DIR, 'predictions')
    os.makedirs(save_dir, exist_ok=True)

    all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    if num_samples:
        samples = random.sample(all_imgs, min(len(all_imgs), num_samples))
    else:
        samples = all_imgs
    
    for img_file in samples:
        img_path = os.path.join(img_dir, img_file)
        # Run inference
        results = model(img_path, verbose=False)
        # Plot
        res_plotted = results[0].plot()
        cv2.imwrite(os.path.join(save_dir, f"pred_{img_file}"), res_plotted)
    print("✅ Prediction viz complete.")

def upload_results(s3, train_name):
    bucket = s3.Bucket(S3_BUCKET)
    
    # 1. Weights & Metrics
    local_run_dir = os.path.join('runs/detect', train_name)
    s3_prefix_run = f"Ostatni/Pollen_viability/trained_models/{train_name}"
    
    print(f"⬆️ Uploading training results to {s3_prefix_run}...")
    if os.path.exists(local_run_dir):
        for root, _, files in os.walk(local_run_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_run_dir)
                bucket.upload_file(local_path, f"{s3_prefix_run}/{rel_path}")

    # 2. Visualizations
    s3_prefix_vis = f"Ostatni/Pollen_viability/trained_models/{train_name}/visualizations"
    print(f"⬆️ Uploading visualizations to {s3_prefix_vis}...")
    if os.path.exists(VIS_DIR):
        for root, _, files in os.walk(VIS_DIR):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, VIS_DIR)
                bucket.upload_file(local_path, f"{s3_prefix_vis}/{rel_path}")
                
    print("✅ Upload complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true', help="Skip actual training")
    args = parser.parse_args()

    s3 = setup_s3()
    
    # 1. Sync Data
    if s3:
        # We assume the bucket structure: Ostatni/Pollen_viability/datasets...
        # Adjust prefixes to match user's S3 structure
        download_s3_prefix(s3, 'Ostatni/Pollen_viability/datasets/pollen_v1', DATASET_ROOT)
        download_s3_prefix(s3, 'Ostatni/Pollen_viability/staging_area', STAGING_AREA)
        download_s3_prefix(s3, 'Ostatni/Pollen_viability/smudges_raw', SMUDGES_RAW)

    # 2. Prep Data
    merge_staged_data()
    generate_synthetic_negatives()
    visualize_dataset(num_samples=None) # Generate GT samples for ALL images

    # 3. Train
    if not args.dry_run:
        print("🚀 Starting Training...")
        # Check for GPU
        device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"   Device: {device}")
        
        model = YOLO('yolov8x.pt')
        run_name = f"pollen_train_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Ensure data.yaml exists
        yaml_path = os.path.join(DATASET_ROOT, 'data.yaml')
        if not os.path.exists(yaml_path):
             # Basic yaml creation if missing
             with open(yaml_path, 'w') as f:
                 f.write(f"path: {os.path.abspath(DATASET_ROOT)}\ntrain: train/images\nval: val/images\nnames:\n  0: viable\n  1: non_viable")

        results = model.train(
            data=yaml_path,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=640,
            device=device,
            name=run_name,
            project='runs/detect',
            agnostic_nms=True,
            mosaic=1.0,
            close_mosaic=20
        )
        
        # 4. Independent Evaluation
        print("📊 Running Evaluation...")
        metrics = model.val(split='val') # or 'test' if available
        print(f"   mAP50-95: {metrics.box.map}")
        
        # 4b. Visualize Predictions
        visualize_predictions(model, split='val', num_samples=None)        
        # 5. Backup
        if s3:
            upload_results(s3, run_name)
    else:
        print("⚠️ Dry run mode: Skipping actual training.")

if __name__ == "__main__":
    import torch # import here to avoid delay
    main()
