import cv2
import boto3
import pandas as pd
import torch
import glob
import os
import urllib.request
import ssl
from ultralytics import YOLO
from torchvision.ops import nms
from botocore.client import Config

# --- CONFIGURATION ---
S3_ENDPOINT = os.environ.get('S3_ENDPOINT')
S3_BUCKET = os.environ.get('S3_BUCKET')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

LOCAL_DETECT_IMAGES = 'detect_images'
LOCAL_DETECTED = 'detected'
LOCAL_MODEL = 'best.pt'

CONF_THRESHOLD = 0.25
IOU_MERGE_THRESHOLD = 0.45
TILE_SIZE = 1600
BORDER_MARGIN = 10

def setup_s3():
    if not all([S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        print("⚠️ S3 Credentials missing. Running in offline mode.")
        return None
    return boto3.resource('s3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4')
    )

def upload_file_robust(s3_client, local_path, bucket, key):
    """
    Robust upload using Presigned URL + urllib PUT to bypass
    MissingContentLength / SSL issues with CESNET S3.
    """
    try:
        # 1. Generate Presigned URL
        url = s3_client.generate_presigned_url('put_object', 
                                             Params={'Bucket': bucket, 'Key': key}, 
                                             ExpiresIn=3600)
        
        # 2. Get file size
        size = os.path.getsize(local_path)
        
        # 3. Create Request with Explicit Content-Length
        with open(local_path, 'rb') as data:
            req = urllib.request.Request(url, data=data, method='PUT')
            req.add_header('Content-Length', str(size))
            
            # 4. Context to ignore SSL
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            with urllib.request.urlopen(req, context=ctx) as f:
                if f.status == 200:
                    print(f"   -> Uploaded {os.path.basename(local_path)} ✅")
                else:
                    print(f"   -> ⚠️ Upload rejected {os.path.basename(local_path)}: Status {f.status}")
                    
    except Exception as e:
        print(f"❌ Failed to upload {local_path}: {e}")
        pass

def download_s3_folder(s3, prefix, local_dir):
    bucket = s3.Bucket(S3_BUCKET)
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"⬇️ Downloading from S3: {prefix} -> {local_dir}")
    for obj in bucket.objects.filter(Prefix=prefix):
        target = os.path.join(local_dir, os.path.relpath(obj.key, prefix))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target), exist_ok=True)
        if obj.key.endswith('/'): continue
        bucket.download_file(obj.key, target)
    print("✅ Download complete.")

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
    s3_resource = setup_s3()
    s3_client = s3_resource.meta.client if s3_resource else None
    
    # 1. Download Data
    if s3_resource:
        try:
            download_s3_folder(s3_resource, 'Ostatni/Pollen_viability/detect_images', LOCAL_DETECT_IMAGES)
        except Exception as e:
             print(f"⚠️ S3 Download Failed (Partial?), processing what we have... Error: {e}")
        
        # Download Model - USER PROVIDED (Note: .ptrom extension found on S3)
        S3_MODEL_KEY = 'Ostatni/Pollen_viability/trained_models/pollen_v1_27/weights/best.ptrom'
        
        if not os.path.exists(LOCAL_MODEL):
            print(f"⬇️ Downloading Model from S3: {S3_MODEL_KEY}...")
            try:
                bucket = s3_resource.Bucket(S3_BUCKET)
                bucket.download_file(S3_MODEL_KEY, LOCAL_MODEL)
                print("✅ Model download complete.")
            except Exception as e:
                print(f"❌ Model Download Failed: {e}")
                print("   ⚠️ Falling back to generic YOLOv8x (Results will be poor!)")

    # 2. Load Model
    model_name = LOCAL_MODEL if os.path.exists(LOCAL_MODEL) else 'yolov8x.pt'
    print(f"🔮 Loading Model: {model_name}")
    model = YOLO(model_name)

    os.makedirs(LOCAL_DETECTED, exist_ok=True)
    images = [f for f in os.listdir(LOCAL_DETECT_IMAGES) if f.lower().endswith(('.jpg', '.png'))]
    data_rows = []

    total_imgs = len(images)
    print(f"🚀 Starting Analysis on {total_imgs} images...")
    if not images:
        print(f"❌ No images found in {LOCAL_DETECT_IMAGES}. Exiting.")
        return 
    
    for i, img_file in enumerate(images):
        if i % 10 == 0:
            print(f"   Processing image {i+1}/{total_imgs} ({(i/total_imgs)*100:.1f}%)...")
        img_path = os.path.join(LOCAL_DETECT_IMAGES, img_file)
        original_img = cv2.imread(img_path)
        if original_img is None: continue
        
        h_orig, w_orig = original_img.shape[:2]
        
        # Processing logic
        if w_orig > 2000 or h_orig > 2000:
            tile_list = get_tiles(original_img, tile_size=TILE_SIZE, overlap=0.25)
            all_boxes, all_scores, all_cls = [], [], []
            
            for (tx, ty, tile) in tile_list:
                results = model(tile, verbose=False, imgsz=TILE_SIZE, conf=CONF_THRESHOLD, project='/app/runs', name='predict')
                for box in results[0].boxes:
                    lx1, ly1, lx2, ly2 = box.xyxy[0].tolist()
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
            results = model(img_path, verbose=False, imgsz=1280, conf=CONF_THRESHOLD, project='/app/runs', name='predict')
            if results[0].boxes:
                final_boxes = results[0].boxes.xyxy.cpu().numpy()
                final_scores = results[0].boxes.conf.cpu().numpy()
                final_cls = results[0].boxes.cls.cpu().numpy()
            else:
                final_boxes, final_scores, final_cls = [], [], []

        v_count, nv_count = 0, 0
        COLOR_VIABLE = (0, 200, 0)
        COLOR_NON_VIABLE = (0, 0, 255)

        for j in range(len(final_boxes)):
            x1, y1, x2, y2 = map(int, final_boxes[j])
            cls_id = int(final_cls[j])
            conf = final_scores[j]
            
            color = COLOR_VIABLE if cls_id == 0 else COLOR_NON_VIABLE
            label = f"{'V' if cls_id == 0 else 'NV'} {conf:.2f}"
            
            if cls_id == 0: v_count += 1
            else: nv_count += 1
            
            cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 4)
            
            # Draw confidence text
            if w_orig < 5000:
                cv2.putText(original_img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        
        out_path = os.path.join(LOCAL_DETECTED, img_file)
        cv2.imwrite(out_path, original_img)
        
        # IMMEDIATE UPLOAD
        if s3_client:
             s3_path = os.path.join('Ostatni/Pollen_viability/detected_images', img_file)
             upload_file_robust(s3_client, out_path, S3_BUCKET, s3_path)

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
        
        # Upload Results (CSV)
        if s3_client:
             upload_file_robust(s3_client, csv_path, S3_BUCKET, 'Ostatni/Pollen_viability/detected_images/pollen_counts.csv')

if __name__ == "__main__":
    run_detection()
