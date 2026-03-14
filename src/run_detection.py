import cv2
import math
import boto3
import pandas as pd
import torch
import glob
import os
import urllib.request
import ssl
import numpy as np
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
S3_MODEL_KEY = os.environ.get('S3_MODEL_KEY', 'Ostatni/Pollen_viability/trained_models/pollen_v1_27/weights/best.ptrom')

CONF_THRESHOLD = 0.25
IOU_MERGE_THRESHOLD = 0.45
TILE_SIZE = 1600
BORDER_MARGIN = 10
EXCLUSION_COLOR = (128, 128, 128) # Grey for excluded particles

def is_touching_edge(mask, img_shape, margin=5):
    """Checks if any point of the mask is too close to the image edge."""
    h, w = img_shape[:2]
    # mask is a numpy array of [x, y] coordinates
    if len(mask) == 0: return False
    
    xs = mask[:, 0]
    ys = mask[:, 1]
    
    if np.any(xs < margin) or np.any(xs > w - margin) or \
       np.any(ys < margin) or np.any(ys > h - margin):
        return True
    return False

def calculate_measurements(mask):
    """Calculates area and equivalent diameter from a polygon mask."""
    # Convert to integer points for cv2
    pts = np.array(mask, np.int32).reshape((-1, 1, 2))
    area = cv2.contourArea(pts)
    # Equivalent diameter: d = 2 * sqrt(area / pi)
    diameter = 2 * math.sqrt(area / math.pi) if area > 0 else 0
    return area, diameter

def check_overlap(mask, other_masks, img_shape, threshold=0.05):
    """
    Checks if a mask overlaps significantly with any other mask.
    Returns True if overlap area > threshold * mask_area.
    """
    h, w = img_shape[:2]
    m_img = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m_img, [np.array(mask, np.int32)], 1)
    
    mask_area = np.sum(m_img)
    if mask_area == 0: return False
    
    for other in other_masks:
        o_img = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(o_img, [np.array(other, np.int32)], 1)
        
        intersection = cv2.bitwise_and(m_img, o_img)
        overlap_area = np.sum(intersection)
        
        if overlap_area > threshold * mask_area:
            return True
    return False

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
        
        # Download Model - Now using environment variable
        print(f"⬇️ Downloading Model from S3: {S3_MODEL_KEY}...")
        
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
    model_name = LOCAL_MODEL if os.path.exists(LOCAL_MODEL) else 'yolov8x-seg.pt'
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
        
        if w_orig > 2000 or h_orig > 2000:
            tile_list = get_tiles(original_img, tile_size=TILE_SIZE, overlap=0.25)
            all_boxes, all_scores, all_cls = [], [], []
            all_masks = []
            
            for (tx, ty, tile) in tile_list:
                results = model(tile, verbose=False, imgsz=TILE_SIZE, conf=CONF_THRESHOLD, project='/app/runs', name='predict')
                
                if results[0].masks is not None:
                    # masks.xy gives coordinates in original image pixel scale
                    for m_idx, mask_coords in enumerate(results[0].masks.xy):
                        if len(mask_coords) == 0: continue
                        
                        box = results[0].boxes[m_idx]
                        lx1, ly1, lx2, ly2 = box.xyxy[0].tolist()
                        on_left = (lx1 < BORDER_MARGIN) and (tx > 0)
                        on_right = (lx2 > TILE_SIZE - BORDER_MARGIN) and (tx + TILE_SIZE < w_orig)
                        on_top = (ly1 < BORDER_MARGIN) and (ty > 0)
                        on_bottom = (ly2 > TILE_SIZE - BORDER_MARGIN) and (ty + TILE_SIZE < h_orig)
                        
                        if not (on_left or on_right or on_top or on_bottom):
                            all_boxes.append([lx1 + tx, ly1 + ty, lx2 + tx, ly2 + ty])
                            all_scores.append(float(box.conf[0]))
                            all_cls.append(int(box.cls[0]))
                            
                            # Shift mask coordinates by tile offset
                            shifted_mask = np.copy(mask_coords)
                            shifted_mask[:, 0] += tx
                            shifted_mask[:, 1] += ty
                            all_masks.append(shifted_mask)
            
            if all_boxes:
                boxes_t = torch.tensor(all_boxes)
                scores_t = torch.tensor(all_scores)
                keep = nms(boxes_t, scores_t, IOU_MERGE_THRESHOLD)
                
                final_boxes = boxes_t[keep].numpy()
                final_scores = scores_t[keep].numpy()
                final_cls = torch.tensor(all_cls)[keep].numpy()
                final_masks = [all_masks[idx] for idx in keep]

                # --- NEW: Robust Visualization for Tiled Results ---
                # Handled in the unified block below
                pass
            else:
                 final_boxes, final_scores, final_cls, final_masks = [], [], [], []
        else:
            # For small images, use standard plotting which is very reliable
            results = model(img_path, verbose=False, imgsz=1280, conf=CONF_THRESHOLD)
            
            if results[0].masks is not None:
                final_boxes = results[0].boxes.xyxy.cpu().numpy()
                final_scores = results[0].boxes.conf.cpu().numpy()
                final_cls = results[0].boxes.cls.cpu().numpy()
                final_masks = results[0].masks.xy
            else:
                final_boxes, final_scores, final_cls, final_masks = [], [], [], []

        # --- UNIFIED PARTICLE PROCESSING ---
        v_count, nv_count = 0, 0
        particle_data = []
        
        # Pre-calculate areas and identify exclusions
        for j in range(len(final_masks)):
            mask = final_masks[j]
            box = final_boxes[j]
            cls_id = int(final_cls[j])
            conf = float(final_scores[j])
            
            is_edge = is_touching_edge(mask, original_img.shape, margin=BORDER_MARGIN)
            
            # Efficient overlap check: only check against other masks
            # We use a simple intersection check if masks are close (based on boxes)
            is_overlap = False
            for k in range(len(final_masks)):
                if j == k: continue
                # Bbox overlap check first
                b1, b2 = box, final_boxes[k]
                if not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3]):
                    # Potential mask overlap
                    # For simplicity and speed in this context, we'll use a slightly cheaper check
                    # but check_overlap is more accurate. Let's use it for now.
                    if check_overlap(mask, [final_masks[k]], original_img.shape):
                        is_overlap = True
                        break
            
            is_excluded = is_edge or is_overlap
            reason = []
            if is_edge: reason.append("edge")
            if is_overlap: reason.append("overlap")
            
            area, diameter = calculate_measurements(mask)
            
            color = EXCLUSION_COLOR if is_excluded else ((0, 255, 0) if cls_id == 0 else (0, 0, 255))
            
            if not is_excluded:
                if cls_id == 0: v_count += 1
                else: nv_count += 1
            
            # Visualization
            pts = np.array(mask, np.int32).reshape((-1, 1, 2))
            overlay = original_img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.4, original_img, 0.6, 0, original_img)
            cv2.polylines(original_img, [pts], True, color, 4)
            
            label = f"{'V' if cls_id == 0 else 'NV'} {conf:.2f}"
            if is_excluded: label += f" ({'+'.join(reason)})"
            
            font_scale = max(0.8, w_orig / 3000.0)
            thickness = max(2, int(font_scale * 2))
            cv2.putText(original_img, label, (int(box[0]), int(box[1])-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
            particle_data.append({
                'filename': img_file,
                'particle_id': j,
                'class': 'viable' if cls_id == 0 else 'non_viable',
                'conf': conf,
                'area_px': area,
                'diameter_px': diameter,
                'is_excluded': is_excluded,
                'exclusion_reason': "+".join(reason) if reason else ""
            })

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
            'total_counted': v_count + nv_count,
            'total_detected': len(final_masks)
        })
        
        # Write detailed measurements per image to a list we'll concat later
        if not hasattr(run_detection, 'all_particle_data'):
            run_detection.all_particle_data = []
        run_detection.all_particle_data.extend(particle_data)

    # Save Summaries and Detailed Measurements
    if data_rows:
        df = pd.DataFrame(data_rows)
        csv_path = os.path.join(LOCAL_DETECTED, 'pollen_counts.csv')
        df.to_csv(csv_path, index=False)
        
        details_df = pd.DataFrame(run_detection.all_particle_data)
        details_path = os.path.join(LOCAL_DETECTED, 'particle_measurements.csv')
        details_df.to_csv(details_path, index=False)
        
        print(f"✅ Detection done. Saved summary to {csv_path} and details to {details_path}")
        
        # Upload Results
        if s3_client:
             upload_file_robust(s3_client, csv_path, S3_BUCKET, 'Ostatni/Pollen_viability/detected_images/pollen_counts.csv')
             upload_file_robust(s3_client, details_path, S3_BUCKET, 'Ostatni/Pollen_viability/detected_images/particle_measurements.csv')

if __name__ == "__main__":
    run_detection()
