import os
import cv2
import boto3
import numpy as np
import czifile
from botocore.client import Config
import argparse
from tqdm import tqdm
import urllib.request
import ssl
import random

S3_ENDPOINT = os.environ.get('S3_ENDPOINT', 'https://s3.cl4.du.cesnet.cz')
S3_BUCKET = os.environ.get('S3_BUCKET')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

S3_PREFIX = 'Ostatni/Pollen_viability/Source'
S3_OUTPUT_PREFIX = 'Ostatni/Pollen_viability/tiles_640'
LOCAL_CZI_DIR = 'data/czi_files'
LOCAL_TILES_DIR = 'data/tiles_640'

def setup_s3():
    if not all([S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        print("⚠️ S3 Credentials missing. Make sure to source .env")
        return None, None
    resource = boto3.resource('s3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4')
    )
    client = boto3.client('s3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4')
    )
    return resource, client

def upload_file_robust(s3_client, local_path, bucket, key):
    try:
        url = s3_client.generate_presigned_url('put_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=3600)
        size = os.path.getsize(local_path)
        with open(local_path, 'rb') as data:
            req = urllib.request.Request(url, data=data, method='PUT')
            req.add_header('Content-Length', str(size))
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, context=ctx) as f:
                if f.status != 200:
                    print(f"   -> ⚠️ Upload rejected {os.path.basename(local_path)}: Status {f.status}")
    except Exception as e:
        print(f"❌ Failed to upload {local_path}: {e}")

def extract_image_from_czi(czi_path):
    with czifile.CziFile(czi_path) as czi:
        img_data = czi.asarray()
        
    img_data = np.squeeze(img_data)
    if img_data.ndim == 3 and img_data.shape[0] in [3, 4]: 
        img_data = np.transpose(img_data, (1, 2, 0))
        
    if img_data.dtype == np.uint16:
        img_min, img_max = img_data.min(), img_data.max()
        img_data = ((img_data - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
        
    if img_data.ndim == 2:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
    elif img_data.ndim == 3 and img_data.shape[2] == 3:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

    return img_data

def is_nonviable_candidate_tile(tile_bgr, min_candidates=1):
    """
    Screens a 640x640 tile for non-viable pollen candidates under Alexander Stain.
    Criteria:
      - Viable pollen is heavily stained with Magenta/Purple (High Saturation, Hue 140-175 / 0-10).
      - Non-viable / aborted pollen lacks deep magenta stain (pale green, grey, transparent, light orange, or empty shells).
      - Filters out background slide whitespace (white/light grey), huge air bubbles (>25,000px), and tiny cytoplasm specks (<250px).
    """
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    
    # 1. Mask slide background (very high lightness, low saturation white background)
    is_bg = (hsv[:, :, 1] < 35) & (hsv[:, :, 2] > 195)
    
    # 2. Mask deep magenta/purple (viable pollen cytoplasm)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    is_magenta = ((h > 135) | (h < 12)) & (s > 55) & (v < 225)
    
    # 3. Detect pollen candidate particles (non-background regions)
    particle_mask = (~is_bg).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    particle_mask = cv2.morphologyEx(particle_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    nonviable_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter size: pollen grain area on 640x640 tile is ~350 to 20,000 px.
        # Ignore tiny cytoplasm specks (< 250 px) and huge air bubbles (> 25000 px)
        if 250 < area < 25000:
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circularity = 4 * np.pi * area / (peri * peri)
                if circularity > 0.22:
                    c_mask = np.zeros(tile_bgr.shape[:2], dtype=np.uint8)
                    cv2.drawContours(c_mask, [cnt], -1, 255, -1)
                    
                    particle_pixels = np.sum(c_mask > 0)
                    if particle_pixels > 0:
                        magenta_pixels = np.sum((c_mask > 0) & is_magenta)
                        magenta_ratio = magenta_pixels / particle_pixels
                        
                        # Non-viable condition: particle lacks deep magenta stain (< 20% magenta)
                        if magenta_ratio < 0.20:
                            nonviable_count += 1
                            
    return nonviable_count >= min_candidates

def tile_image(img_data, base_name, s3_client, tile_size=640, overlap=0.1, nonviable_only=False):
    h, w = img_data.shape[:2]
    os.makedirs(LOCAL_TILES_DIR, exist_ok=True)
    
    stride = int(tile_size * (1 - overlap))
    y_starts = list(range(0, h - tile_size, stride))
    if h > tile_size and y_starts[-1] + tile_size < h: y_starts.append(h - tile_size)
    x_starts = list(range(0, w - tile_size, stride))
    if w > tile_size and x_starts[-1] + tile_size < w: x_starts.append(w - tile_size)
        
    if h <= tile_size and w <= tile_size:
        y_starts, x_starts = [0], [0]
        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        padded[:h, :w] = img_data
        img_data = padded

    total_generated = 0
    kept_count = 0
    for y in y_starts:
        for x in x_starts:
            total_generated += 1
            tile = img_data[y:y+tile_size, x:x+tile_size]
            
            if nonviable_only and not is_nonviable_candidate_tile(tile):
                continue
                
            filename = f"{base_name}_tile_{y}_{x}.jpg"
            out_path = os.path.join(LOCAL_TILES_DIR, filename)
            cv2.imwrite(out_path, tile)
            
            # Upload to S3 organized by sample
            if s3_client:
                s3_key = f"{S3_OUTPUT_PREFIX}/{base_name}/{filename}"
                upload_file_robust(s3_client, out_path, S3_BUCKET, s3_key)
            
            # Cleanup local tile to save space
            os.remove(out_path)
            kept_count += 1
            
    print(f"Generated {total_generated} tiles -> Kept {kept_count} non-viable candidate tiles for {base_name}")

import json

def load_sample_viability_index():
    index_path = os.path.join(os.path.dirname(__file__), 'sample_viability_index.json')
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Warning loading sample index: {e}")
    return {}

def get_czi_sample_id(key_or_filename):
    basename = os.path.basename(key_or_filename)
    return basename.split('_')[0]

def get_nonviable_rank(key_or_filename, index):
    sample_id = get_czi_sample_id(key_or_filename)
    if sample_id in index:
        data = index[sample_id]
        return (data.get("non_viable", 0), data.get("non_viable_rate", 0.0))
    return (0, 0.0)

def main(args):
    s3_resource, s3_client = setup_s3()
    os.makedirs(LOCAL_CZI_DIR, exist_ok=True)
    index = load_sample_viability_index()
    if index:
        print(f"📊 Loaded historical viability index for {len(index)} samples.")
    
    if args.download and s3_client:
        print(f"Fetching list of .czi files from S3 ({S3_PREFIX})...")
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
        
        keys = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.czi'):
                        keys.append(obj['Key'])
        
        if args.pattern:
            keys = [k for k in keys if args.pattern.lower() in k.lower()]
            print(f"🔍 Filtered down to {len(keys)} files matching pattern '{args.pattern}'")

        if args.min_nonviable_pct > 0.0:
            threshold = args.min_nonviable_pct / 100.0
            filtered = [k for k in keys if get_nonviable_rank(k, index)[1] >= threshold or get_nonviable_rank(k, index)[0] >= 10]
            if filtered:
                print(f"🎯 Filtered CZI files down to {len(filtered)} files with >={args.min_nonviable_pct}% non-viable estimate.")
                keys = filtered

        if args.prioritize_nonviable:
            print("🎯 Sorting CZI files by historical non-viable pollen yield...")
            keys.sort(key=lambda k: get_nonviable_rank(k, index), reverse=True)

        if args.limit and args.limit > 0:
            if args.prioritize_nonviable or args.pattern:
                keys = keys[:args.limit]
                print(f"Selected top {len(keys)} prioritized CZI files.")
            else:
                keys = random.sample(keys, min(args.limit, len(keys)))
                print(f"Randomly selected {len(keys)} files for processing.")

        for key in keys:
            filename = os.path.basename(key)
            local_path = os.path.join(LOCAL_CZI_DIR, filename)
            rank = get_nonviable_rank(filename, index)
            print(f"Downloading {filename} (Est. Non-Viable: {rank[0]} grains, {rank[1]*100:.1f}%)...")
            s3_client.download_file(S3_BUCKET, key, local_path)
            
            # Process immediately to save ephemeral storage
            try:
                print(f"Processing {filename}...")
                img_data = extract_image_from_czi(local_path)
                tile_image(img_data, os.path.splitext(filename)[0], s3_client, tile_size=640, nonviable_only=args.nonviable_only)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            finally:
                # Cleanup original CZI to conserve K8s ephemeral storage
                if os.path.exists(local_path):
                    os.remove(local_path)
    else:
        # Local processing only
        czi_files = [f for f in os.listdir(LOCAL_CZI_DIR) if f.endswith('.czi')]
        if args.pattern:
            czi_files = [f for f in czi_files if args.pattern.lower() in f.lower()]
        if args.prioritize_nonviable:
            czi_files.sort(key=lambda f: get_nonviable_rank(f, index), reverse=True)
        for czi_file in czi_files:
            czi_path = os.path.join(LOCAL_CZI_DIR, czi_file)
            base_name = os.path.splitext(czi_file)[0]
            try:
                img_data = extract_image_from_czi(czi_path)
                tile_image(img_data, base_name, s3_client, tile_size=640, nonviable_only=args.nonviable_only)
            except Exception as e:
                print(f"Error processing {czi_file}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and preprocess .czi files")
    parser.add_argument('--download', action='store_true', help="Download files from S3 first")
    parser.add_argument('--limit', type=int, default=0, help="Limit processing to N files")
    parser.add_argument('--prioritize-nonviable', action='store_true', help="Prioritize files from samples with high non-viable yields")
    parser.add_argument('--nonviable-only', action='store_true', help="Screen and keep only tiles containing candidate non-viable pollen")
    parser.add_argument('--min-nonviable-pct', type=float, default=0.0, help="Minimum non-viable percentage threshold (e.g. 5.0)")
    parser.add_argument('--pattern', type=str, default='', help="Filter CZI files by filename pattern or sample ID (e.g. 5-8-B_AA012_s2x)")
    args = parser.parse_args()
    main(args)

