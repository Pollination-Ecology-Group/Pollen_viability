import os
import shutil
import boto3
from botocore.client import Config
from ultralytics import SAM

# Configuration
S3_ENDPOINT = os.environ.get('S3_ENDPOINT', 'https://s3.cl4.du.cesnet.cz')
S3_BUCKET = os.environ.get('S3_BUCKET')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

LOCAL_DIR = '/app/datasets/pollen_v1'
OUTPUT_DIR = '/app/datasets/pollen_v1_seg'
S3_PREFIX_IN = 'Ostatni/Pollen_viability/datasets/pollen_v1'
S3_PREFIX_OUT = 'Ostatni/Pollen_viability/datasets/pollen_v1_seg'

def setup_s3():
    return boto3.resource('s3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4', s3={'payload_signing_enabled': False})
    )

def download_dataset(s3):
    bucket = s3.Bucket(S3_BUCKET)
    print(f"⬇️ Downloading dataset from S3: {S3_PREFIX_IN}")
    count = 0
    for obj in bucket.objects.filter(Prefix=S3_PREFIX_IN):
        rel_path = os.path.relpath(obj.key, S3_PREFIX_IN)
        if rel_path == "." or rel_path.startswith("_"): continue
        dest_path = os.path.join(LOCAL_DIR, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        bucket.download_file(obj.key, dest_path)
        count += 1
    print(f"✅ Downloaded {count} files.")

def upload_dataset(s3):
    bucket = s3.Bucket(S3_BUCKET)
    print(f"⬆️ Uploading converted dataset to S3: {S3_PREFIX_OUT}")
    count = 0
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, OUTPUT_DIR)
            s3_path = f"{S3_PREFIX_OUT}/{rel_path}"
            bucket.upload_file(local_path, s3_path)
            count += 1
    print(f"✅ Uploaded {count} files.")

def convert_dataset_to_polygons():
    print("🧠 Starting SAM Conversion on GPU...")
    model = SAM('sam_b.pt')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for split in ['train', 'val']:
        split_img_dir = os.path.join(LOCAL_DIR, split, 'images')
        split_lbl_dir = os.path.join(LOCAL_DIR, split, 'labels')
        
        if not os.path.exists(split_img_dir): continue
            
        out_img_dir = os.path.join(OUTPUT_DIR, split, 'images')
        out_lbl_dir = os.path.join(OUTPUT_DIR, split, 'labels')
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)
        
        images = [f for f in os.listdir(split_img_dir) if f.endswith(('.jpg', '.png'))]
        print(f"Processing {len(images)} images in {split} split...")
        
        for i, img_name in enumerate(images):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(images)} images in {split}...", end='\r')
                
            img_path = os.path.join(split_img_dir, img_name)
            lbl_name = os.path.splitext(img_name)[0] + '.txt'
            lbl_path = os.path.join(split_lbl_dir, lbl_name)
            
            out_img_path = os.path.join(out_img_dir, img_name)
            out_lbl_path = os.path.join(out_lbl_dir, lbl_name)
            
            # Copy image
            shutil.copy(img_path, out_img_path)
            
            if not os.path.exists(lbl_path):
                open(out_lbl_path, 'w').close()
                continue
                
            bboxes, classes = [], []
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        # Since image size isn't immediately known without cv2, we can just load cv2
                        import cv2
                        img = cv2.imread(img_path)
                        if img is None: break
                        h, w = img.shape[:2]
                        
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        bboxes.append([x1, y1, x2, y2])
                        classes.append(cls)
            
            if not bboxes:
                shutil.copy(lbl_path, out_lbl_path)
                continue
                
            results = model.predict(img_path, bboxes=bboxes, verbose=False)
            
            with open(out_lbl_path, 'w') as out_f:
                result = results[0]
                if result.masks is not None:
                    for idx, segment in enumerate(result.masks.xyn):
                        if len(segment) > 0:
                            cls = classes[idx]
                            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in segment)
                            out_f.write(f"{cls} {coords}\n")
        print(f"\nFinished {split} split.")

    # Convert data.yaml
    yaml_src = os.path.join(LOCAL_DIR, 'data.yaml')
    if os.path.exists(yaml_src):
        with open(yaml_src, 'r') as f:
            content = f.read().replace('pollen_v1', 'pollen_v1_seg')
        with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
            f.write(content)

if __name__ == '__main__':
    s3 = setup_s3()
    download_dataset(s3)
    convert_dataset_to_polygons()
    upload_dataset(s3)
