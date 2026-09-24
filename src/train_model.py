
import os
# Force YOLO to use a writable config directory
os.environ['YOLO_CONFIG_DIR'] = '/app/yolo_config'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import sys
import boto3
import shutil
import glob
import random
import cv2
import numpy as np
import argparse
from botocore.client import Config
from ultralytics import YOLO, settings
from datetime import datetime
import smtplib
from email.message import EmailMessage
import urllib.request
import ssl

# Update global settings to prevent "Permission denied" in /ultralytics/runs
# This fixes the AMP check crash
settings.update({
    'runs_dir': '/app/runs',
    'datasets_dir': '/app/datasets',
    'weights_dir': '/app/weights'
})

# --- CONFIGURATION FROM ENV ---
S3_ENDPOINT = os.environ.get('S3_ENDPOINT', 'https://s3.cl4.du.cesnet.cz')
S3_BUCKET = os.environ.get('S3_BUCKET')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')
RECEIVER_EMAIL = "jakubstenc@gmail.com"

# Paths — dataset is built from CZI-derived sources only (no legacy Roboflow S4x data)
LOCAL_ROOT    = 'Pollen_viability'
DATASET_ROOT  = os.path.join(LOCAL_ROOT, 'datasets/pollen_czi')
STAGING_AREA  = os.path.join(LOCAL_ROOT, 'staged_area')
SMUDGES_RAW   = os.path.join(LOCAL_ROOT, 'smudges_raw')
HARD_NEGATIVES = os.path.join(LOCAL_ROOT, 'hard_negatives')
HARD_POSITIVES = os.path.join(LOCAL_ROOT, 'hard_positives')
TRAIN_DIR     = os.path.join(DATASET_ROOT, 'train')
VAL_DIR       = os.path.join(DATASET_ROOT, 'val')
VIS_DIR       = 'visualizations'

def setup_s3():
    if not all([S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        raise ValueError("❌ S3 Environment variables missing! Cannot sync data.")
    return boto3.resource('s3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4', s3={'payload_signing_enabled': False})
    )

def download_s3_prefix(s3, prefix, local_dir):
    bucket = s3.Bucket(S3_BUCKET)
    print(f"⬇️ Downloading {prefix} → {local_dir}")
    count = 0
    for obj in bucket.objects.filter(Prefix=prefix):
        # Skip S3 directory markers (keys ending with '/')
        if obj.key.endswith('/'):
            continue
        rel_path = os.path.relpath(obj.key, prefix)
        if rel_path == "." or rel_path.startswith("_"):
            continue
        dest_path = os.path.join(local_dir, rel_path)
        if not os.path.exists(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            bucket.download_file(obj.key, dest_path)
            count += 1
            if count % 100 == 0:
                print(f"   Downloaded {count}...", end='\r')
    print(f"✅ Downloaded {count} new files.")

def scaffold_dataset():
    """Create the empty train/val directory structure from scratch."""
    for split_dir in [TRAIN_DIR, VAL_DIR]:
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)
    print(f"📁 Dataset scaffold created at {DATASET_ROOT}")


def merge_hard_positives(exclude_bases: set = None):
    """Copy hard_positives image+label pairs into train/val.

    exclude_bases: set of tile basenames (no extension, no RF hash) already
    added via staged ZIPs.  Tiles in this set are skipped to avoid duplicates.
    """
    if not os.path.exists(HARD_POSITIVES):
        print("⚠️  hard_positives/ not found — no primary annotated data!")
        return
    exclude_bases = exclude_bases or set()
    pos_imgs = [f for f in os.listdir(HARD_POSITIVES)
                if f.lower().endswith(('.jpg', '.png'))]
    merged = skipped = 0
    for fname in pos_imgs:
        base = os.path.splitext(fname)[0]
        if base in exclude_bases:
            skipped += 1
            continue  # already in dataset from staged ZIP (better annotation)
        lbl_path = os.path.join(HARD_POSITIVES, base + '.txt')
        if not os.path.exists(lbl_path):
            continue  # skip unannotated images
        is_val     = random.random() < 0.2
        target_dir = VAL_DIR if is_val else TRAIN_DIR
        out_name   = f"hard_pos_{fname}"
        shutil.copy2(os.path.join(HARD_POSITIVES, fname),
                     os.path.join(target_dir, 'images', out_name))
        shutil.copy2(lbl_path,
                     os.path.join(target_dir, 'labels', os.path.splitext(out_name)[0] + '.txt'))
        merged += 1
    if skipped:
        print(f"  ⚠️  Skipped {skipped} hard_positives already covered by staged ZIP.")
    print(f"✅ Merged {merged} hard_positives pairs (primary annotated data).")


def print_dataset_summary():
    """Print a breakdown of how many images each source contributed."""
    from collections import Counter
    print("\n" + "═" * 55)
    print("📊 Dataset Composition Summary")
    print("═" * 55)
    for split_name, split_dir in [('train', TRAIN_DIR), ('val', VAL_DIR)]:
        img_dir = os.path.join(split_dir, 'images')
        lbl_dir = os.path.join(split_dir, 'labels')
        if not os.path.exists(img_dir):
            continue
        source_counts = Counter()
        annot_counts  = Counter()  # class counts across all labels
        total_grains  = 0
        for fn in os.listdir(img_dir):
            if fn.startswith('hard_pos_'):    source_counts['hard_positives'] += 1
            elif fn.startswith('hard_neg_') or fn.startswith('curated_neg_'):
                                              source_counts['hard_negatives'] += 1
            elif fn.startswith('syn_neg_'):   source_counts['smudges_synthetic'] += 1
            else:                             source_counts['staging_area'] += 1
            lbl = os.path.join(lbl_dir, os.path.splitext(fn)[0] + '.txt')
            if os.path.exists(lbl):
                for line in open(lbl).read().strip().splitlines():
                    parts = line.strip().split()
                    if parts:
                        annot_counts[int(parts[0])] += 1
                        total_grains += 1
        total_imgs = sum(source_counts.values())
        print(f"  [{split_name}]  {total_imgs} images  |  {total_grains} grain annotations")
        for src, n in sorted(source_counts.items(), key=lambda x: -x[1]):
            print(f"    {src:<22}: {n:4d} images")
        cls_names = {0: 'viable', 1: 'non_viable', 2: 'intermediate'}
        for cls_id, cnt in sorted(annot_counts.items()):
            print(f"    class {cls_id} ({cls_names.get(cls_id,'?'):<13}): {cnt:4d} annotations")
    print("═" * 55 + "\n")



# Canonical class ordering that the model and all inference scripts expect.
CANONICAL_CLASSES = ['viable', 'non_viable', 'intermediate']

def _build_class_remap(zip_data_yaml_path: str) -> dict:
    """
    Read a Roboflow-exported data.yaml and return a {src_id → dst_id} mapping
    so that class IDs are remapped to CANONICAL_CLASSES ordering.

    Roboflow exports alphabetically by default:
        0=intermediate  1=non_viable  2=viable
    We want:
        0=viable        1=non_viable  2=intermediate

    Also handles 'non-viable' (hyphen) as an alias for 'non_viable'.
    """
    import yaml

    remap = {}
    try:
        with open(zip_data_yaml_path) as f:
            meta = yaml.safe_load(f)
        src_names = meta.get('names', [])
        # Normalise: replace hyphens with underscores, lowercase
        src_names = [n.lower().replace('-', '_') for n in src_names]
        canonical_norm = [c.lower().replace('-', '_') for c in CANONICAL_CLASSES]
        for src_id, name in enumerate(src_names):
            if name in canonical_norm:
                dst_id = canonical_norm.index(name)
                remap[src_id] = dst_id
            else:
                print(f"   ⚠️  Unknown class '{name}' in export — keeping as id {src_id}")
                remap[src_id] = src_id
        print(f"   🗺️  Class remapping from export: {dict(zip(src_names, [remap[i] for i in range(len(src_names))]))}")
    except Exception as e:
        print(f"   ⚠️  Could not read data.yaml ({e}) — using identity mapping")
    return remap


def _remap_label_file(src_path: str, dst_path: str, remap: dict):
    """Read a YOLO label file, remap class IDs, write to dst_path."""
    lines_out = []
    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            src_cls = int(parts[0])
            dst_cls = remap.get(src_cls, src_cls)
            lines_out.append(f"{dst_cls} " + " ".join(parts[1:]))
    with open(dst_path, 'w') as f:
        f.write("\n".join(lines_out))


def _strip_rf_hash(filename: str) -> str:
    """Remove Roboflow's .rf.HASH suffix to recover the original tile basename.
    e.g. 'tile_0_16128_jpg.rf.abc123.jpg' -> 'tile_0_16128'
    """
    import re
    base = os.path.splitext(filename)[0]           # strip final .jpg/.png
    base = re.sub(r'\.rf\.[a-f0-9]+$', '', base)  # strip .rf.HASH
    base = re.sub(r'_jpg$|_png$', '', base)        # strip _jpg/_png artifact
    return base


def merge_staged_data() -> set:
    print("🔄 Checking for new data in staging area...")
    if not os.path.exists(STAGING_AREA):
        return set()

    zips = glob.glob(os.path.join(STAGING_AREA, "*.zip"))
    if not zips:
        print("ℹ️ No new zips found in staging area.")
        return set()

    staged_bases: set = set()  # stripped original basenames merged from all ZIPs
    for zip_path in sorted(zips):
        print(f"\n1️⃣ Processing: {os.path.basename(zip_path)}")

        temp_dir = os.path.join(LOCAL_ROOT, 'temp_merge')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        shutil.unpack_archive(zip_path, temp_dir)

        # ── Build class remapping from the ZIP's data.yaml ────────────────────
        remap = {}
        for root, _, files in os.walk(temp_dir):
            if 'data.yaml' in files:
                remap = _build_class_remap(os.path.join(root, 'data.yaml'))
                break
        if not remap:
            print("   ⚠️  No data.yaml found in ZIP — using identity class mapping")

        # ── Collect image+label pairs ──────────────────────────────────────────
        found_pairs = []
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png')):
                    base = os.path.splitext(f)[0]
                    lbl = os.path.join(root, base + '.txt')
                    if os.path.exists(lbl):
                        found_pairs.append((os.path.join(root, f), lbl))

        print(f"   Found {len(found_pairs)} image+label pairs.")

        # Verify class counts after remapping
        cls_counts: dict = {}
        for _, lbl in found_pairs:
            for line in open(lbl).read().strip().splitlines():
                parts = line.strip().split()
                if parts:
                    dst = remap.get(int(parts[0]), int(parts[0]))
                    name = CANONICAL_CLASSES[dst] if dst < len(CANONICAL_CLASSES) else f'cls{dst}'
                    cls_counts[name] = cls_counts.get(name, 0) + 1
        print(f"   Annotation counts after remapping: {cls_counts}")

        # ── 80/20 train/val split ──────────────────────────────────────────────
        random.shuffle(found_pairs)
        split_idx   = int(len(found_pairs) * 0.2)
        val_batch   = found_pairs[:split_idx]
        train_batch = found_pairs[split_idx:]

        for batch, dest_dir in [(train_batch, TRAIN_DIR), (val_batch, VAL_DIR)]:
            img_dest = os.path.join(dest_dir, 'images')
            lbl_dest = os.path.join(dest_dir, 'labels')
            os.makedirs(img_dest, exist_ok=True)
            os.makedirs(lbl_dest, exist_ok=True)
            for img_path, lbl_path in batch:
                fname    = os.path.basename(img_path)
                out_base = os.path.splitext(fname)[0]
                orig_base = _strip_rf_hash(fname)
                shutil.copy2(img_path, os.path.join(img_dest, fname))
                _remap_label_file(lbl_path,
                                  os.path.join(lbl_dest, out_base + '.txt'),
                                  remap)
                staged_bases.add(orig_base)

        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"✅ Merged {len(found_pairs)} pairs "
              f"({len(train_batch)} train / {len(val_batch)} val).")

    print(f"   Total unique tile basenames from staged ZIPs: {len(staged_bases)}")
    return staged_bases


def process_all_negatives():
    print("🧪 Processing all Hard Negatives and Smudges...")
    total_count = 0
    CANVAS_SIZE = 640
    BG_COLOR = (200, 200, 200)

    # 1. Process Smudges (Synthetic)
    if os.path.exists(SMUDGES_RAW):
        raw_files = [f for f in os.listdir(SMUDGES_RAW) if f.lower().endswith(('.jpg', '.png'))]
        for fname in raw_files:
            img_path = os.path.join(SMUDGES_RAW, fname)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Resize logic onto canvas
            h, w = img.shape[:2]
            scale = min(CANVAS_SIZE/h, CANVAS_SIZE/w) * 0.8
            new_w, new_h = int(w*scale), int(h*scale)
            resized = cv2.resize(img, (new_w, new_h))
            
            canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), BG_COLOR, dtype=np.uint8)
            y_off = (CANVAS_SIZE - new_h) // 2
            x_off = (CANVAS_SIZE - new_w) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            
            is_val = random.random() < 0.2
            target_dir = VAL_DIR if is_val else TRAIN_DIR
            out_name = f"syn_neg_{fname}"
            
            cv2.imwrite(os.path.join(target_dir, 'images', out_name), canvas)
            with open(os.path.join(target_dir, 'labels', os.path.splitext(out_name)[0]+'.txt'), 'w') as f: pass
            total_count += 1
            
    # 2. Process Curated Hard Negatives (Direct Copy, empty labels)
    if os.path.exists(HARD_NEGATIVES):
        hard_files = [f for f in os.listdir(HARD_NEGATIVES) if f.lower().endswith(('.jpg', '.png'))]
        for fname in hard_files:
            img_path = os.path.join(HARD_NEGATIVES, fname)
            
            is_val = random.random() < 0.2
            target_dir = VAL_DIR if is_val else TRAIN_DIR
            out_name = f"curated_neg_{fname}"
            
            shutil.copy2(img_path, os.path.join(target_dir, 'images', out_name))
            with open(os.path.join(target_dir, 'labels', os.path.splitext(out_name)[0]+'.txt'), 'w') as f: pass
            total_count += 1

    # 3. Process Curated Hard Positives (image + label pairs, already annotated)
    if os.path.exists(HARD_POSITIVES):
        pos_imgs = [f for f in os.listdir(HARD_POSITIVES) if f.lower().endswith(('.jpg', '.png'))]
        merged = 0
        for fname in pos_imgs:
            img_path = os.path.join(HARD_POSITIVES, fname)
            lbl_path = os.path.join(HARD_POSITIVES, os.path.splitext(fname)[0] + '.txt')
            if not os.path.exists(lbl_path):
                continue  # skip images without a companion label
            
            is_val = random.random() < 0.2
            target_dir = VAL_DIR if is_val else TRAIN_DIR
            out_name = f"hard_pos_{fname}"
            
            shutil.copy2(img_path, os.path.join(target_dir, 'images', out_name))
            shutil.copy2(lbl_path, os.path.join(target_dir, 'labels', os.path.splitext(out_name)[0] + '.txt'))
            merged += 1
        print(f"   ✅ Merged {merged} hard positive image+label pairs.")
        total_count += merged
    else:
        print("   ℹ️  No hard_positives directory found — skipping.")

    print(f"✅ Integrated {total_count} total negative samples into dataset.")

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
                            # Segmentation model annotations: cls x1 y1 ... xn yn
                            cls = int(parts[0])
                            
                            coords = list(map(float, parts[1:]))
                            points = np.array(coords).reshape(-1, 2)
                            points[:, 0] *= w
                            points[:, 1] *= h
                            pts = points.astype(np.int32).reshape((-1, 1, 2))
                            
                            color = COLOR_MAP.get(cls, (255, 255, 255))
                            cv2.polylines(img, [pts], True, color, 2)
                            
                            tx, ty = pts[0][0]
                            cv2.putText(img, str(cls), (tx, ty-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
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

    print("✅ Upload complete.")

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

def upload_results(s3, local_dir, train_name):
    s3_client = s3.meta.client
    
    # 1. Weights & Metrics
    s3_prefix_run = f"Ostatni/Pollen_viability/trained_models/{train_name}"
    
    print(f"⬆️ Uploading training results from {local_dir} to {s3_prefix_run}...")
    if os.path.exists(local_dir):
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_prefix_run}/{rel_path}"
                upload_file_robust(s3_client, local_path, S3_BUCKET, s3_key)

    # 2. Visualizations
    s3_prefix_vis = f"Ostatni/Pollen_viability/trained_models/{train_name}/visualizations"
    print(f"⬆️ Uploading visualizations to {s3_prefix_vis}...")
    if os.path.exists(VIS_DIR):
        for root, _, files in os.walk(VIS_DIR):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, VIS_DIR)
                s3_key = f"{s3_prefix_vis}/{rel_path}"
                upload_file_robust(s3_client, local_path, S3_BUCKET, s3_key)

def send_notification(subject, body):
    if not GMAIL_APP_PASSWORD:
        print("⚠️ Skipping email notification: GMAIL_APP_PASSWORD not set.")
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = RECEIVER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    try:
        print(f"📧 Sending notification to {RECEIVER_EMAIL}...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(RECEIVER_EMAIL, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        print("✅ Notification sent!")
    except Exception as e:
        print(f"❌ Failed to send notification: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true', help="Skip actual training")
    args = parser.parse_args()

    s3 = setup_s3()
    
    # 1. Scaffold empty dataset structure (CZI-derived sources only)
    scaffold_dataset()

    # 2. Download all CZI-derived annotation sources from S3
    if s3:
        download_s3_prefix(s3, 'Ostatni/Pollen_viability/staged_area',
                           STAGING_AREA)
        download_s3_prefix(s3, 'Ostatni/Pollen_viability/smudges_raw',
                           SMUDGES_RAW)
        download_s3_prefix(s3, 'Ostatni/Pollen_viability/active_learning/hard_negatives',
                           HARD_NEGATIVES)
        download_s3_prefix(s3, 'Ostatni/Pollen_viability/active_learning/hard_positives',
                           HARD_POSITIVES)

    # 3. Assemble dataset (order matters: staged ZIPs first — better annotations)
    #    a) Roboflow ZIPs from staged_area/ — all 3 classes, class-remapped
    staged_bases = merge_staged_data()
    #    b) hard_positives — viable-only tiles NOT already covered by staged ZIP
    merge_hard_positives(exclude_bases=staged_bases)
    #    c) Background tiles — hard_negatives + synthetic smudges (empty labels)
    process_all_negatives()

    # 4. Print composition before training so we know what went in
    print_dataset_summary()

    visualize_dataset(num_samples=None)  # Ground-truth visualisation for all images

    # 3. Train
    if not args.dry_run:
        print("🚀 Starting Training...")
        # Check for GPU
        device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"   Device: {device}")
        
        model = YOLO('yolo11m-seg.pt')
        run_name = f"pollen_train_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # ALWAYS overwrite data.yaml to ensure paths are correct for this container
        yaml_path = os.path.join(DATASET_ROOT, 'data.yaml')
        print(f"🔧 Fixing data.yaml at {yaml_path}...")
        with open(yaml_path, 'w') as f:
            # We use absolute paths to be safe
            abs_root = os.path.abspath(DATASET_ROOT)
            f.write(f"path: {abs_root}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            # Assuming standard classes for this project
            f.write("names:\n  0: viable\n  1: non_viable\n  2: intermediate\n")

        results = model.train(
            data=yaml_path,
            epochs=args.epochs,
            patience=0, # Disable early stopping to ensure fixed epoch count
            batch=args.batch,
            imgsz=640,
            device=device,
            task='segment',
            name=run_name,
            project=os.path.join(os.getcwd(), 'runs/detect'),
            agnostic_nms=True,
            val=True,
            degrees=180,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            close_mosaic=20
        )
        
        # Capture the actual directory where YOLO saved the results
        actual_save_dir = results.save_dir
        print(f"📂 YOLO saved results to: {actual_save_dir}")
        
        # 4. Independent Evaluation
        print("📊 Running Evaluation...")
        metrics = model.val(split='val') # or 'test' if available
        print(f"   mAP50-95: {metrics.box.map}")
        
        # 4b. Visualize Predictions
        visualize_predictions(model, split='val', num_samples=None)        
        # 5. Backup
        if s3:
            upload_results(s3, actual_save_dir, run_name)
            
            # 6. Notify
            send_notification(
                subject=f"🚀 Training Complete: {run_name}",
                body=f"Hello Jakub,\n\nYour training run '{run_name}' has finished successfully.\nResults have been uploaded to S3."
            )
    else:
        print("⚠️ Dry run mode: Skipping actual training.")

if __name__ == "__main__":
    import torch # import here to avoid delay
    main()
