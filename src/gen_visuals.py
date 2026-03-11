import os
import cv2
import random

# Use the absolute path provided by the user
DATASET_ROOT = '/home/meow/cesnet_cloud/bucket/Ostatni/Pollen_viability/datasets/pollen_v1'
TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
VAL_DIR = os.path.join(DATASET_ROOT, 'val')
VIS_DIR = 'visualizations' # Will be created in CWD

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
        
        if not os.path.exists(img_dir):
            print(f"Skipping {split_name}: {img_dir} not found")
            continue
        
        all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
        if not all_imgs: continue
        
        if num_samples:
             samples = random.sample(all_imgs, min(len(all_imgs), num_samples))
        else:
             samples = all_imgs
        
        print(f"Processing {len(samples)} images for {split_name}...")
        for i, img_file in enumerate(samples):
            if i % 100 == 0: print(f"  {i}/{len(samples)}", end='\r')
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
    print("\n✅ Visualization complete.")

if __name__ == "__main__":
    visualize_dataset()
