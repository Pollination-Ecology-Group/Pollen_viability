import os
import cv2
import random
import numpy as np

DATASET_ROOT = '/home/meow/cesnet_cloud/bucket/Ostatni/Pollen_viability/datasets/pollen_v1_seg'
TRAIN_DIR = os.path.join(DATASET_ROOT, 'train')
VIS_DIR = 'visualizations_seg'

def visualize_dataset(num_samples=9):
    print("🎨 Generating Dataset Visualizations for Polygons...")
    os.makedirs(VIS_DIR, exist_ok=True)
    COLOR_MAP = {0: (0, 255, 0), 1: (0, 0, 255)}
    
    img_dir = os.path.join(TRAIN_DIR, 'images')
    lbl_dir = os.path.join(TRAIN_DIR, 'labels')
    
    all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
    samples = random.sample(all_imgs, min(len(all_imgs), num_samples))
    
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
                    if len(parts) > 5:
                        cls = int(parts[0])
                        color = COLOR_MAP.get(cls, (255, 255, 255))
                        coords = list(map(float, parts[1:]))
                        points = np.array(coords).reshape(-1, 2)
                        points[:, 0] *= w
                        points[:, 1] *= h
                        pts = points.astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [pts], True, color, 2)
                        
                        tx, ty = pts[0][0]
                        cv2.putText(img, str(cls), (tx, ty-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(os.path.join(VIS_DIR, img_file), img)
    print("✅ Visualization complete.")

if __name__ == "__main__":
    visualize_dataset()
