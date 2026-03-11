import os
import cv2
import shutil
from ultralytics import SAM

def convert_dataset_to_polygons(data_dir, sam_model_name='sam_b.pt'):
    """
    Converts a YOLO bounding box dataset into a YOLO segmentation (polygon) dataset
    using the Segment Anything Model (SAM) and ground-truth bounding boxes as prompts.
    """
    print(f"Loading SAM model: {sam_model_name}")
    # SAM models: sam_b.pt (base), sam_l.pt (large), sam_h.pt (huge)
    # Using base model by default for speed.
    model = SAM(sam_model_name)
    
    # Create new directory for the polygon dataset
    output_dir = data_dir + '_seg'
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'val']:
        split_img_dir = os.path.join(data_dir, split, 'images')
        split_lbl_dir = os.path.join(data_dir, split, 'labels')
        
        if not os.path.exists(split_img_dir):
            continue
            
        out_img_dir = os.path.join(output_dir, split, 'images')
        out_lbl_dir = os.path.join(output_dir, split, 'labels')
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
            
            # Copy images to the new directory because cloud buckets might not support symlinks
            if not os.path.exists(out_img_path):
                shutil.copy(img_path, out_img_path)
            
            if not os.path.exists(lbl_path):
                # If no labels, create empty label file
                open(out_lbl_path, 'w').close()
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            bboxes = []
            classes = []
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # We only process bbox lines (5 values: class cx cy w h)
                    if len(parts) == 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        
                        # Convert YOLO normalized (cx, cy, w, h) to absolute (x1, y1, x2, y2)
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        
                        bboxes.append([x1, y1, x2, y2])
                        classes.append(cls)
            
            if not bboxes:
                # No valid bounding boxes to use as prompts, just copy original label file
                shutil.copy(lbl_path, out_lbl_path)
                continue
                
            # Run SAM prediction using the extracted bounding boxes as prompts
            results = model.predict(img_path, bboxes=bboxes, verbose=False)
            
            # Write segmentations to the new label file
            with open(out_lbl_path, 'w') as out_f:
                result = results[0]
                if result.masks is not None:
                    # masks.xyn returns normalized coordinates for polygons (required for YOLO format)
                    for idx, segment in enumerate(result.masks.xyn):
                        if len(segment) > 0:
                            cls = classes[idx]
                            # Format line: class x1 y1 x2 y2 ... xn yn
                            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in segment)
                            out_f.write(f"{cls} {coords}\n")
        print(f"\nFinished {split} split. {' '*20}")

    # Copy data.yaml and update the path internally
    yaml_src = os.path.join(data_dir, 'data.yaml')
    if os.path.exists(yaml_src):
        yaml_dst = os.path.join(output_dir, 'data.yaml')
        with open(yaml_src, 'r') as f:
            yaml_content = f.read()
        # Update directory references if present
        yaml_content = yaml_content.replace('pollen_v1', 'pollen_v1_seg')
        with open(yaml_dst, 'w') as f:
            f.write(yaml_content)
            
    print(f"✅ Conversion complete. New dataset is created at:\n   {output_dir}")

if __name__ == '__main__':
    # Using the dataset root found in your original gen_visuals.py
    base_dataset_path = '/home/meow/cesnet_cloud/bucket/Ostatni/Pollen_viability/datasets/pollen_v1'
    convert_dataset_to_polygons(base_dataset_path)
