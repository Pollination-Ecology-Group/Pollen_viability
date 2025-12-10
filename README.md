# Pollen Viability Detector (YOLOv8)

An automated computer vision pipeline for detecting and counting viable vs. non-viable pollen grains in microscope imagery. This tool handles challenges specific to pollen analysis, including variable lighting, staining differences, and high-resolution full-slide scans.

## 📊 Performance
* **Model:** YOLOv8 (Large/Medium)
* **Accuracy (mAP@50):** ~99% for Viable, ~96% for Non-Viable
* **Resolution Strategy:** Dynamic switching between Standard (640px) for crops and High-Res (1600px) for full slides.

## 📂 Project Structure
```text
.
├── datasets/
│   └── pollen_v1/              # Main dataset folder (YOLO format)
│       ├── train/              # Training images and labels
│       ├── val/                # Validation images and labels
│       └── data.yaml           # Configuration file for YOLO
├── trained_models/             # Saved model weights (e.g., pollen_yolo_aug_2025-12-09.pt)
├── detect_images/              # INPUT: Put new microscope images here
├── detected/                   # OUTPUT: Annotated images appear here
├── staged_area/                # STAGING: Upload new Roboflow zips here to update the dataset
├── backups/                    # SAFETY: Auto-generated backups of the dataset
├── pollen_counts_universal.csv # OUTPUT: Spreadsheet with final counts
├── pollen_viability.ipynb      # MAIN SCRIPT: All-in-one pipeline (Maintenance, Training, Detection)
└── README.md

<img width="1332" height="1007" alt="image" src="https://github.com/user-attachments/assets/3410b492-e5e1-4c50-bd0d-1ddecb7e4805" />
