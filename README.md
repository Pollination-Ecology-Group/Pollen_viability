# Pollen Viability Detector (YOLOv8)
<img width="1332" height="1007" alt="image" src="https://github.com/user-attachments/assets/3410b492-e5e1-4c50-bd0d-1ddecb7e4805" />

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
```

# 🚀 How to Use

## 1. Routine Detection (Counting Pollen)
Follow these steps to process raw images and generate count data.

1.  **Upload**: Place your raw images (`.jpg`, `.png`, `.tif`) in the `detect_images/` folder on Google Drive.
2.  **Run**:  Open Pollen_viability_Routine_detection.ipynb in Google Colab  [![Open In Colab](https://colab.research.google.com/drive/1k8-OALzmpi5dXWK-UFuo4KsKqwbP8H6T).
3.  **Execute**: Run the section labeled **"7. Detect, compute..."**.

### Results
* **Visuals**: Check the `detected/` folder to view images with generated bounding boxes.
* **Data**: Download `pollen_counts_universal.csv` for the final count summary.

---

## 2. Updating the Model (Adding New Data)
Follow these steps to improve the model using new annotations.

1.  **Export**: Get your new data from Roboflow as a **YOLOv8 Zip** file.
2.  **Upload**: Place the zip file in `staged_area/` (or `staged_area/labels/`).
3.  **Run**:  Open pollen_viability_training.ipynb in Google Colab anmd run it [![Open In Colab](https://colab.research.google.com/drive/1fo5vSY2gq35_XyJZwm0M4DC9EyFXCXvo#scrollTo=DTjLLuXsLTZp&uniqifier=5).
    > *Note: The script automatically detects the zip, checks for duplicates, and splits data (85% Train / 15% Val).*
4.  **Train**: Run the **"3. Model Training"** section to retrain and save a new `.pt` model file.

---

# 🧬 Technical Details

### Class Logic
| Class | Type | Characteristics | Threshold |
| :--- | :--- | :--- | :--- |
| **0** | **Viable** | Stained dark | Conf > **0.40** |
| **1** | **Non-Viable** | Pale / Transparent | Conf > **0.25*** |

> *\*Threshold lowered for Class 1 to capture faint grains.*

### Augmentations
The model training is optimized for biological imagery using the following augmentations:
* **Heavy Rotation**
* **Vertical Flips**
* **Low Saturation Noise**
