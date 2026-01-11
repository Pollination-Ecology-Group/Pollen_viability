# 🌸 Pollen Viability Detector (YOLOv8)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)
![Status](https://img.shields.io/badge/Status-Active-success)

<img width="1332" height="1007" alt="image" src="https://github.com/user-attachments/assets/3410b492-e5e1-4c50-bd0d-1ddecb7e4805" />

An automated computer vision pipeline for assessing pollen viability using the **Alexander Stain** technique. This project leverages YOLOv8 to detect and classify pollen grains as either **Viable** (Magenta/Purple) or **Non-Viable** (Green/Shriveled) from high-resolution microscope images.

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Methodology](#-methodology)
- [Workflow Setup](#-workflow-setup)
- [Kubernetes Workflow](/vignettes/kubernetes_workflow.md)
- [Labeling Guide](#-labeling-guide)
- [Training](#-training-cesnet-cluster)
- [Routine Detection](#-routine-detection-coming-soon)

---

## 🔭 Project Overview
This tool automates the tedious process of counting pollen grains on microscope slides. It uses deep learning to replicate the judgment of a biologist following the Alexander Stain protocol.

* **Input:** High-resolution microscope images (TIF/JPG).
* **Output:** Annotated images with bounding boxes, CSV summaries of counts, and viability percentages.
* **Performance:** Optimized for large, crowded scans using tiling and global Non-Maximum Suppression (NMS).

---

## 🔬 Methodology
The project runs on the **CESNET MetaCentrum** high-performance computing cluster, utilizing NVIDIA A40/A100 GPUs for rapid training and inference.

### Key Strategies used in this model:
1.  **Mosaic Disabling:** We disable mosaic augmentation for the final 10 epochs of training. This stabilizes the training loss and prevents the model from hallucinating pollen in empty space (ghost detections).
2.  **High-Res Tiling:** For inference, large microscope scans (>2000px) are sliced into overlapping tiles (1600x1600px). This prevents the "Squish Effect," ensuring small grains retain their texture and are not lost during resizing.
3.  **Border Patrol:** To prevent double-counting grains on tile edges, we implement a custom logic that ignores edge detections, relying on the neighboring tile to capture the grain centrally.

---

## ⚙️ Workflow Setup

This project has migrated from Google Colab to the **CESNET Cluster** for better performance and data privacy.

### 1. Connecting to CESNET
Detailed instructions on how to set up your environment, connect via SSH, and manage files can be found in our connection guide:

👉 **[CESNET Connection Guide](/vignettes/CESNET_connection_guide.md)** 

### 2. Kubernetes Workflow
For instructions on running detection and training jobs:
👉 **[Kubernetes Workflow Guide](/vignettes/kubernetes_workflow.md)** 

### 2. Environment Installation
The environment uses a custom Jupyter kernel. Dependencies include `ultralytics`, `opencv-python-headless`, and `pandas`.
*(See the `setup_environment.py` script in the repo for automated dependency handling)*.

---

## 🏷️ Labeling Guide (Roboflow)

We use **Roboflow** for annotating datasets. If you are contributing to the dataset, please adhere to strict **Alexander Stain** criteria.

**Source:** *Pollen Viability Staining Guide (Alexander Stain)*

### 1. Class: `viable` (Target)
**Definition:** Pollen grains that contain healthy, full cytoplasm and are capable of fertilization.

* **Color:** Must be **Magenta-Red** or **Dark Purple**.
* **Content:** The grain appears "full" and dense, indicating the presence of cytoplasm.
* **Shape:** Typically regular, round, or plump.
* **Rule of Thumb:** If it is dark red/purple and looks "full," label it **Viable**.

> **Example:** ![Viable Example](path/to/viable_example.jpg) *(Add your example image here)*

### 2. Class: `non_viable` (Target)
**Definition:** Pollen grains that are aborted, empty, or dead. They lack the cytoplasm required for fertilization.

* **Color:** Primarily **Pale**.
* **Content:** Appears empty; only the outer cell wall is visible.
* **Shape:** Often shrivelled, collapsed, or significantly smaller than viable grains.
* **Intermediate Cases:**
    * **Pale Pink / Orange:** Grains that are very pale or only partially stained red are likely aborted. Label as **Non-Viable** unless they are fully dark.
    * **Empty Shells:** Transparent or faint outlines with no colour.

> **Example:** ![Non-Viable Example](path/to/non_viable_example.jpg) *(Add your example image here)*

### 3. Edge Cases & Rules
* **Clusters:** If grains are touching, draw a separate box for **each individual grain**. Do not draw one big box around a clump.
* **Cut-off Grains:** If >50% of the grain is visible at the image edge, label it. If <50% is visible, ignore it.
* **Debris:** Do not label dirt, bubbles, or stain blobs. Only label clear pollen grains.

---

## 🏋️ Training (CESNET Cluster)

Training is performed using the `Pollen_viability_training_cluster.ipynb` notebook.

**Steps Overview:**
1.  **Data Loading:** Automatically downloads the latest dataset version from Roboflow using the API.
2.  **Configuration:**
    * **Epochs:** 50
    * **Image Size:** 640px
    * **Batch Size:** 32 (Adjusted for GPU VRAM)
    * **Optimizer:** `auto` (YOLO default)
3.  **Refinement:** The training disables Mosaic augmentation (`close_mosaic=10`) for the final 10 epochs to refine detection of small objects.
4.  **Backup:** The best model weights (`best.pt`) are automatically zipped and backed up to S3 storage to prevent data loss.

---

## 🔍 Routine Detection (Coming Soon)

*This section will be updated with the finalized `run_detection.py` script for local "Drop & Run" usage.*

The routine detection pipeline features:
* **Automatic Tiling:** Handles large microscope slides without manual cropping.
* **Artifact Removal:** Cleans up overlapping boxes from tiling.
* **CSV Reports:** Generates summary statistics (Viability %) automatically.

---

**License:** MIT

