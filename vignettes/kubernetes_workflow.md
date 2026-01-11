# Kubernetes Workflow Guide for Pollen Viability

This guide documents the automated workflow for deploying Pollen Viability detection and training jobs to the CERIT-SC Kubernetes cluster.

## 🏗️ Architecture Overview

The system runs as ephemeral **Kubernetes Jobs**. It follows this data flow:

1.  **Input**: Data (images/datasets) is read from **S3 Storage**.
2.  **Processing**: A Docker container crashes processing logic (YOLOv8 + Custom Scripts).
3.  **Output**: Results (CSV, Annotated Images, Trained Models) are uploaded back to **S3**.

## 🚀 Quick Start

### 1. Prerequisites
- **Kubeconfig**: You must have a valid `kubeconfig.yaml` in the project root.
    > **How to obtain:**
    > 1. Log in to [Rancher Dashboard](https://rancher.cloud.e-infra.cz).
    > 2. Open your assigned cluster.
    > 3. Click **"Download Kubeconfig"** (Top Right) or "Kubeconfig File" icon.
    > 4. Save as `kubeconfig.yaml` in this project folder.
    > 5. **Security:** Run `chmod 600 kubeconfig.yaml` to secure the file.
- **Docker**: Must be installed locally to build images.
- **kubectl**: auto-downloaded by the scripts if missing.

### 2. Running Detection
To detect viable pollen in new images:

1.  Ensure input images are in S3: `Ostatni/Pollen_viability/detect_images`.
2.  Run the deployment script:
    ```bash
    ./deploy_pollen.sh
    ```
3.  **What happens?**
    - The script builds a Docker image.
    - Pushes it to `ttl.sh` (ephemeral registry).
    - Launches a K8s Job (`pollen-detector-job`) in namespace `stenc-ns`.
    - Streams logs until completion.
4.  **Results**: Check S3: `Ostatni/Pollen_viability/detected_images`.

### 3. Running Training
To retrain the YOLOv8 model:

1.  Ensure datasets are in S3 (`datasets/pollen_v1`).
2.  Run the training deployment:
    ```bash
    ./deploy_training.sh
    ```
3.  **Process**:
    - Syncs datasets and staging area from S3.
    - Merges new labelled data.
    - Generates synthetic negative samples.
    - Trains the model (approx 2-4 hours).
    - Uploads the best model (`best.pt`) and charts to S3.

## 📂 File Structure

| File                    | Purpose                                                                      |
| ----------------------- | ---------------------------------------------------------------------------- |
| `deploy_pollen.sh`      | Orchestrates the detection workflow. Builds Docker -> Deploys Job.           |
| `deploy_training.sh`    | Orchestrates the training workflow.                                          |
| `pollen-job.yaml`       | Kubernetes Manifest for detection. Defines resources (GPU/RAM) and env vars. |
| `pollen-train-job.yaml` | Kubernetes Manifest for training.                                            |
| `Dockerfile`            | Defines the environment (Python, YOLOv8, non-root user).                     |
| `run_detection.py`      | Core script for detection logic + S3 sync.                                   |
| `train_model.py`        | Core script for training logic + S3 sync.                                    |

## 🔑 Access & Security
- **Namespace**: `stenc-ns` (Your isolated environment).
- **Users**: Pods run as `appuser` (UID 1000) for security compliance.
- **Secrets**: AWS credentials are injected via Environment Variables in the YAML files.

## 🛠️ Troubleshooting

- **"Forbidden" errors**: Ensure `kubeconfig.yaml` is correct and you are deploying to `stenc-ns`.
- **"ImagePullBackOff"**: The `ttl.sh` image might have expired (they last 24h). Re-run the deploy script to rebuild and push.
- **Empty Output**: Check S3 paths in `run_detection.py`. Verify source images exist.
