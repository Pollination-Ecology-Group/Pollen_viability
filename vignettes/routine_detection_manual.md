# Routine Pollen Detection Manual

This guide covers the complete workflow for conducting routine pollen detection using the automated Kubernetes pipeline. 

## 1. Accessing the CESNET S3 Storage

Before running the detection pipeline, you need access to the Faculty's CESNET S3 storage to upload images and download results.

👉 **[CESNET S3 Connection Guide](CESNET_connection_guide.md)** — Follow this guide to apply for access, generate your API keys via Gatekeeper, and connect using Cyberduck (or Rclone for advanced users).

---

## 2. Routine Detection Workflow

Once your S3 access is configured, you can run the routine pollen detection pipeline. 

### Step 1: Upload Images
Using Cyberduck, upload the high-resolution microscope images (TIF or JPG format) you want to analyze into the corresponding input directory on the S3 storage (typically `Ostatni/Pollen_viability/detect_images/`, depending on your specific prefix structure).

### Step 2: Run the Kubernetes Job
On your local machine (or the designated VM where the project repository `Pollen_viability` is set up), open a terminal and run the deployment script:

```bash
cd Pollen_viability
./deploy_pollen.sh
```
*(Note: If you encounter Docker permission issues on a fresh machine, you may need to run it with `sudo ./deploy_pollen.sh` or configure your user group with `sg docker`).*

This script will:
1. Build a temporary Docker container with the latest detection code.
2. Push the container to a temporary registry (`ttl.sh`).
3. Deploy a Kubernetes job (`pollen-detector-job`) to the CESNET cluster.
4. Stream the live logs so you can monitor the progress.

### Step 3: Retrieve the Results
The detection script will automatically process the images in chunks, utilizing the cluster's GPU. 

Once the job finishes:
* **Annotated Images:** The images with overlaid prediction bounding boxes/polygons will be uploaded back to the S3 bucket (typically into `detected_images/`). You can download these using Cyberduck.
* **Counts and Statistics:** A summary CSV file (`pollen_counts.csv`) containing the total counts of viable and non-viable grains, as well as calculated viability percentages, will be generated.
* The `deploy_pollen.sh` script automatically limits syncing your local `pollen_counting_results/` directory using the `src/sync_results.py` sync script right after detection finishes.

If you ever need to manually download the latest CSVs without re-running the detection pipeline, you can run:
```bash
python3 src/sync_results.py
```

### Troubleshooting
If you encounter any issues setting up the connection or running the pipeline, please reach out or schedule a meeting for assistance.
