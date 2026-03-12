# Routine Pollen Detection Manual

This guide covers the complete workflow for conducting routine pollen detection using the automated Kubernetes pipeline. 

## 1. Accessing the CESNET S3 Storage
To upload your microscope images and retrieve the final results, you first need access to the Faculty's CESNET S3 storage.

### 1.1 Applying for Access
1. **Fill out the application** for our Virtual Organization (VO) at: [https://einfra.cesnet.cz/fed/registrar/?vo=VO_prfuk](https://einfra.cesnet.cz/fed/registrar/?vo=VO_prfuk)
2. Once your application is approved, an administrator will add you to the appropriate group.

### 1.2 Generating Access Keys
Next, you must generate access credentials in the Gatekeeper system.
1. Go to the **Gatekeeper application**: [https://access.du.cesnet.cz/](https://access.du.cesnet.cz/)
2. Click the blue **LOG IN** button. 
3. From the list of institutions, select **Univerzita Karlova** (Charles University) and log in using your university CAS credentials (e.g., `matouse3` + password).
4. In the left menu, select the virtual organization **`VO_prfuk`**, and then select the group **`VO_prfuk_1200:FEG`**.
5. Click the blue **+ New key** button to create a new key pair. You can name it anything you like.
6. **Important:** Note down your **Access Key** and **Secret Key**. Once you close the window, the Secret Key will never be shown again. If you lose it, you will need to generate a new pair of keys.

### 1.3 Setting up Cyberduck
We recommend using Cyberduck to transfer files between your computer and the S3 storage.
1. **Download and install** Cyberduck: [https://cyberduck.io/download/](https://cyberduck.io/download/)
2. Open the program and click **Open Connection** (Nové spojení).
3. Select **Amazon S3** from the protocol dropdown menu.
4. Fill in the connection details:
   * **Server:** `s3.cl4.du.cesnet.cz`
   * **Access Key:** Enter the Access Key you generated.
   * **Secret Key:** Enter the Secret Key you generated.
5. Expand **More Options** (Více voleb) and in the **Path** field, enter `/feg` (or the specific path to your project bucket).
6. Click **Connect** (Připojit).
7. *Tip:* For faster access in the future, save this connection as a bookmark. Click the bookmark panel icon, click the **"+"** button at the bottom, and enter a name in the "Nickname" field. Leave the other settings as they are.
8. You can now use the File menu to create folders, upload images, and download results.

*(For detailed official guides, see the [CESNET Gatekeeper guide](https://du.cesnet.cz/cs/navody/object_storage/gatekeeper/start) and [Cyberduck guide](https://du.cesnet.cz/cs/navody/object_storage/cyberduck/start).)*

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
