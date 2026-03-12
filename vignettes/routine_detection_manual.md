# Routine Pollen Detection Manual

This guide walks you through the full process of running an automated pollen viability analysis — from preparing your images to downloading your results. **No programming experience is required.**

> [!NOTE]
> This pipeline runs the heavy computation on a remote university server (CESNET cluster), so it works even on modest laptops. You only need to start the job from your computer — the actual image analysis happens remotely.

---

## Before You Start: One-Time Setup

You only need to do these steps once when setting up the workflow for the first time.

### Step 0.1 — Get Access to the Cloud Storage (S3)

All images and results are stored in the CESNET S3 cloud. You need access before anything else.

👉 **[Follow the CESNET S3 Connection Guide](CESNET_connection_guide.md)**

This will take you through applying for access, generating your personal keys, and installing **Cyberduck** (the tool you will use to upload and download files).

---

### Step 0.2 — Set Up the Detection Script on Your Computer

The detection pipeline is run via a script. You need to have the project code and its required tools installed **once** on the computer you'll use to launch jobs.

#### 0.2a — Install Required Software

You will need three programs installed:

| Program | What it does | Download |
|---|---|---|
| **Git** | Downloads the project code | [git-scm.com](https://git-scm.com/downloads) |
| **Docker Desktop** | Packages the detection code to run on the cluster | [docker.com/get-started](https://www.docker.com/get-started/) |
| **WSL 2** *(Windows only)* | Lets Windows run Linux-style scripts | [Microsoft Docs](https://learn.microsoft.com/en-us/windows/wsl/install) |

> [!IMPORTANT]
> On **Windows**: After installing WSL 2, open **Docker Desktop → Settings → Resources → WSL Integration** and enable it for your Linux distribution.
> On **Linux**: After installing Docker, run `sudo usermod -aG docker $USER` and log out and back in.

#### 0.2b — Download the Project Code

Open a terminal (on Windows: search for **"Ubuntu"** or **"WSL"** in the Start menu) and type:

```bash
git clone https://github.com/Pollination-Ecology-Group/Pollen_viability.git
cd Pollen_viability
```

This downloads all the scripts you need into a folder called `Pollen_viability`.

#### 0.2c — Download the Cluster Access File

You will also need the **`kubeconfig.yaml`** file — this is the key that lets your computer talk to the CESNET cluster. Ask the project lead to send you this file, then place it inside the `Pollen_viability` folder you just downloaded.

---

## Routine Detection — Step by Step

Once setup is complete, running detection every time takes just 3 steps.

### Step 1 — Upload Your Images

Using **Cyberduck**, navigate to the `Ostatni/Pollen_viability/detect_images/` folder in the S3 bucket and drag your microscope images (TIF or JPG format) into it.

> [!TIP]
> You can upload a whole folder at once. Just drag it into the Cyberduck window.

---

### Step 2 — Launch the Detection Job

Open a terminal (on Windows: **Ubuntu / WSL**) and navigate to the project folder:

```bash
cd Pollen_viability
```

Then run:

```bash
./deploy_pollen.sh
```

> [!NOTE]
> The first time you run this, Docker will download several components. This may take 5–15 minutes depending on your internet speed. Subsequent runs are much faster.

You will see output in the terminal that looks like this:
```
🌸 Pollen Detector Deployment Script
🚀 1. Building Docker image...
☁️ 2. Pushing image...
🚀 4. Deploying Job to Cluster...
👀 6. Streaming logs...
```

The terminal will then show live progress from the cluster. **You don't need to do anything** — just wait.

> [!TIP]
> If you see a `permission denied` error when running `./deploy_pollen.sh`, try:
>
> **Linux:** `sg docker -c ./deploy_pollen.sh`
>
> **Windows (WSL):** Make sure Docker Desktop is running and WSL integration is enabled.

---

### Step 3 — Download Your Results

Once the detection finishes (the terminal shows it has completed), your results are automatically uploaded to the S3 bucket. 

Open **Cyberduck** and navigate to:
- `Ostatni/Pollen_viability/detected_images/` — **Annotated images** with coloured bounding boxes showing which pollen grains are viable (green) and non-viable (red).
- `Ostatni/Pollen_viability/pollen_counting_results/` — **CSV summary files** (`pollen_counts.csv`) with counts and viability percentages per image.

Download these files to your computer by right-clicking → **Download** in Cyberduck.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `permission denied` when running script | See the tip in Step 2 above |
| Docker Desktop not starting | Make sure WSL 2 is installed and enabled |
| Script says "cannot connect to cluster" | Check that `kubeconfig.yaml` is in the `Pollen_viability` folder |
| No files appear in S3 after job finishes | Wait a few minutes; large jobs can take time to upload results |
| Any other issue | Contact the project lead or schedule a meeting |
