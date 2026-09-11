#!/bin/bash
set -e

echo "🌸 1. Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "📦 2. Installing requirements (This might take a minute to download PyTorch...)"
pip install --upgrade pip
pip install streamlit ultralytics boto3 Pillow streamlit-drawable-canvas
pip uninstall -y opencv-python
pip install --force-reinstall opencv-python-headless

echo "🔑 3. Exporting S3 Credentials..."
export S3_ENDPOINT="https://s3.cl4.du.cesnet.cz"
export S3_BUCKET="bucket"
export AWS_ACCESS_KEY_ID="1Y920BKC0SAWPNDE8RD6"
export AWS_SECRET_ACCESS_KEY="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD"

echo "🚀 4. Starting GUI..."
streamlit run app_gui.py --server.headless true --server.fileWatcherType none
