#!/bin/bash
set -e

NAMESPACE="stenc-ns"
KUBECTL="./kubectl"

if [ -f "./kubeconfig.yaml" ]; then
    export KUBECONFIG="$(pwd)/kubeconfig.yaml"
fi

echo "👀 Streaming logs from pollen-size-analysis-job until completion..."
$KUBECTL logs -f job/pollen-size-analysis-job -n $NAMESPACE

echo "📥 Downloading consolidated results from S3..."
mkdir -p output/size_analysis
python3 - <<'PY'
import boto3, os
from botocore.client import Config
s3 = boto3.client('s3',
    endpoint_url=os.environ.get('S3_ENDPOINT', 'https://s3.cl4.du.cesnet.cz'),
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', '1Y920BKC0SAWPNDE8RD6'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', 'SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD'),
    config=Config(signature_version='s3v4'))
bucket = os.environ.get('S3_BUCKET', 'peg-bucket')
pfx = 'Ostatni/Pollen_viability/size_analysis/'
for fname in ['pollen_slide_summaries.csv', 'pollen_grain_measurements.csv']:
    try:
        s3.download_file(bucket, pfx + fname, f'output/size_analysis/{fname}')
        sz = os.path.getsize(f'output/size_analysis/{fname}')
        print(f"   ✅ Downloaded output/size_analysis/{fname} ({sz/1e3:.1f} KB)")
    except Exception as e:
        print(f"   ⚠️ Could not download {fname}: {e}")
PY

echo "Rendering Quarto report..."
quarto render outcomes/pollen_viability_report.qmd
echo "🎉 Quarto render complete!"
