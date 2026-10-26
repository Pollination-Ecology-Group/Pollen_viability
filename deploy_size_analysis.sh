#!/bin/bash
set -e

NAMESPACE="stenc-ns"
KUBECTL="./kubectl"

if [ -f "./kubeconfig.yaml" ]; then
    export KUBECONFIG="$(pwd)/kubeconfig.yaml"
fi

echo "=========================================================="
echo "🌸 Pollen Viability Size & Stitching Job (CESNET K8s)"
echo "=========================================================="

echo "☁️  1. Uploading ConfigMaps..."
$KUBECTL create configmap size-analysis-script \
    --from-file=pollen_size_analyzer.py=src/pollen_size_analyzer.py \
    -n $NAMESPACE --dry-run=client -o yaml | $KUBECTL apply -f -

$KUBECTL create configmap size-analysis-meta \
    --from-file=plant_metadata_540.csv=metadata/plant_metadata_540.csv \
    -n $NAMESPACE --dry-run=client -o yaml | $KUBECTL apply -f -

echo "🧹 2. Cleaning up old size analysis job..."
$KUBECTL delete job pollen-size-analysis-job -n $NAMESPACE --ignore-not-found --wait=true

echo "🚀 3. Launching pollen-size-analysis-job..."
$KUBECTL apply -f k8s/pollen-size-analysis-job.yaml

echo "⏳ 4. Waiting for pod to start (pulling image if needed)..."
while true; do
    PHASE=$($KUBECTL get pod -l job-name=pollen-size-analysis-job -n $NAMESPACE -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Waiting")
    if [ "$PHASE" = "Running" ] || [ "$PHASE" = "Succeeded" ] || [ "$PHASE" = "Failed" ]; then
        echo "   Pod is in phase: $PHASE"
        break
    fi
    echo "   Current phase: $PHASE (waiting...)"
    sleep 10
done

echo "👀 5. Streaming logs until completion..."
$KUBECTL logs -f job/pollen-size-analysis-job -n $NAMESPACE

echo "📥 6. Downloading consolidated results from S3..."
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

echo "🎉 Ready to render the Quarto report!"
