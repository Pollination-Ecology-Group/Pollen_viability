#!/bin/bash
# deploy_detect.sh — Run FastSAM detection on all S3 tiles (K8s GPU cluster)
#
# Replaces the old YOLO-based detection with FastSAM (class-agnostic segmentation).
# FastSAM writes _det.json files in the same format — Pinder reads them unchanged.
#
# Usage:
#   ./deploy_detect.sh            # skip tiles that already have _det.json
#   ./deploy_detect.sh --force    # overwrite ALL existing _det.json (re-detect everything)
set -e

NAMESPACE="stenc-ns"

# ── kubectl setup ──────────────────────────────────────────────────────────────
if ! command -v kubectl &>/dev/null; then
    if [ ! -f "./kubectl" ]; then
        echo "⬇️ kubectl not found. Downloading..."
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        chmod +x kubectl
    fi
    KUBECTL="./kubectl"
else
    KUBECTL="kubectl"
fi

if [ -f "./kubeconfig.yaml" ]; then
    export KUBECONFIG="$(pwd)/kubeconfig.yaml"
fi

echo "----------------------------------------------"
echo "🌸 Pollen SAM Detection Batch Job (K8s GPU cluster)"
echo "----------------------------------------------"

# ── Optional --force flag ──────────────────────────────────────────────────────
FORCE_FLAG=""
if [[ "$1" == "--force" ]]; then
    FORCE_FLAG="--force"
    echo "⚠️  --force — all existing _det.json will be overwritten!"
fi

# ── Upload FastSAM script as ConfigMap ────────────────────────────────────────
echo "☁️  1. Uploading run_sam_s3.py as ConfigMap (sam-script)..."
$KUBECTL create configmap sam-script \
    --from-file=run_sam_s3.py=src/run_sam_s3.py \
    -n $NAMESPACE --dry-run=client -o yaml | $KUBECTL apply -f -

# ── Upload FastSAM-s.pt weights to S3 (if not already there) ──────────────────
echo "🔧 2. Checking FastSAM-s.pt on S3..."
set -a                            # auto-export all variables
source .env 2>/dev/null || true   # load credentials from .env
set +a                            # stop auto-exporting
MODEL_S3_KEY="Ostatni/Pollen_viability/trained_models/FastSAM-s.pt"
MODEL_LOCAL="FastSAM-s.pt"

if [ -f "$MODEL_LOCAL" ]; then
    python3 - <<PYEOF
import boto3, os
from botocore.client import Config
s3 = boto3.client('s3',
    endpoint_url=os.environ.get('S3_ENDPOINT', 'https://s3.cl4.du.cesnet.cz'),
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    config=Config(signature_version='s3v4'))
bucket = os.environ.get('S3_BUCKET', 'bucket')
key = 'Ostatni/Pollen_viability/trained_models/FastSAM-s.pt'
try:
    s3.head_object(Bucket=bucket, Key=key)
    print('   FastSAM-s.pt already on S3 — skipping upload.')
except Exception:
    print('   Uploading FastSAM-s.pt to S3...')
    s3.upload_file('FastSAM-s.pt', bucket, key)
    print('   ✅ Uploaded.')
PYEOF
else
    echo "   ⚠️  FastSAM-s.pt not found locally — K8s job will try to download from S3."
    echo "      If it fails, copy FastSAM-s.pt to the repo root and re-run this script."
fi

# ── Patch --force into the job yaml if requested ──────────────────────────────
JOB_YAML="k8s/pollen-sam-job.yaml"
if [ "$FORCE_FLAG" = "--force" ]; then
    echo "🔧 3. Patching --force into job command..."
    sed 's/--all-folders/--all-folders --force/' "$JOB_YAML" > /tmp/pollen-sam-job-force.yaml
    JOB_YAML="/tmp/pollen-sam-job-force.yaml"
else
    echo "🔧 3. Using job yaml as-is (skip existing _det.json)..."
fi

# ── Clean up old job ───────────────────────────────────────────────────────────
echo "🧹 4. Cleaning up old sam job..."
$KUBECTL delete job pollen-sam-job -n $NAMESPACE --ignore-not-found

# ── Launch job ─────────────────────────────────────────────────────────────────
echo "🚀 5. Launching FastSAM detection job on GPU cluster..."
$KUBECTL apply -f $JOB_YAML

# ── Wait for pod ───────────────────────────────────────────────────────────────
echo "⏳ 6. Waiting for pod to start..."
max_retries=300
count=0
echo -n "   Waiting (max 10m)..."
while : ; do
    POD_NAME=$($KUBECTL get pods -n $NAMESPACE -l job-name=pollen-sam-job \
        --sort-by=.metadata.creationTimestamp \
        -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || echo "")

    if [ -n "$POD_NAME" ]; then
        POD_INFO=$($KUBECTL get pod "$POD_NAME" -n $NAMESPACE \
            -o jsonpath='{.status.phase} {.status.containerStatuses[0].state}' \
            2>/dev/null || echo "NotFound {}")
        PHASE=$(echo "$POD_INFO" | cut -d' ' -f1)
        STATE=$(echo "$POD_INFO" | cut -d' ' -f2-)

        if [ "$PHASE" = "Running" ] || [ "$PHASE" = "Succeeded" ] || [ "$PHASE" = "Failed" ]; then
            if echo "$STATE" | grep -qvE "waiting|ContainerCreating|PodInitializing"; then
                echo " ✅ ($PHASE)"
                break
            fi
        fi
    fi

    if [ $count -gt $max_retries ]; then
        echo " ❌ Timeout waiting for pod."
        exit 1
    fi

    echo -n "."
    sleep 2
    count=$((count+1))
done

# ── Stream logs ────────────────────────────────────────────────────────────────
echo "👀 7. Streaming logs (Ctrl+C to detach — job continues on cluster)..."
$KUBECTL logs -f job/pollen-sam-job -n $NAMESPACE --ignore-errors || true

echo ""
echo "✅ FastSAM detection job complete!"
echo "   All tiles now have companion _det.json files (FastSAM segmentation)."
echo "   Pinder Swipe Mode will show grain crops for manual viable/NV labelling."
