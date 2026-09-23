#!/bin/bash
# deploy_inference_test.sh — Run single-image inference test on K8s GPU cluster
#
# Uploads run_inference_test.py as a ConfigMap, launches a lightweight job
# that downloads ONE tile, runs the new YOLO model, and saves annotated
# output + JSON to S3 under:
#   Ostatni/Pollen_viability/inference_tests/<timestamp>/
#
# Usage:
#   ./deploy_inference_test.sh
#   ./deploy_inference_test.sh "Ostatni/Pollen_viability/tiles_640/some/tile.jpg"
#                                 ↑ optional: specify a particular tile to test
set -e

NAMESPACE="stenc-ns"
JOB_NAME="pollen-inference-test-job"
JOB_YAML="k8s/pollen-inference-test-job.yaml"
SCRIPT_SRC="src/run_inference_test.py"

# ── kubectl setup ─────────────────────────────────────────────────────────────
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

echo "──────────────────────────────────────────────────────"
echo "🌸 Pollen — Single-Image Inference Test (K8s GPU)"
echo "   Model: pollen_train_20260923_1457/weights/best.pt"
echo "──────────────────────────────────────────────────────"

# ── Optional specific tile argument ──────────────────────────────────────────
PATCH_YAML="$JOB_YAML"
if [ -n "$1" ]; then
    TILE_KEY="$1"
    echo "📌 Pinning test tile: $TILE_KEY"
    TMP_YAML="/tmp/${JOB_NAME}-patched.yaml"
    sed "s|# - name: TEST_TILE_KEY.*|- name: TEST_TILE_KEY|;
         s|#   value: \"Ostatni.*\"|  value: \"${TILE_KEY}\"|" \
        "$JOB_YAML" > "$TMP_YAML"
    PATCH_YAML="$TMP_YAML"
fi

# ── Upload script as ConfigMap ────────────────────────────────────────────────
echo "☁️  1. Uploading $SCRIPT_SRC as ConfigMap (inference-test-script)..."
$KUBECTL create configmap inference-test-script \
    --from-file=run_inference_test.py="$SCRIPT_SRC" \
    -n $NAMESPACE --dry-run=client -o yaml | $KUBECTL apply -f -

# ── Clean up old job ──────────────────────────────────────────────────────────
echo "🧹 2. Cleaning up old inference test job (if any)..."
$KUBECTL delete job "$JOB_NAME" -n $NAMESPACE --ignore-not-found

# ── Launch job ────────────────────────────────────────────────────────────────
echo "🚀 3. Launching inference test job on GPU cluster..."
$KUBECTL apply -f "$PATCH_YAML"

# ── Wait for pod ──────────────────────────────────────────────────────────────
echo "⏳ 4. Waiting for pod to start (max 5 min)..."
max_retries=150
count=0
echo -n "   Waiting..."
while : ; do
    POD_NAME=$($KUBECTL get pods -n $NAMESPACE -l "job-name=$JOB_NAME" \
        --sort-by=.metadata.creationTimestamp \
        -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || echo "")

    if [ -n "$POD_NAME" ]; then
        PHASE=$($KUBECTL get pod "$POD_NAME" -n $NAMESPACE \
            -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
        STATE=$($KUBECTL get pod "$POD_NAME" -n $NAMESPACE \
            -o jsonpath='{.status.containerStatuses[0].state}' 2>/dev/null || echo "{}")

        if [[ "$PHASE" =~ ^(Running|Succeeded|Failed)$ ]]; then
            if ! echo "$STATE" | grep -qE "waiting|ContainerCreating|PodInitializing"; then
                echo " ✅ ($PHASE)"
                break
            fi
        fi
    fi

    if [ $count -gt $max_retries ]; then
        echo " ❌ Timeout. Check cluster status manually."
        exit 1
    fi
    echo -n "."
    sleep 2
    count=$((count+1))
done

# ── Stream logs ───────────────────────────────────────────────────────────────
echo "👀 5. Streaming logs..."
$KUBECTL logs -f "job/$JOB_NAME" -n $NAMESPACE --ignore-errors || true

# ── Final status ──────────────────────────────────────────────────────────────
echo ""
FINAL_PHASE=$($KUBECTL get job "$JOB_NAME" -n $NAMESPACE \
    -o jsonpath='{.status.conditions[0].type}' 2>/dev/null || echo "Unknown")

if [ "$FINAL_PHASE" = "Complete" ]; then
    echo "✅ Inference test job finished successfully!"
else
    echo "⚠️  Job ended with status: $FINAL_PHASE"
fi

echo ""
echo "📦 Results saved to S3:"
echo "   s3://bucket/Ostatni/Pollen_viability/inference_tests/<timestamp>/"
echo "   ├── test_input.jpg   — raw input tile"
echo "   ├── test_output.jpg  — annotated output (bboxes + masks + labels)"
echo "   └── test_results.json — structured detection JSON"
echo ""
echo "💡 Download results locally:"
echo "   python3 - <<'EOF'"
echo "import boto3, os"
echo "from botocore.client import Config"
echo "s3 = boto3.client('s3', endpoint_url='https://s3.cl4.du.cesnet.cz',"
echo "    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],"
echo "    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],"
echo "    config=Config(signature_version='s3v4'))"
echo "prefix = 'Ostatni/Pollen_viability/inference_tests/'"
echo "for page in s3.get_paginator('list_objects_v2').paginate(Bucket='bucket', Prefix=prefix):"
echo "    for obj in page.get('Contents', []):"
echo "        k = obj['Key']; fn = k.split('/')[-1]"
echo "        if fn: s3.download_file('bucket', k, fn); print('⬇️ ', fn)"
echo "EOF"
