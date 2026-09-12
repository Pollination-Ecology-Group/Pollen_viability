#!/bin/bash
# deploy_detect.sh — Run batch YOLO detection on all S3 tiles (K8s GPU cluster)
# Usage: ./deploy_detect.sh [--force]
#   --force   Set FORCE_REDETECT=1 to re-run detection on all tiles
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
echo "🌸 Pollen Detection Batch Job (K8s GPU cluster)"
echo "----------------------------------------------"

# ── Optional --force flag ──────────────────────────────────────────────────────
FORCE_REDETECT="0"
if [[ "$1" == "--force" ]]; then
    FORCE_REDETECT="1"
    echo "⚠️  FORCE_REDETECT=1 — all tiles will be re-processed!"
fi

# ── Upload detection script as ConfigMap ──────────────────────────────────────
echo "☁️  1. Uploading detection script as ConfigMap..."
$KUBECTL create configmap detect-script \
    --from-file=run_detections_s3.py=src/run_detections_s3.py \
    -n $NAMESPACE --dry-run=client -o yaml | $KUBECTL apply -f -

# ── Patch FORCE_REDETECT if --force ───────────────────────────────────────────
if [ "$FORCE_REDETECT" = "1" ]; then
    echo "🔧 2. Patching FORCE_REDETECT=1 in job manifest..."
    sed 's/value: "0"  # FORCE_REDETECT/value: "1"/' \
        k8s/pollen-detect-job.yaml > /tmp/pollen-detect-job-force.yaml
    JOB_YAML="/tmp/pollen-detect-job-force.yaml"
else
    JOB_YAML="k8s/pollen-detect-job.yaml"
fi

# ── Clean up old job ───────────────────────────────────────────────────────────
echo "🧹 3. Cleaning up old detect job..."
$KUBECTL delete job pollen-detect-job -n $NAMESPACE --ignore-not-found

# ── Launch job ─────────────────────────────────────────────────────────────────
echo "🚀 4. Launching detection job on GPU cluster..."
$KUBECTL apply -f $JOB_YAML

# ── Wait for pod ───────────────────────────────────────────────────────────────
echo "⏳ 5. Waiting for pod to start..."
max_retries=300
count=0
echo -n "   Waiting (max 10m)..."
while : ; do
    POD_NAME=$($KUBECTL get pods -n $NAMESPACE -l job-name=pollen-detect-job \
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
echo "👀 6. Streaming logs (Ctrl+C to detach — job continues on cluster)..."
$KUBECTL logs -f job/pollen-detect-job -n $NAMESPACE --ignore-errors || true

echo ""
echo "✅ Detection job complete!"
echo "   Tiles in S3 now have companion _det.json detection files."
echo "   Pinder (pinder.streamlit.app) will pick them up on next page load."
