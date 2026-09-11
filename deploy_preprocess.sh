#!/bin/bash
set -e

RANDOM_ID=$(openssl rand -hex 4)
IMAGE_NAME="ttl.sh/pollen-preprocess-$RANDOM_ID:24h"
NAMESPACE="stenc-ns"

echo "-------------------------------------"
echo "🌸 Pollen Preprocessing Deployment Script"
echo "-------------------------------------"


# Check for kubectl
if ! command -v kubectl &> /dev/null; then
    if [ ! -f "./kubectl" ]; then
        echo "⬇️ kubectl not found in PATH. Downloading local binary..."
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

echo "☁️ 1. Deploying script as configmap..."
$KUBECTL create configmap preprocess-script --from-file=src/preprocess_czi.py -n stenc-ns --dry-run=client -o yaml | $KUBECTL apply -f -

echo "🧹 2. Cleaning up old jobs..."
$KUBECTL delete job pollen-preprocess-job -n $NAMESPACE --ignore-not-found

echo "🚀 3. Deploying job..."
$KUBECTL apply -f k8s/pollen-preprocess-job.yaml

echo "⏳ 5. Waiting for Pod to start..."
max_retries=300 
count=0
echo -n "   Waiting for pod to start (max 10m)..."
while : ; do
    POD_NAME=$($KUBECTL get pods -n $NAMESPACE -l job-name=pollen-preprocess-job --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || echo "")
    
    if [ -n "$POD_NAME" ]; then
        POD_INFO=$($KUBECTL get pod "$POD_NAME" -n $NAMESPACE -o jsonpath='{.status.phase} {.status.containerStatuses[0].state}' 2>/dev/null || echo "NotFound {}")
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
        echo " ❌ Timeout waiting for pod. pod=$POD_NAME phase=$PHASE state=$STATE"
        exit 1
    fi
    
    echo -n "."
    sleep 2
    count=$((count+1))
done

echo "👀 6. Streaming logs..."
$KUBECTL logs -f job/pollen-preprocess-job -n $NAMESPACE --ignore-errors || true

echo "✅ Preprocessing complete! Check S3 for the generated tiles."
