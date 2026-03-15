
#!/bin/bash
set -e

# Configuration
# Using ttl.sh ephemeral registry (no login required, images last 24h)
# We add a random suffix to avoid collisions
RANDOM_ID=$(openssl rand -hex 4)
IMAGE_NAME="ttl.sh/pollen-detector-$RANDOM_ID:24h"
NAMESPACE="stenc-ns"

echo "-------------------------------------"
echo "🌸 Pollen Detector Deployment Script"
echo "-------------------------------------"

echo "🚀 1. Building Docker image..."
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker permission denied or Docker not running."
    echo "   👉 Try running this script with sudo: sudo ./deploy_pollen.sh"
    exit 1
fi
docker build -t $IMAGE_NAME .

echo "☁️ 2. Pushing image to registry..."
# Ensure you are logged in via 'docker login'
docker push $IMAGE_NAME

echo "🧹 3. Cleaning up old jobs..."
# Check for kubectl
if ! command -v kubectl &> /dev/null; then
    if [ ! -f "./kubectl" ]; then
        echo "⬇️ kubectl not found. Downloading locally..."
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        chmod +x kubectl
    fi
    KUBECTL="./kubectl"
else
    KUBECTL="kubectl"
fi

# Check for local kubeconfig
if [ -f "./kubeconfig.yaml" ]; then
    echo "🔑 Using local kubeconfig.yaml..."
    export KUBECONFIG="$(pwd)/kubeconfig.yaml"
fi

echo "🧹 3. Cleaning up old jobs..."
$KUBECTL delete job pollen-detector-job -n $NAMESPACE --ignore-not-found

echo "🚀 4. Deploying Job to Cluster..."
# We need to update the yaml with the generated dynamic image name
sed "s|image: .*|image: $IMAGE_NAME|g" k8s/pollen-job.yaml | $KUBECTL apply -f -

echo "⏳ 5. Waiting for Pod to start..."
# Wait for the pod to be created and reach a loggable state
max_retries=300 # 10 minutes (300 * 2s)
count=0
echo -n "   Waiting for pod to start (max 10m)..."
while : ; do
    # Get the name of the most recent pod for this job
    POD_NAME=$($KUBECTL get pods -n $NAMESPACE -l job-name=pollen-detector-job --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || echo "")
    
    if [ -n "$POD_NAME" ]; then
        # Get Pod Phase and Container State
        POD_INFO=$($KUBECTL get pod "$POD_NAME" -n $NAMESPACE -o jsonpath='{.status.phase} {.status.containerStatuses[0].state}' 2>/dev/null || echo "NotFound {}")
        PHASE=$(echo "$POD_INFO" | cut -d' ' -f1)
        STATE=$(echo "$POD_INFO" | cut -d' ' -f2-)

        if [ "$PHASE" = "Running" ] || [ "$PHASE" = "Succeeded" ] || [ "$PHASE" = "Failed" ]; then
            # Ensure container is not still creating
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
$KUBECTL logs -f job/pollen-detector-job -n $NAMESPACE --ignore-errors || true


echo ""
echo "🔄 7. Auto-syncing results..."
PYTHON_CMD="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
fi

$PYTHON_CMD src/sync_results.py

# Fix permissions if running as root (via sudo)
if [ -n "$SUDO_USER" ]; then
    chown -R $SUDO_USER:$SUDO_USER pollen_counting_results
fi

