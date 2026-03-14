
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
max_retries=30
count=0
echo -n "   Waiting for pod..."
while : ; do
    # Get Pod Phase and Container State
    POD_JSON=$($KUBECTL get pods -n $NAMESPACE -l job-name=pollen-detector-job -o json 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(json.dumps(data['items'][0] if data.get('items') else {}))" 2>/dev/null || echo "{}")
    POD_PHASE=$(echo "$POD_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', {}).get('phase', 'NotFound'))" 2>/dev/null || echo "NotFound")
    
    # Check if container is actually running or finished (not in ContainerCreating)
    CONTAINER_STATE=$(echo "$POD_JSON" | python3 -c "import sys, json; 
try:
    s = json.load(sys.stdin).get('status', {}).get('containerStatuses', [{}])[0].get('state', {})
    if 'waiting' in s: print(s['waiting'].get('reason', 'Waiting'))
    elif 'running' in s: print('Running')
    elif 'terminated' in s: print('Terminated')
    else: print('Unknown')
except: print('NotFound')
" 2>/dev/null || echo "NotFound")

    if [ "$POD_PHASE" = "Running" ] || [ "$POD_PHASE" = "Succeeded" ] || [ "$POD_PHASE" = "Failed" ]; then
        if [ "$CONTAINER_STATE" != "ContainerCreating" ] && [ "$CONTAINER_STATE" != "PodInitializing" ] && [ "$CONTAINER_STATE" != "Waiting" ]; then
             echo " ✅ ($POD_PHASE/$CONTAINER_STATE)"
             break
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

