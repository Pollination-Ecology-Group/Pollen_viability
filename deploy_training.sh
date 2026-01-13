
#!/bin/bash
set -e

# Configuration
# Using ttl.sh ephemeral registry
RANDOM_ID=$(openssl rand -hex 4)
IMAGE_NAME="ttl.sh/pollen-trainer-$RANDOM_ID:24h"
NAMESPACE="stenc-ns"

echo "-------------------------------------"
echo "🌸 Pollen Training Deployment Script"
echo "-------------------------------------"

echo "🚀 1. Building Docker image..."
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker permission denied or Docker not running."
    echo "   👉 Try running this script with sudo: sudo ./deploy_training.sh"
    exit 1
fi
docker build -t $IMAGE_NAME .

echo "☁️ 2. Pushing image to registry..."
# Retry push up to 5 times for flaky networks
n=0
until [ "$n" -ge 5 ]
do
   docker push $IMAGE_NAME && break
   n=$((n+1)) 
   echo "⚠️ Push failed. Retrying ($n/5) in 5 seconds..."
   sleep 5
done

if [ "$n" -ge 5 ]; then
   echo "❌ Failed to push image after 5 attempts."
   exit 1
fi

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

echo "🧹 3. Cleaning up old training jobs..."
$KUBECTL delete job pollen-train-job -n $NAMESPACE --ignore-not-found

echo "🚀 4. Deploying Training Job..."
# Inject dynamic image name
sed "s|image: ttl.sh/pollen-detector-REPLACE_ME:24h|image: $IMAGE_NAME|g" k8s/pollen-train-job.yaml | $KUBECTL apply -f -

echo "⏳ 5. Waiting for Pod to initialize..."
echo "   (This may take a few minutes if pulling the heavy 6GB image for the first time)"
echo -n "   Waiting."

# Retry getting logs until successful (indicating container is running)
count=0
# Try to just read logs (not stream) to check readiness
while ! $KUBECTL logs job/pollen-train-job -n $NAMESPACE > /dev/null 2>&1; do
    echo -n "."
    sleep 5
    count=$((count+1))
    if [ $count -ge 120 ]; then # 10 minutes timeout
        echo ""
        echo "❌ Timed out waiting for pod to start. Please check status manually:"
        echo "   $KUBECTL describe job pollen-train-job -n $NAMESPACE"
        exit 1
    fi
done

echo ""
echo "👀 6. Streaming logs..."
$KUBECTL logs -f job/pollen-train-job -n $NAMESPACE

