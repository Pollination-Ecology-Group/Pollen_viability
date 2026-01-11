
#!/bin/bash
set -e

# Configuration
# Using ttl.sh ephemeral registry
RANDOM_ID=$(openssl rand -hex 4)
IMAGE_NAME="ttl.sh/pollen-trainer-$RANDOM_ID:24h"

echo "-------------------------------------"
echo "🌸 Pollen Training Deployment Script"
echo "-------------------------------------"

echo "🚀 1. Building Docker image..."
sudo docker build -t $IMAGE_NAME .

echo "☁️ 2. Pushing image to registry..."
sudo docker push $IMAGE_NAME

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

echo "🧹 3. Cleaning up old training jobs..."
$KUBECTL delete job pollen-train-job --ignore-not-found

echo "🚀 4. Deploying Training Job..."
# Inject dynamic image name
sed "s|image: ttl.sh/pollen-detector-REPLACE_ME:24h|image: $IMAGE_NAME|g" pollen-train-job.yaml | $KUBECTL apply -f -

echo "⏳ 5. Waiting for Pod to start..."
sleep 5

echo "👀 6. Streaming logs..."
$KUBECTL logs -f job/pollen-train-job
