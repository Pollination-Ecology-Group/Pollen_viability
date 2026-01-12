
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
sed "s|image: .*|image: $IMAGE_NAME|g" pollen-job.yaml | $KUBECTL apply -f -

echo "⏳ 5. Waiting for Pod to start..."
# Wait a bit for the pod to be scheduled
sleep 5

echo "👀 6. Streaming logs..."
$KUBECTL logs -f job/pollen-detector-job -n $NAMESPACE
