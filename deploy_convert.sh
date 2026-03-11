#!/bin/bash
set -e

RANDOM_ID=$(openssl rand -hex 4)
IMAGE_NAME="ttl.sh/pollen-convert-$RANDOM_ID:24h"
NAMESPACE="stenc-ns"

echo "-------------------------------------"
echo "♻️ Pollen Conversion Deployment Script"
echo "-------------------------------------"

echo "🚀 1. Building Docker image..."
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker permission denied or Docker not running."
    exit 1
fi
docker build -t $IMAGE_NAME .

echo "☁️ 2. Pushing image to registry..."
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

KUBECTL="kubectl"
if [ -f "./kubectl" ]; then KUBECTL="./kubectl"; fi
if [ -f "./kubeconfig.yaml" ]; then export KUBECONFIG="$(pwd)/kubeconfig.yaml"; fi

echo "🧹 3. Cleaning up old conversion jobs..."
$KUBECTL delete job pollen-convert-job -n $NAMESPACE --ignore-not-found

echo "🚀 4. Deploying Conversion Job..."
sed "s|image: ttl.sh/pollen-detector-REPLACE_ME:24h|image: $IMAGE_NAME|g" k8s/pollen-convert-job.yaml | $KUBECTL apply -f -

echo "👀 5. Streaming logs..."
sleep 5
# Wait for pod
while ! $KUBECTL logs job/pollen-convert-job -n $NAMESPACE > /dev/null 2>&1; do
    echo -n "."
    sleep 5
done
echo ""

$KUBECTL logs -f job/pollen-convert-job -n $NAMESPACE
