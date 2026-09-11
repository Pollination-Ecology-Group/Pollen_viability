#!/bin/bash
set -e

NAMESPACE="stenc-ns"
GUI_FILE="app_gui.py"
DEPLOYMENT_FILE="k8s/pollen-gui-deployment.yaml"

echo "🌸 1. Updating ConfigMap with latest GUI code..."
# Delete existing configmap if it exists
kubectl delete configmap pollen-gui-code -n $NAMESPACE --ignore-not-found
# Create new configmap
kubectl create configmap pollen-gui-code --from-file=$GUI_FILE -n $NAMESPACE

echo "🚀 2. Deploying GUI to Kubernetes..."
kubectl apply -f $DEPLOYMENT_FILE
kubectl rollout restart deployment/pollen-gui -n $NAMESPACE

echo "⏳ Waiting for GUI Pod to be ready (This will take a minute since it installs pip packages on startup)..."
kubectl rollout status deployment/pollen-gui -n $NAMESPACE --timeout=120s

echo "🌐 3. Creating Port-Forward to local machine..."
echo "------------------------------------------------------"
echo "The GUI is now running on the cluster!"
echo "To access it, keep this script running and open your browser to:"
echo "http://localhost:8501"
echo "Press Ctrl+C to stop the port-forward when you're done."
echo "------------------------------------------------------"

# Forward port 8501 to local machine
kubectl port-forward svc/pollen-gui-svc 8501:8501 -n $NAMESPACE
