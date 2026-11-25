#!/usr/bin/env bash
# Deploy or update the Pollen Viability Labeling App on Kubernetes (stenc-ns)
# Live URL: https://viability-label-stenc.dyn.cloud.e-infra.cz
set -euo pipefail
cd "$(dirname "$0")"

if [ -f "./kubeconfig.yaml" ]; then
  export KUBECONFIG="$(pwd)/kubeconfig.yaml"
fi
KUBECTL_BIN="kubectl"
if [ -x "./kubectl" ]; then
  KUBECTL_BIN="./kubectl"
fi
K="$KUBECTL_BIN -n stenc-ns"

echo "📦 Creating/updating ConfigMap viability-label-src..."
$K create configmap viability-label-src \
  --from-file=app.py=src/label_app/app.py \
  --from-file=index.html=src/label_app/index.html \
  --from-file=common.py=src/label_app/common.py \
  --from-file=project.yaml=src/label_app/project.yaml \
  --dry-run=client -o yaml | $K apply -f -

echo "🚀 Applying Kubernetes deployment & ingress..."
$K apply -f k8s/viability-label-app.yaml

echo "🔄 Restarting rollout..."
$K rollout restart deployment/viability-label

echo "⏳ Waiting for rollout status..."
$K rollout status deployment/viability-label --timeout=180s || true

echo "✅ Pollen Viability Labeling app is live at: https://viability-label-stenc.dyn.cloud.e-infra.cz"
