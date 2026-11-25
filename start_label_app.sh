#!/usr/bin/env bash
# Start the Pollen Viability Labeling App locally
set -euo pipefail
cd "$(dirname "$0")"

# Activate virtualenv if present
if [[ -d ".venv" ]]; then
  source .venv/bin/activate
elif [[ -d "venv" ]]; then
  source venv/bin/activate
fi

# Load .env if present
if [[ -f ".env" ]]; then
  set -a
  source .env 2>/dev/null || true
  set +a
fi

export PORT="${PORT:-8050}"
export HOST="${HOST:-127.0.0.1}"
export SAM_WEIGHTS="${SAM_WEIGHTS:-sam_b.pt}"

echo "=================================================="
echo "🌸 Starting Pollen Viability Labeling App (SAM & Polygons)"
echo "   URL: http://${HOST}:${PORT}"
echo "   SAM Weights: ${SAM_WEIGHTS}"
echo "=================================================="

exec python3 src/label_app/app.py
