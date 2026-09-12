#!/bin/bash
# watch_and_detect.sh
# ──────────────────────────────────────────────────────────────────
# Watches pollen-preprocess-job on K8s. Once it completes
# (Succeeded or Failed), automatically launches the detection job.
#
# Usage:
#   ./watch_and_detect.sh           # waits for preprocess, then detects
#   ./watch_and_detect.sh --force   # same but forces re-detection of all tiles
#
# Keep this running in a terminal or tmux session.
# ──────────────────────────────────────────────────────────────────
set -e

NAMESPACE="stenc-ns"
POLL_INTERVAL=30   # seconds between checks
FORCE_FLAG=""

if [[ "$1" == "--force" ]]; then
    FORCE_FLAG="--force"
    echo "⚠️  --force mode: detection will re-run all tiles."
fi

# ── kubectl setup ──────────────────────────────────────────────────
if ! command -v kubectl &>/dev/null; then
    KUBECTL="./kubectl"
else
    KUBECTL="kubectl"
fi

if [ -f "./kubeconfig.yaml" ]; then
    export KUBECONFIG="$(pwd)/kubeconfig.yaml"
fi

echo "──────────────────────────────────────────────────────────"
echo "🔭 Pollen Preprocessing Watcher"
echo "   Watching: pollen-preprocess-job in namespace $NAMESPACE"
echo "   Poll interval: ${POLL_INTERVAL}s"
echo "   Will auto-launch: ./deploy_detect.sh $FORCE_FLAG"
echo "──────────────────────────────────────────────────────────"
echo ""

# ── Wait loop ──────────────────────────────────────────────────────
iteration=0
while true; do
    STATUS=$($KUBECTL get job pollen-preprocess-job -n $NAMESPACE \
        -o jsonpath='{.status.conditions[?(@.type=="Complete")].status} {.status.conditions[?(@.type=="Failed")].status}' \
        2>/dev/null || echo "NotFound")

    COMPLETE=$(echo "$STATUS" | awk '{print $1}')
    FAILED=$(echo "$STATUS"   | awk '{print $2}')

    # Also get latest active/succeeded/failed counts for live display
    COUNTS=$($KUBECTL get job pollen-preprocess-job -n $NAMESPACE \
        -o jsonpath='active={.status.active} succeeded={.status.succeeded} failed={.status.failed}' \
        2>/dev/null || echo "unknown")

    NOW=$(date '+%H:%M:%S')
    iteration=$((iteration + 1))

    if [ "$COMPLETE" = "True" ]; then
        echo ""
        echo "[$NOW] ✅ pollen-preprocess-job SUCCEEDED!"
        echo "       Launching detection job now…"
        echo ""
        break
    elif [ "$FAILED" = "True" ]; then
        echo ""
        echo "[$NOW] ⚠️  pollen-preprocess-job FAILED (status=$COUNTS)"
        echo "       Launching detection anyway (tiles generated so far will be processed)…"
        echo ""
        break
    elif [ "$STATUS" = "NotFound" ]; then
        echo "[$NOW] ❓ Job not found — it may have already finished. Launching detection…"
        break
    else
        # Progress: print last 2 log lines for live feedback
        LAST_LOG=$($KUBECTL logs job/pollen-preprocess-job -n $NAMESPACE --tail=2 2>/dev/null \
            | tr '\n' ' ' | cut -c1-90 || echo "(log unavailable)")
        printf "[$NOW] ⏳ Still running ($COUNTS) | %s\n" "$LAST_LOG"
        sleep $POLL_INTERVAL
    fi
done

# ── Auto-launch detection ──────────────────────────────────────────
echo "🚀 Launching: ./deploy_detect.sh $FORCE_FLAG"
echo ""
bash ./deploy_detect.sh $FORCE_FLAG
