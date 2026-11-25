#!/usr/bin/env bash
# Deploy / update the Pollen Viability QC review app (separate from Pollen_latitude's pollen-review).
#   bash deploy_viability_review.sh                         # update code, keep existing passwords
#   bash deploy_viability_review.sh --reset-passwords jakub eliska [more users…]
# Passwords live in review_app/.credentials (git-ignored, chmod 600). The cluster only gets
# PBKDF2-SHA256 hashes in the secret 'viability-review-users'; the app checks them itself.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f "./kubeconfig.yaml" ]; then
  export KUBECONFIG="$(pwd)/kubeconfig.yaml"
fi
KUBECTL_BIN="kubectl"
if [ -x "./kubectl" ]; then
  KUBECTL_BIN="./kubectl"
fi
NS=stenc-ns
CRED=review_app/.credentials

if [[ "${1:-}" == "--reset-passwords" ]]; then
  shift; users=("$@"); [[ ${#users[@]} -eq 0 ]] && users=(jakub eliska)
  : > "$CRED"; chmod 600 "$CRED"
  for u in "${users[@]}"; do echo "$u  $(openssl rand -base64 18 | tr -dc 'A-Za-z0-9' | head -c 14)" >> "$CRED"; done
  echo "🔑 New credentials:"; cat "$CRED"
elif [[ ! -s "$CRED" ]]; then
  echo "❌ $CRED missing — run with --reset-passwords jakub eliska"; exit 1
fi

# hash passwords → users.json secret
python3 - "$CRED" > /tmp/viability_users.json <<'PY'
import hashlib, json, os, sys
users = {}
for line in open(sys.argv[1]):
    if line.strip():
        u, pw = line.split()
        salt = os.urandom(16); it = 200_000
        users[u.lower()] = f"pbkdf2${it}${salt.hex()}${hashlib.pbkdf2_hmac('sha256', pw.encode(), salt, it).hex()}"
print(json.dumps(users))
PY
$KUBECTL_BIN create secret generic viability-review-users -n $NS --from-file=users.json=/tmp/viability_users.json \
  --dry-run=client -o yaml | $KUBECTL_BIN apply -f -
rm -f /tmp/viability_users.json
$KUBECTL_BIN delete secret viability-review-auth -n $NS --ignore-not-found >/dev/null

$KUBECTL_BIN create configmap viability-review-src -n $NS \
  --from-file=app.py=review_app/app.py --from-file=index.html=review_app/index.html \
  --dry-run=client -o yaml | $KUBECTL_BIN apply -f -
$KUBECTL_BIN apply -f k8s/viability-review.yaml
$KUBECTL_BIN rollout restart deployment/viability-review -n $NS
$KUBECTL_BIN rollout status deployment/viability-review -n $NS --timeout=180s
echo "✅ https://viability-review-stenc.dyn.cloud.e-infra.cz"
