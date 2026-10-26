#!/usr/bin/env bash
# apply-all.sh — Apply all K8s manifests in dependency order
set -euo pipefail

cd "$(dirname "$0")/../../infra/k8s"

for svc in postgres-source postgres-platform seaweedfs redpanda polaris mlflow; do
  manifest="$svc/manifest.yaml"
  if [[ -f "$manifest" ]]; then
    echo "==> $svc"
    kubectl apply -f "$manifest" 2>&1 | sed 's/^/  /'
    echo
  else
    echo "!! skipping $svc (no $manifest)"
  fi
done

echo "Done. Check status with: kubectl get pods -o wide"
