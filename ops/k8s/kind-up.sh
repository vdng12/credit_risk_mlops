#!/usr/bin/env bash
# kind-up.sh — Bootstrap the full local K8s environment
set -euo pipefail

echo "==> Creating kind cluster (risk-cluster)..."
kind create cluster --config "$(dirname "$0")/../../infra/kind.yaml" 2>&1 | sed 's/^/  /'

echo
echo "==> Applying all manifests..."
bash "$(dirname "$0")/apply-all.sh"

echo
echo "==> Cluster ready!"
echo "  kubectl cluster-info"
echo "  kubectl get pods -o wide"
echo
echo "  Port mappings:"
echo "    Postgres Source    localhost:5432"
echo "    Postgres Platform  localhost:5433"
echo "    SeaweedFS S3       localhost:8333"
echo "    SeaweedFS Filer    localhost:8888"
echo "    SeaweedFS Master   localhost:9333"
echo "    Redpanda           localhost:9092"
echo "    Polaris REST       localhost:8181"
echo "    Polaris Mgmt       localhost:8182"
