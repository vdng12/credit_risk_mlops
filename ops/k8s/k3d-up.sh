#!/usr/bin/env bash
# k3d-up.sh — Bootstrap the full local K8s environment with k3d
set -euo pipefail

CLUSTER="risk-cluster"

if k3d cluster list 2>/dev/null | grep -q "$CLUSTER"; then
  echo "==> Cluster '$CLUSTER' already exists."
else
  echo "==> Creating k3d cluster ($CLUSTER)..."
  k3d cluster create "$CLUSTER" \
    --k3s-arg "--kube-apiserver-arg=service-node-port-range=1-65535@server:0" `# NodePort = host port` \
    -p "5432:5432"   `# postgres-source` \
    -p "5433:5433"   `# postgres-platform` \
    -p "8333:8333"   `# seaweedfs S3` \
    -p "8888:8888"   `# seaweedfs filer UI` \
    -p "9333:9333"   `# seaweedfs master UI` \
    -p "9092:9092"   `# redpanda` \
    -p "8181:8181"   `# polaris REST` \
    -p "8182:8182"   `# polaris mgmt` \
    -p "5000:5000"   `# mlflow UI` \
    -p "5050:5050"   `# pgAdmin (tools)` \
    -p "8080:8080"   `# redpanda console (tools)`
fi

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
echo "    MLflow             localhost:5000"