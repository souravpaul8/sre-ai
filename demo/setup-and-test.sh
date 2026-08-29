#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/kind-config.yaml"
CLUSTER_NAME="kind" # Default kind cluster name

echo "=== 1. Checking kind cluster status ==="
if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    echo "Cluster '${CLUSTER_NAME}' already exists. Skipping cluster creation."
else
    echo "=== Provisioning kind cluster ==="
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Error: Configuration file '$CONFIG_PATH' not found."
        echo "Please place your 'kind-config.yaml' in the same folder as this script."
        exit 1
    fi
    echo "Using kind config: $CONFIG_PATH"
    kind create cluster --config "$CONFIG_PATH"
fi

echo "=== 2. Ensuring 'production' namespace exists ==="
kubectl get namespace production >/dev/null 2>&1 || kubectl create namespace production

echo "=== 3. Deploying pods to 'production' namespace ==="
kubectl run nginx --image=nginx --namespace=production 2>/dev/null || echo "Pod 'nginx' already exists."
kubectl run payment-service --image=nginx --namespace=production 2>/dev/null || echo "Pod 'payment-service' already exists."

echo "=== 4. Waiting for pods to reach 'Running' state ==="
kubectl wait --for=condition=Ready pod/nginx --namespace=production --timeout=60s
kubectl wait --for=condition=Ready pod/payment-service --namespace=production --timeout=60s

echo "=== 5. Running Python verification test ==="
PYTHONPATH=src python3 -c "
from llm_playground_cli.tools.k8s import get_pod_status_tool

result = get_pod_status_tool.execute(namespace='production')
print('Success:', result.success)
print('Data:', result.data)
"