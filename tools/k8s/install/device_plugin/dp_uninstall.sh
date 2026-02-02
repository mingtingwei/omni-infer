#!/bin/bash
set -e

echo "=============================================="
echo " Device Plugin Cleanup "
echo "=============================================="

############################################
# 0 基础安全检查 & 模式选择
############################################

# 0: 节点级清理（安全，不动集群态）
# 1: 集群级彻底清理（危险，需明确）
DO_CLUSTER_CLEAN=${DO_CLUSTER_CLEAN:-1}

if [[ $EUID -ne 0 ]]; then
  echo "[FATAL] Please run this script as root user"
  exit 1
fi

if ! command -v kubectl &>/dev/null; then
  echo "[FATAL] kubectl is not found, cannot clean up Device Plugin"
  exit 1
fi

if ! command -v helm &>/dev/null; then
  echo "[FATAL] helm is not found, cannot clean up Device Plugin"
  exit 1
fi

if ! kubectl get ns &>/dev/null; then
  echo "[FATAL] Current kubeconfig is unavailable, cannot access the cluster"
  exit 1
fi

echo "[OK] kubectl & helm & kubeconfig are working properly"

if [[ "$DO_CLUSTER_CLEAN" == "1" ]]; then
  echo "[WARN] DO_CLUSTER_CLEAN=1: Performing [cluster-level] Device Plugin cleanup"
else
  echo "[INFO] DO_CLUSTER_CLEAN=0: Performing [node-level] cleanup (safe mode)"
fi

############################################
# 1 卸载 Helm Release（仅集群级）
############################################

if [[ "$DO_CLUSTER_CLEAN" == "1" ]]; then
  echo ">>> Step 1: Uninstalling Helm release madp..."
  helm uninstall madp -n kube-system 2>/dev/null || true
  sleep 3
else
  echo ">>> Step 1: Skipped (node clean mode)"
fi

############################################
# 2 强制删除 Device Plugin Pod（仅集群级）
############################################

if [[ "$DO_CLUSTER_CLEAN" == "1" ]]; then
  echo ">>> Step 2: Force deleting Device Plugin Pods..."

  kubectl delete pod -n kube-system -l app=modelarts-device-plugin \
    --force --grace-period=0 2>/dev/null || true

  kubectl get pods -n kube-system 2>/dev/null | grep -i device-plugin \
    | awk '{print $1}' \
    | xargs -r kubectl delete pod -n kube-system --force --grace-period=0 || true

  sleep 3
else
  echo ">>> Step 2: Skipped (node clean mode)"
fi

############################################
# 3 删除 Device Plugin 工作负载（仅集群级）
############################################

if [[ "$DO_CLUSTER_CLEAN" == "1" ]]; then
  echo ">>> Step 3: Deleting Device Plugin workloads..."

  kubectl delete daemonset  -n kube-system -l app=modelarts-device-plugin 2>/dev/null || true
  kubectl delete deployment -n kube-system -l app=modelarts-device-plugin 2>/dev/null || true
else
  echo ">>> Step 3: Skipped (node clean mode)"
fi

############################################
# 4 删除 Service / ConfigMap / Secret（仅集群级）
############################################

if [[ "$DO_CLUSTER_CLEAN" == "1" ]]; then
  echo ">>> Step 4: Deleting Device Plugin Services / ConfigMaps / Secrets..."

  kubectl delete svc    -n kube-system -l app=modelarts-device-plugin 2>/dev/null || true
  kubectl delete cm     -n kube-system -l app=modelarts-device-plugin 2>/dev/null || true
  kubectl delete secret -n kube-system -l app=modelarts-device-plugin 2>/dev/null || true

  kubectl get secret -n kube-system -l app=modelarts-device-plugin -o name 2>/dev/null \
    | xargs -r kubectl delete -n kube-system --force --grace-period=0 || true
else
  echo ">>> Step 4: Skipped (node clean mode)"
fi

############################################
# 4.1 清理 Device Plugin Webhook（仅集群级）
############################################

if [[ "$DO_CLUSTER_CLEAN" == "1" ]]; then
  echo ">>> Step 4.1: Removing Device Plugin Webhooks..."

  kubectl get mutatingwebhookconfigurations 2>/dev/null | grep device-plugin | awk '{print $1}' \
    | xargs -r kubectl delete mutatingwebhookconfiguration || true

  kubectl get validatingwebhookconfigurations 2>/dev/null | grep device-plugin | awk '{print $1}' \
    | xargs -r kubectl delete validatingwebhookconfiguration || true
else
  echo ">>> Step 4.1: Skipped (node clean mode)"
fi

############################################
# 5 清理 Node Label（仅集群级）
############################################

if [[ "$DO_CLUSTER_CLEAN" == "1" ]]; then
  echo ">>> Step 5: Removing NPU node labels..."

  NPU_NODES=$(kubectl get nodes --show-labels \
    | grep accelerator/huawei-npu \
    | awk '{print $1}')

  if [[ -z "$NPU_NODES" ]]; then
    echo "[OK] No nodes with accelerator/huawei-npu label found"
  else
    for node in $NPU_NODES; do
      echo ">>> Removing label from node: $node"
      kubectl label nodes "$node" accelerator/huawei-npu- || true
    done
  fi
else
  echo ">>> Step 5: Skipped (node clean mode)"
fi

############################################
# 6 本地 Device Plugin 工作目录（明确不删除）
############################################

echo ">>> Step 6: Keeping local Device Plugin directory SAFE"

############################################
# 7 删除 containerd 中的 Device Plugin 镜像（节点级，安全）
############################################

echo ">>> Step 7: Removing Device Plugin images from containerd (node-level)..."

if ! systemctl is-active --quiet containerd; then
  echo "[WARN] containerd is not running, skipping image cleanup"
else
  ctr -n=k8s.io images ls 2>/dev/null | grep modelarts-device-plugin \
    | awk '{print $1}' \
    | xargs -r ctr -n=k8s.io images rm || true
fi

############################################
# 8 最终验证
############################################

echo ">>> Final verification..."

if [[ "$DO_CLUSTER_CLEAN" == "1" ]]; then
  helm ls -n kube-system | grep madp \
    && echo "[WARN] Helm release madp still exists" \
    || echo "[OK] Helm release madp removed"

  kubectl get pods -n kube-system | grep -i device-plugin \
    && echo "[WARN] Device Plugin Pod still exists" \
    || echo "[OK] Device Plugin Pods removed"

  kubectl get validatingwebhookconfigurations | grep device-plugin \
    && echo "[WARN] Device Plugin validating webhook still exists" \
    || echo "[OK] Device Plugin validating webhook removed"
else
  echo "[INFO] Node clean completed (cluster untouched)"
fi

echo "=============================================="
echo " Device Plugin Cleanup COMPLETED (STABLE)     "
echo "=============================================="
