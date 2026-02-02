#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo " Volcano Cleanup"
echo "=========================================="

############################################
# 0 基础安全检查
############################################

# 0: 仅节点级清理（默认，安全）
# 1: 集群级彻底清理（危险，需显式指定）
DO_CLUSTER_CLEAN=${DO_CLUSTER_CLEAN:-0}

if [[ $EUID -ne 0 ]]; then
  echo "[FATAL] Please run this script as root user"
  exit 1
fi

if ! command -v kubectl &>/dev/null; then
  echo "[FATAL] kubectl is not found, cannot clean up Volcano"
  exit 1
fi

if ! kubectl get ns &>/dev/null; then
  echo "[FATAL] Current kubeconfig is unavailable, cannot access the cluster"
  exit 1
fi

echo "[OK] kubectl & kubeconfig are working properly"

if [ "$DO_CLUSTER_CLEAN" = "1" ]; then
  echo "[WARN] DO_CLUSTER_CLEAN=1: Cluster-level cleanup will be performed"
else
  echo "[INFO] Node-level cleanup mode (default)"
fi

############################################
# 1 强制删除 Volcano 运行态 Pod
# - 仅在集群级清理时执行
############################################

if [ "$DO_CLUSTER_CLEAN" = "1" ]; then
  echo ">>> Step 1: Force deleting Volcano Pods..."

  kubectl delete pod -n volcano-system -l app=volcano-admission --force --grace-period=0 2>/dev/null || true
  kubectl delete pod -n volcano-system -l app=volcano-controllers --force --grace-period=0 2>/dev/null || true
  kubectl delete pod -n volcano-system -l app=volcano-scheduler --force --grace-period=0 2>/dev/null || true
  kubectl delete pod -n volcano-system --all --force --grace-period=0 2>/dev/null || true

  sleep 5
else
  echo ">>> Step 1: Skipped (node clean mode)"
fi

############################################
# 2 删除 init Job（仅集群级）
############################################

if [ "$DO_CLUSTER_CLEAN" = "1" ]; then
  echo ">>> Step 2: Deleting Volcano init Job..."

  kubectl delete job volcano-admission-init -n volcano-system 2>/dev/null || true
  kubectl delete pod -n volcano-system -l job-name=volcano-admission-init \
    --force --grace-period=0 2>/dev/null || true
else
  echo ">>> Step 2: Skipped (node clean mode)"
fi

############################################
# 3 删除 TLS Secret（仅集群级）
############################################

if [ "$DO_CLUSTER_CLEAN" = "1" ]; then
  echo ">>> Step 3: Deleting Volcano TLS secret..."

  kubectl delete secret volcano-admission-secret -n volcano-system 2>/dev/null || true
else
  echo ">>> Step 3: Skipped deleting TLS secret"
fi

############################################
# 4 删除 Volcano 工作负载资源（仅集群级）
############################################

if [ "$DO_CLUSTER_CLEAN" = "1" ]; then
  echo ">>> Step 4: Deleting Volcano workloads..."

  kubectl delete deployment -n volcano-system --all 2>/dev/null || true
  kubectl delete daemonset -n volcano-system --all 2>/dev/null || true
  kubectl delete statefulset -n volcano-system --all 2>/dev/null || true
  kubectl delete service -n volcano-system --all 2>/dev/null || true
else
  echo ">>> Step 4: Skipped (node clean mode)"
fi

############################################
# 5 删除 Webhook（仅集群级）
############################################

if [ "$DO_CLUSTER_CLEAN" = "1" ]; then
  echo ">>> Step 5: Removing Volcano Webhook configurations..."

  kubectl get validatingwebhookconfigurations 2>/dev/null | grep volcano | awk '{print $1}' \
    | xargs -r kubectl delete validatingwebhookconfiguration || true

  kubectl get mutatingwebhookconfigurations 2>/dev/null | grep volcano | awk '{print $1}' \
    | xargs -r kubectl delete mutatingwebhookconfiguration || true
else
  echo ">>> Step 5: Skipped (node clean mode)"
fi

############################################
# 6 删除 Volcano CRD（仅集群级）
############################################

if [ "$DO_CLUSTER_CLEAN" = "1" ]; then
  echo ">>> Step 6: Deleting Volcano CRDs..."

  kubectl get crd 2>/dev/null | grep volcano | awk '{print $1}' \
    | xargs -r kubectl delete crd || true
else
  echo ">>> Step 6: Skipped (node clean mode)"
fi

############################################
# 7 删除 Volcano Namespace（仅集群级）
############################################

if [ "$DO_CLUSTER_CLEAN" = "1" ]; then
  echo ">>> Step 7: Deleting Volcano namespace..."

  kubectl delete namespace volcano-system 2>/dev/null || true
  sleep 5
else
  echo ">>> Step 7: Skipped (node clean mode)"
fi

############################################
# 8 强制解除 namespace finalizer（仅集群级）
############################################

if [ "$DO_CLUSTER_CLEAN" = "1" ]; then
  if kubectl get namespace volcano-system &>/dev/null; then
    STATUS=$(kubectl get ns volcano-system -o jsonpath='{.status.phase}')
    if [[ "$STATUS" == "Terminating" ]]; then
      echo ">>> Step 8: Removing namespace finalizer..."

      kubectl patch namespace volcano-system \
        -p '{"spec":{"finalizers":[]}}' \
        --type=merge 2>/dev/null || true
    fi
  fi
else
  echo ">>> Step 8: Skipped (node clean mode)"
fi

############################################
# 9 删除本地 TLS 文件（节点级，安全）
############################################

echo ">>> Step 9: Removing local TLS cert files..."

rm -f "$SCRIPT_DIR/volcano-tls.key" "$SCRIPT_DIR/volcano-tls.crt" 2>/dev/null || true

############################################
# 10 本地安装目录处理（保留）
############################################

echo ">>> Step 10: Keeping local Volcano files (offline assets preserved)"

############################################
# 11 删除本地 containerd 中的 Volcano 镜像
############################################

echo ">>> Step 11: Removing local Volcano images from containerd..."

ctr -n=k8s.io images ls 2>/dev/null | grep volcano | awk '{print $1}' \
  | xargs -r ctr -n=k8s.io images rm || true

############################################
# 最终验证（仅集群级给提示）
############################################

echo ">>> Final verification..."

if [ "$DO_CLUSTER_CLEAN" = "1" ]; then
  kubectl get ns | grep -w volcano-system \
    && echo "[WARN] volcano namespace still exists" \
    || echo "[OK] volcano namespace removed"

  kubectl get crd | grep volcano \
    && echo "[WARN] volcano CRD still exists" \
    || echo "[OK] volcano CRDs removed"
else
  echo "[INFO] Node clean completed (cluster untouched)"
fi

echo "=========================================="
echo " Volcano Cleanup COMPLETED "
echo "=========================================="