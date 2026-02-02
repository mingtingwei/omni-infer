#!/bin/bash
set -e

# 1. 卸载 KubeInfer Helm 组件
if command -v helm &> /dev/null; then
  echo "Uninstalling KubeInfer components..."
  helm uninstall modelarts-infers-operator || echo "Helm release not found, skip."
else
  echo "Helm not found, skip Helm uninstall."
fi

# 2. 清理 containerd 镜像（只删 modelarts 相关）
echo "Removing modelarts images from containerd..."

IMAGES=$(ctr -n=k8s.io images ls | awk '{print $1}' | grep modelarts || true)

if [[ -n "$IMAGES" ]]; then
  for img in $IMAGES; do
    echo "Removing image: $img"
    ctr -n=k8s.io images rm "$img" || true
  done
else
  echo "No modelarts images found, skip."
fi

# 3. 检查 operator 是否还存在
echo "Checking remaining operator pods..."
kubectl get po -A | grep operator || echo "No operator pods found."

# 4. 删除 Helm 本体（可选）
if [[ -f /usr/local/bin/helm ]]; then
  echo "Removing Helm binary..."
  rm -f /usr/local/bin/helm
else
  echo "Helm binary not found, skip."
fi

# 5. 清理 Helm 下载残留（不依赖版本/架构）
if [[ -d /usr/local/src ]]; then
  echo "Cleaning Helm download cache in /usr/local/src ..."
  rm -f /usr/local/src/helm-*-linux-*.tar.gz || true
  rm -rf /usr/local/src/linux-* || true
else
  echo "/usr/local/src not found, skip Helm cache cleanup."
fi

echo "Uninstallation complete!"