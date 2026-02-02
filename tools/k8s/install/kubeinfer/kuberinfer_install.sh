#!/bin/bash
set -e

# 记录你最开始执行脚本的目录（kubeinfer）
WORK_DIR=$(pwd)

# 定义 Helm 版本和架构
HELM_VERSION=v3.18.5
ARCH=$(uname -m)

# 判断架构类型
case "$ARCH" in
  aarch64|arm64)
    HELM_ARCH=arm64
    ;;
  x86_64)
    HELM_ARCH=amd64
    ;;
  *)
    echo "Unsupported arch: $ARCH"
    exit 1
    ;;
esac

# 安装 Helm
echo "Downloading Helm ${HELM_VERSION} for ${HELM_ARCH}..."
cd /usr/local/src

# 用 -k 忽略代理的证书问题
curl -kLO https://get.helm.sh/helm-${HELM_VERSION}-linux-${HELM_ARCH}.tar.gz

# 解压并移动到合适的目录
tar -zxvf helm-${HELM_VERSION}-linux-${HELM_ARCH}.tar.gz
mv linux-${HELM_ARCH}/helm /usr/local/bin/helm
chmod +x /usr/local/bin/helm

# 检查 Helm 安装版本
echo "Helm version installed:"
helm version

# 切回你最初的 kubeinfer 目录
cd "$WORK_DIR"

# 当前目录就是 kubeinfer
echo "Current directory: $(pwd)"

# 查找 tar 文件
TAR_FILE=modelarts-infers-operator-*-aarch64-*.tar

if ! ls $TAR_FILE >/dev/null 2>&1; then
  echo "ERROR: No tar file found: $TAR_FILE"
  exit 1
fi

# 加载镜像到每个节点
echo "Loading image to all nodes..."
ctr -n=k8s.io images import $TAR_FILE

# 查找 tgz 文件
TGZ_FILE=modelarts-infers-operator-*-*.tgz

if ! ls $TGZ_FILE >/dev/null 2>&1; then
  echo "ERROR: No tgz file found: $TGZ_FILE"
  exit 1
fi

# 安装 KubeInfer 组件
echo "Installing kubeinfer components..."
helm install modelarts-infers-operator $TGZ_FILE -f values.yaml

# 查看安装情况
echo "Checking installation status..."
sleep 5
kubectl get po | grep operator

# 提示用户查看 pod 状态
echo "If the status is not running, run the following command to get more details:"
echo "kubectl describe po <pod-name>"

# 完成
echo "Deployment complete!"