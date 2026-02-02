#!/bin/bash
set -e

########################################
# K8s_uninstall.sh
# 一键卸载 / 清理 本机 Kubernetes (kubeadm + kubelet + etcd + 配置等)
# 默认不删除 containerd，如需一起删掉，把 REMOVE_CONTAINERD 改成 1
########################################

# 是否同时卸载 containerd（包括 /usr/local/bin/containerd*、/etc/containerd 等）
REMOVE_CONTAINERD=0
REAL_USER=${SUDO_USER:-$USER}
REAL_HOME=$(eval echo "~$REAL_USER")

### 0. 必须用 root 权限跑
if [[ $EUID -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

echo ">>> 停止 kubelet / containerd 等服务..."

systemctl stop kubelet 2>/dev/null || true
systemctl disable kubelet 2>/dev/null || true

# 如果你还有 cri-dockerd / docker，可以按需关掉（可选）
systemctl stop cri-docker 2>/dev/null || true
systemctl stop docker 2>/dev/null || true

systemctl stop containerd 2>/dev/null || true
systemctl disable containerd 2>/dev/null || true

echo ">>> kubeadm reset（如果安装过 kubeadm）..."
if command -v kubeadm &>/dev/null; then
  # kubeadm reset 本身可能失败，不影响后续清理
  kubeadm reset --force || true
fi

echo ">>> 杀掉残留的 control-plane 相关进程..."
pkill -f kube-apiserver          2>/dev/null || true
pkill -f kube-controller-manager 2>/dev/null || true
pkill -f kube-controller         2>/dev/null || true
pkill -f kube-scheduler          2>/dev/null || true
pkill -f etcd                    2>/dev/null || true

echo ">>> 解除 /var/lib/kubelet/pods 下的挂载..."
if mount | grep -q '/var/lib/kubelet/pods'; then
  mount | grep '/var/lib/kubelet/pods' | awk '{print $3}' | sort -r | while read mp; do
    umount -f "$mp" 2>/dev/null || true
  done
fi

echo ">>> 清理 kubelet / etcd / Kubernetes 配置目录..."
rm -rf /var/lib/kubelet/*
rm -rf /var/lib/etcd
rm -rf /etc/kubernetes
rm -rf /var/run/kubernetes 2>/dev/null || true

echo ">>> 清理 CNI 配置和二进制（如有）..."
rm -rf /etc/cni/net.d 2>/dev/null || true
rm -rf /opt/cni/bin   2>/dev/null || true
rm -rf /var/lib/cni   2>/dev/null || true

echo ">>> 安全检查：是否存在正在运行的 Docker 容器..."

DOCKER_RUNNING=0
DOCKER_HAS_CONTAINER=0

if command -v docker &>/dev/null && systemctl is-active --quiet docker; then
  DOCKER_RUNNING=1
  if docker ps --format '{{.ID}}' | grep -q .; then
    DOCKER_HAS_CONTAINER=1
  fi
fi

if [[ "$DOCKER_RUNNING" -eq 1 && "$DOCKER_HAS_CONTAINER" -eq 1 ]]; then
  echo "!!! 检测到正在运行的 Docker 业务容器"
  echo "!!! 为防止误伤 Docker 网络，已【跳过 iptables / IPVS 清理】"
else
  echo ">>> 未检测到 Docker 业务容器，安全清理 K8s 网络规则..."

  iptables -F
  iptables -t nat -F
  iptables -t mangle -F
  iptables -X

  if command -v ipvsadm &>/dev/null; then
    ipvsadm -C
  fi
fi

echo ">>> 清理当前用户的 kubeconfig (~/.kube)..."
rm -rf "${REAL_HOME}/.kube" 2>/dev/null || true

echo ">>> 卸载 kube 相关 rpm 包（如果存在）..."
if command -v yum &>/dev/null; then
  yum remove -y kubelet kubeadm kubectl kubernetes-cni cri-tools 2>/dev/null || true
fi

########################################
# 可选部分：卸载 containerd
########################################
if [[ "$REMOVE_CONTAINERD" -eq 1 ]]; then
  echo ">>> 按配置删除 containerd 及其配置文件..."

  systemctl stop containerd 2>/dev/null || true
  systemctl disable containerd 2>/dev/null || true

  rm -f /etc/systemd/system/containerd.service
  rm -rf /etc/systemd/system/containerd.service.d 2>/dev/null || true

  rm -rf /etc/containerd 2>/dev/null || true
  rm -rf /var/lib/containerd 2>/dev/null || true

  rm -f /usr/local/bin/containerd \
        /usr/local/bin/containerd-shim* \
        /usr/local/bin/ctr 2>/dev/null || true
fi

echo ">>> 重新加载 systemd 配置..."
systemctl daemon-reload

echo
echo "Kubernetes 已卸载 / 清理完成。"
if [[ "$REMOVE_CONTAINERD" -eq 1 ]]; then
  echo "注意：containerd 也已被卸载，如需再次部署请重新安装。"
else
  echo "注意：containerd 仍然保留，可直接用你的 K8s_Oneclick_Setup.sh 重新部署。"
fi
