#!/bin/bash
set -e

# 脚本真实目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 真正的原始用户（支持 sudo）
REAL_USER=${SUDO_USER:-$USER}
REAL_HOME=$(eval echo "~$REAL_USER")

### 0. 必须用 root 权限跑
if [[ $EUID -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

### 一、部署前系统检查
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

fail() {
  echo -e "${RED}[FAIL] $1${NC}"
  exit 1
}

ok() {
  echo -e "${GREEN}[OK] $1${NC}"
}

warn() {
  echo -e "${YELLOW}[WARN] $1${NC}"
}

echo "  Kubernetes Preflight Check (v2.1 Final) "

# CPU & 内存
echo ">>> Checking CPU & Memory..."
lscpu | egrep 'Model name|CPU\(s\)'
free -h
ok "CPU & Memory detected"

# 磁盘空间检查（>=15% 或 >=20G）
echo ">>> Checking Disk Space..."

check_disk() {
  mount_point=$1

  if ! df -h "$mount_point" &>/dev/null; then
    warn "Mount point $mount_point not exists, skipped"
    return
  fi

  usage=$(df -h "$mount_point" | awk 'NR==2 {print $5}' | sed 's/%//')
  avail=$(df -BG "$mount_point" | awk 'NR==2 {print $4}' | sed 's/G//')

  if [[ "$usage" -ge 85 && "$avail" -lt 20 ]]; then
    fail "Disk insufficient on $mount_point (Used: ${usage}%, Free: ${avail}G)"
  else
    ok "Disk OK on $mount_point (Used: ${usage}%, Free: ${avail}G)"
  fi
}

check_disk /
check_disk /var
check_disk /run
check_disk /var/lib/containerd
check_disk /var/lib/kubelet

# 自动补全 /etc/hosts
echo ">>> Checking /etc/hosts..."

NODE_IP=$(ip route get 1 | awk '{print $7}')
NODE_NAME=$(hostname)

if ! grep -q "$NODE_IP" /etc/hosts; then
  echo "$NODE_IP    $NODE_NAME" >> /etc/hosts
  ok "/etc/hosts auto updated: $NODE_IP $NODE_NAME"
else
  ok "/etc/hosts already contains node mapping"
fi

# OS 信息
echo ">>> OS Information:"
cat /etc/os-release | head -5
ok "OS detected"

# MAC & product_uuid 校验
echo ">>> Checking MAC & Product UUID..."

ip link | grep link/ether || warn "No MAC address detected by ip link"


UUID=$(cat /sys/class/dmi/id/product_uuid 2>/dev/null)

if [[ -z "$UUID" ]]; then
  fail "product_uuid is empty"
else
  ok "product_uuid detected: $UUID"
fi

# Swap 检查（必须关闭）
echo ">>> Checking Swap..."

SWAP_STATUS=$(swapon --show)

if [[ -n "$SWAP_STATUS" ]]; then
  fail "Swap is ENABLED. Please disable using: swapoff -a"
else
  ok "Swap is disabled"
fi

# containerd 状态检查
echo ">>> Checking containerd..."

if ! systemctl is-active --quiet containerd; then
  ok "containerd is NOT running"
else
  ok "containerd is running"
fi

# kube-apiserver 端口 6443 占用检查（必须为空）
echo ">>> Checking kube-apiserver port 6443..."

PORT_CHECK=$(ss -lntp | grep 6443 || echo "no-listen")

if [[ "$PORT_CHECK" != "no-listen" ]]; then
  echo "$PORT_CHECK"
  fail "Port 6443 is already in use. Please clean up old kube-apiserver or related process!"
else
  ok "Port 6443 is free (no-listen)"
fi

# 最终放行
echo -e "${GREEN} ALL PREFLIGHT CHECKS PASSED — SAFE TO DEPLOY K8S ${NC}"

### 二、关闭 SELinux 并调整内核参数
setenforce 0 2>/dev/null || true
sed -i 's/^SELINUX=enforcing/SELINUX=permissive/' /etc/selinux/config || true

cat <<EOF | tee /etc/modules-load.d/containerd.conf
overlay
br_netfilter
EOF
modprobe overlay || true
modprobe br_netfilter || true

cat <<EOF | tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF
sysctl --system

### 三、安装 containerd v1.7.5 并配置 systemd cgroup driver + 代理

CONTAINERD_VERSION=1.7.5
ARCH=$(uname -m)
case "$ARCH" in
  x86_64)
    ARCH_MAP="amd64"
    ;;
  aarch64|arm64)
    ARCH_MAP="arm64"
    ;;
  *)
    echo "Unsupported ARCH: $ARCH"
    exit 1
    ;;
esac

mkdir -p /usr/local/src
cd /usr/local/src

# 如本地已有对应 tar.gz，则复用；否则从 GitHub 下载（-k 忽略证书问题）
if [[ ! -f containerd-${CONTAINERD_VERSION}-linux-${ARCH_MAP}.tar.gz ]]; then
  curl -kLO https://github.com/containerd/containerd/releases/download/v${CONTAINERD_VERSION}/containerd-${CONTAINERD_VERSION}-linux-${ARCH_MAP}.tar.gz
fi

mkdir -p /usr/local/bin
tar -C /usr/local -xzf containerd-${CONTAINERD_VERSION}-linux-${ARCH_MAP}.tar.gz

mkdir -p /etc/containerd

# 只在第一次不存在时生成默认配置，避免覆盖你已经调好的 config.toml
if [[ ! -f /etc/containerd/config.toml ]]; then
  /usr/local/bin/containerd config default > /etc/containerd/config.toml
fi

# 启用 systemd cgroup（如果字段存在就替换；如果不存在也不会报错）
sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml || true

# 把 pause 镜像改成 3.10（默认是 3.8）
sed -i 's#sandbox_image = ".*"#sandbox_image = "registry.k8s.io/pause:3.10"#' /etc/containerd/config.toml || true

# 为常见 registry 配置 TLS 免校验（解决代理 MITM 导致的 x509 unknown authority）
REGISTRIES=("registry.k8s.io" "registry-1.docker.io" "quay.io" "cdn01.quay.io")
for reg in "${REGISTRIES[@]}"; do
  if ! grep -q "registry.configs.\"$reg\"" /etc/containerd/config.toml; then
    cat <<EOF >> /etc/containerd/config.toml

[plugins."io.containerd.grpc.v1.cri".registry.configs."$reg".tls]
  insecure_skip_verify = true
EOF
  fi
done

# 不再启用 config_path，避免与 configs.tls 冲突（否则 CRI 插件会报 invalid plugin config）
# 保留 /etc/containerd/certs.d 仅作备用，不通过 config_path 生效
mkdir -p /etc/containerd/certs.d/registry.k8s.io
cat >/etc/containerd/certs.d/registry.k8s.io/hosts.toml <<'EOF'
server = "https://registry.k8s.io"

[host."https://registry.k8s.io"]
  capabilities = ["pull", "resolve"]
  skip_verify = true
EOF

### 四、配置 containerd systemd 服务 + 代理（统一从这里进代理）

cat <<EOF | tee /etc/systemd/system/containerd.service
[Unit]
Description=containerd container runtime
Documentation=https://containerd.io
After=network.target local-fs.target

[Service]
ExecStartPre=-/sbin/modprobe overlay
ExecStart=/usr/local/bin/containerd
Type=simple
Delegate=yes
KillMode=process
Restart=always
RestartSec=5
TimeoutStartSec=0
LimitNPROC=infinity
LimitCORE=infinity
LimitNOFILE=65535
TasksMax=infinity
OOMScoreAdjust=-999

[Install]
WantedBy=multi-user.target
EOF

# 写 systemd drop-in 给 containerd 配 proxy + NO_PROXY
mkdir -p /etc/systemd/system/containerd.service.d
cat >/etc/systemd/system/containerd.service.d/http-proxy.conf <<EOF
[Service]
# 通过公司代理访问公网 registry （这里 % 不需要你手动双写）
Environment="HTTP_PROXY=$http_proxy"
Environment="HTTPS_PROXY=$https_proxy"

# 内网 / 集群地址不走代理
Environment="NO_PROXY=$NO_PROXY"
EOF

systemctl daemon-reload
systemctl enable containerd
systemctl restart containerd 

# 启动 + 自检
if ! systemctl start containerd; then
  echo ">>> containerd 启动失败，最近 50 行日志如下："
  journalctl -xeu containerd.service --no-pager | tail -n 50
  exit 1
fi

### 五、安装 Kubernetes 组件（1.31.x）之前，先确保 containernetworking-plugins 就绪

# 1. 如果还没装 containernetworking-plugins，手动从官方 openEuler 仓库下一个 rpm
if ! rpm -q containernetworking-plugins &>/dev/null; then
  echo ">>> containernetworking-plugins 未安装，尝试手动下载 rpm..."

  mkdir -p /usr/local/src
  cd /usr/local/src

  CNI_RPM="containernetworking-plugins-1.1.1-6.oe2203sp4.aarch64.rpm"
  CNI_URL="https://repo.openeuler.org/openEuler-22.03-LTS-SP4/everything/aarch64/Packages/${CNI_RPM}"

  if [[ ! -f ${CNI_RPM} ]]; then
    curl -kLO "${CNI_URL}" || {
      echo "!!! 手动下载 ${CNI_RPM} 失败，请检查网络或公司镜像。"
      exit 1
    }
  fi

  rpm -Uvh "${CNI_RPM}" || {
    echo "!!! 安装 ${CNI_RPM} 失败，请手动排查。"
    exit 1
  }
fi

# 2. 如果 containernetworking-plugins 已安装，且 CNI 在 /usr/libexec/cni，但 /opt/cni/bin 不存在，则创建软链接
if rpm -q containernetworking-plugins &>/dev/null; then
  if [[ -d /usr/libexec/cni && ! -e /opt/cni/bin ]]; then
    mkdir -p /opt/cni
    ln -s /usr/libexec/cni /opt/cni/bin
  fi
fi

### 然后再写 yum 仓库    + 安装 kubelet/kubeadm/kubectl

cat <<EOF | tee /etc/yum.repos.d/kubernetes.repo
[kubernetes]
name=Kubernetes
baseurl=https://pkgs.k8s.io/core:/stable:/v1.31/rpm/
enabled=1
gpgcheck=1
gpgkey=https://pkgs.k8s.io/core:/stable:/v1.31/rpm/repomd.xml.key
exclude=kubelet kubeadm kubectl cri-tools kubernetes-cni
EOF

ps -ef | egrep "yum|dnf" | grep -v grep | awk '{print $2}' | xargs -r kill -9

yum install -y kubelet kubeadm kubectl cri-tools \
  --disableexcludes=kubernetes \
  --disablerepo=openEuler-everything \
  --setopt=sslverify=false \
  --nogpgcheck

### 六、给 kubelet 配 no-proxy，避免 CNI / Calico 访问 10.96.0.1 走代理

mkdir -p /etc/systemd/system/kubelet.service.d
cat >/etc/systemd/system/kubelet.service.d/10-no-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY="
Environment="HTTPS_PROXY="
Environment="http_proxy="
Environment="https_proxy="
Environment="NO_PROXY=$NO_PROXY"
Environment="no_proxy=$no_proxy"
EOF

systemctl daemon-reload
systemctl enable kubelet
systemctl restart kubelet || true

### 七、暴力清理旧的 Kubernetes 集群状态（保证从任意脏环境起步）

# 1. 停 kubelet，避免它不断拉起旧的 static pods
systemctl stop kubelet || true

# 2. 尝试 kubeadm reset（能清一部分状态，不成功也无所谓）
kubeadm reset --force || true

# 3. 杀掉所有可能残留的 control-plane 进程
pkill -f kube-apiserver || true
pkill -f kube-controller-manager || true
pkill -f kube-controller || true
pkill -f kube-scheduler || true
pkill -f etcd || true

# 4. 解除 /var/lib/kubelet/pods 下的挂载，否则 rm -rf 会提示 Device or resource busy
if mount | grep -q '/var/lib/kubelet/pods'; then
  echo "Umounting /var/lib/kubelet/pods mounts..."
  mount | grep '/var/lib/kubelet/pods' | awk '{print $3}' | sort -r | while read mp; do
    umount -f "$mp" || true
  done
fi

# 5. 清理 kubelet 本地状态和证书
rm -rf /var/lib/kubelet/*

# 6. 清理旧的配置、证书和 etcd 数据
rm -rf /etc/kubernetes/manifests/*
rm -rf /etc/kubernetes/pki
rm -rf /var/lib/etcd

# 7. 把所有 kubeconfig 也删了（包括 admin.conf / super-admin.conf 等）
rm -f /etc/kubernetes/*.conf

### 八、创建 kubeadm 初始化配置（注意修改 advertiseAddress）

# 自动获取当前机器的主 IP（从默认路由获取）
MASTER_IP=$(ip route get 1 | awk '{print $7; exit}')

KUBEADM_CFG="${SCRIPT_DIR}/kubeadm-cgroup.yaml"

cat <<EOF > "${KUBEADM_CFG}"
apiVersion: kubeadm.k8s.io/v1beta4
kind: InitConfiguration
localAPIEndpoint:
  advertiseAddress: ${MASTER_IP}
  bindPort: 6443
nodeRegistration:
  criSocket: "unix:///var/run/containerd/containerd.sock"
---
apiVersion: kubeadm.k8s.io/v1beta4
kind: ClusterConfiguration
kubernetesVersion: v1.31.14
networking:
  podSubnet: "192.168.0.0/16"
---
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
cgroupDriver: systemd
EOF

### 九、初始化 Kubernetes 控制平面（对已知 Crisocket 错误做兼容）

set +e
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u no_proxy -u NO_PROXY \
    kubeadm init --config "${KUBEADM_CFG}" 2>&1 | tee /tmp/kubeadm-init.log
INIT_RC=${PIPESTATUS[0]}
set -e

if [[ $INIT_RC -ne 0 ]]; then
  if grep -q "Error writing Crisocket information for the control-plane node: nodes" /tmp/kubeadm-init.log; then
    echo "kubeadm init 在 upload-config/kubelet 阶段写 criSocket 时遇到已知问题（Node 尚未注册），忽略并继续后续步骤。"
  else
    echo "kubeadm init 失败，日志在 /tmp/kubeadm-init.log，请检查。"
    exit $INIT_RC
  fi
fi

### 十、配置 kubectl 凭据
mkdir -p "${REAL_HOME}/.kube"
cp /etc/kubernetes/admin.conf "${REAL_HOME}/.kube/config"
chown "${REAL_USER}:${REAL_USER}" "${REAL_HOME}/.kube/config"

### 十一、安装 Calico 网络插件
cd "${SCRIPT_DIR}"
curl -kLO https://raw.githubusercontent.com/projectcalico/calico/v3.31.1/manifests/calico.yaml
kubectl apply -f calico.yaml

### 十二、（可选）单节点场景下，清理可能挡调度的 taint（包括 disk-pressure）
kubectl taint nodes --all node-role.kubernetes.io/control-plane-               || true
kubectl taint nodes --all node-role.kubernetes.io/master-                      || true
kubectl taint nodes --all node.kubernetes.io/disk-pressure:NoSchedule-         || true

### 十三、检查集群状态
kubectl get nodes
kubectl get pods -n kube-system -o wide

echo
echo "Kubernetes 集群部署完成！请检查上面节点和 kube-system Pod 状态。"
