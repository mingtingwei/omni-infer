#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

############################################
# 0. 说明
# - 本脚本需要在每个节点各执行一次
# - 每个节点：修 containerd + 导入本地镜像
# - 只有最后一台节点（DO_RESTART=1）执行 Pod 重建
############################################

DO_RESTART=${DO_RESTART:-0}   # 0: 不重建 Pod（默认）  1: 重建 Pod

############################################
# 1. 准备命名空间 & 安装 Volcano YAML（幂等）
############################################

echo "Step 1: Preparing namespace & installing Volcano YAML..."

kubectl get namespace volcano-system &>/dev/null || kubectl create namespace volcano-system

# 从本地 tar 包自动推导 Volcano 版本
VOLCANO_VERSION=$(ls "$SCRIPT_DIR"/vc-*-v*.tar 2>/dev/null \
  | sed -n 's/.*-\(v[0-9]\+\.[0-9]\+\.[0-9]\+\)\.tar/\1/p' \
  | sort -u)

if [ -z "$VOLCANO_VERSION" ]; then
  echo "[FATAL] Cannot determine Volcano version from local tar files"
  exit 1
fi

if [ "$(echo "$VOLCANO_VERSION" | wc -l)" -ne 1 ]; then
  echo "[FATAL] Multiple Volcano versions found:"
  echo "$VOLCANO_VERSION"
  exit 1
fi

echo "Detected Volcano version: $VOLCANO_VERSION"

# 下载并修正 Volcano YAML（只在首次不存在时）
if [ ! -f volcano-development.yaml ]; then
  curl -kLO https://raw.githubusercontent.com/volcano-sh/volcano/release-1.12/installer/volcano-development.yaml

  # 避免每次都强制拉远端镜像
  sed -i 's/imagePullPolicy: Always/imagePullPolicy: IfNotPresent/g' volcano-development.yaml

  # 统一 volcanosh 镜像版本为本地 tar 对应版本
  sed -i "s|\(docker.io/volcanosh/[^:]*:\)v[0-9]\+\.[0-9]\+\.[0-9]\+|\1${VOLCANO_VERSION}|g" volcano-development.yaml
fi

# 打印最终使用的镜像，方便排查
echo "Final Volcano images to be applied:"
grep -n 'docker.io/volcanosh/' volcano-development.yaml

kubectl apply -f volcano-development.yaml
echo "Volcano YAML applied."

############################################
# 2. 修 containerd：docker.io TLS skip verify
############################################

echo "Step 2: Configuring containerd..."

CONFIG_FILE="/etc/containerd/config.toml"
UNIT_DIR="/etc/systemd/system/containerd.service.d"
UNIT_FILE="$UNIT_DIR/10-config.conf"

# 2.0 确保 config.toml 存在
if [ ! -f "$CONFIG_FILE" ]; then
  mkdir -p /etc/containerd
  containerd config default > "$CONFIG_FILE"
fi

# 2.1 确保 containerd 使用 --config 启动
if ! systemctl cat containerd | grep -q -- '--config'; then
  mkdir -p "$UNIT_DIR"
  cat >"$UNIT_FILE" <<EOF
[Service]
ExecStart=
ExecStart=/usr/local/bin/containerd --config $CONFIG_FILE
EOF
  systemctl daemon-reexec
  systemctl daemon-reload
fi

# 2.2 幂等处理 docker.io TLS 配置（关键修复点）
TLS_BLOCK='[plugins."io.containerd.grpc.v1.cri".registry.configs."docker.io".tls]'
TLS_COUNT=$(grep -cF "$TLS_BLOCK" "$CONFIG_FILE" || true)

if [ "$TLS_COUNT" -eq 0 ]; then
  echo "Adding docker.io TLS skip verify config"
  cat >>"$CONFIG_FILE" <<'EOF'

[plugins."io.containerd.grpc.v1.cri".registry.configs."docker.io".tls]
  insecure_skip_verify = true
EOF
elif [ "$TLS_COUNT" -eq 1 ]; then
  echo "docker.io TLS config already present, skip"
else
  echo "Multiple docker.io TLS configs found ($TLS_COUNT), cleaning up"
  sed -i '/registry\.configs\."docker\.io"\.tls\]/,/^\s*$/d' "$CONFIG_FILE"
  cat >>"$CONFIG_FILE" <<'EOF'

[plugins."io.containerd.grpc.v1.cri".registry.configs."docker.io".tls]
  insecure_skip_verify = true
EOF
fi

# 2.3 重启 containerd 并校验
systemctl restart containerd
sleep 2

systemctl is-active --quiet containerd || {
  journalctl -u containerd -n 80 --no-pager
  exit 1
}

echo "containerd configured."

############################################
# 3. 导入本地 Volcano 镜像（仅影响本节点）
############################################

echo "Step 3: Importing local Volcano images..."

ctr -n=k8s.io images import "$SCRIPT_DIR/vc-webhook-manager-v1.12.2.tar"
ctr -n=k8s.io images import "$SCRIPT_DIR/vc-scheduler-v1.12.2.tar"
ctr -n=k8s.io images import "$SCRIPT_DIR/vc-controller-manager-v1.12.2.tar"

echo "Local images imported."

############################################
# 3.1 直接拉取 Volcano 相关镜像
############################################
#ctr -n=k8s.io images pull docker.io/volcanosh/vc-webhook-manager:v1.12.2
#ctr -n=k8s.io images pull docker.io/volcanosh/vc-scheduler:v1.12.2
#ctr -n=k8s.io images pull docker.io/volcanosh/vc-controller-manager:v1.12.2

############################################
# 4. admission TLS Secret（只在不存在时创建）
############################################

echo "Step 4: Ensuring admission TLS secret..."

if kubectl get secret volcano-admission-secret -n volcano-system &>/dev/null; then
  echo "admission TLS secret already exists, skip creation."
else
  echo "Generating admission TLS cert with SAN..."

  openssl genrsa -out volcano-tls.key 2048

  cat > openssl-volcano.cnf <<'EOF'
[req]
prompt = no
distinguished_name = dn
x509_extensions = v3_req

[dn]
CN = volcano-admission-service.volcano-system.svc

[v3_req]
subjectAltName = @alt

[alt]
DNS.1 = volcano-admission-service
DNS.2 = volcano-admission-service.volcano-system
DNS.3 = volcano-admission-service.volcano-system.svc
DNS.4 = volcano-admission-service.volcano-system.svc.cluster.local
EOF

  openssl req -x509 -new -nodes \
    -key volcano-tls.key \
    -sha256 \
    -days 3650 \
    -out volcano-tls.crt \
    -config openssl-volcano.cnf \
    -extensions v3_req

  kubectl create secret generic volcano-admission-secret \
    --from-file=tls.crt=volcano-tls.crt \
    --from-file=tls.key=volcano-tls.key \
    --from-file=ca.crt=volcano-tls.crt \
    -n volcano-system \
    --dry-run=client -o yaml | kubectl apply -f -
fi

############################################
# 5. 仅在 DO_RESTART=1 时执行核心 Pod 重建
############################################

if [ "$DO_RESTART" != "1" ]; then
  echo "Step 5: Skip Pod restart (DO_RESTART=0)."
else
  echo "Step 5: Restarting Volcano core components..."

  kubectl delete job volcano-admission-init -n volcano-system --ignore-not-found

  kubectl delete pod -n volcano-system -l app=volcano-admission --force --grace-period=0
  kubectl delete pod -n volcano-system -l app=volcano-controllers --force --grace-period=0

  echo "Waiting for volcano-admission to be Ready..."
  kubectl wait --for=condition=Available deployment/volcano-admission \
    -n volcano-system \
    --timeout=180s

  kubectl delete pod -n volcano-system -l app=volcano-scheduler --force --grace-period=0

  echo "Volcano core Pods restarted."
fi

############################################
# 6. 最终状态检查
############################################

echo "Step 6: Final status check..."
kubectl get pods -n volcano-system

echo "Done."