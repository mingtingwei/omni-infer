#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

############################################
# 0. 说明
# - 默认：仅节点级准备（安全）
# - DO_RESTART=1：允许集群级 reconcile（慎用）
############################################

DO_RESTART=${DO_RESTART:-0}   # 0: 节点级（默认） 1: 集群级

############################################
# 1. 基础检查
############################################

if [[ $EUID -ne 0 ]]; then
  echo "[FATAL] Please run this script as root"
  exit 1
fi

if ! command -v kubectl &>/dev/null; then
  echo "[FATAL] kubectl is not installed"
  exit 1
fi

echo "[OK] basic check passed"

############################################
# 2. 给当前节点打 NPU 标签（幂等）
############################################

echo "Step 2: Labeling node..."

NODE_NAME=$(kubectl get nodes --no-headers | awk '{print $1}' | grep "$(hostname -s)" || true)

if [ -z "$NODE_NAME" ]; then
  echo "[FATAL] Failed to identify current node name"
  exit 1
fi

kubectl label node "$NODE_NAME" accelerator/huawei-npu=true --overwrite

############################################
# 3. containerd：确认正在使用 config.toml（只校验，不改 ExecStart）
############################################

echo "Step 3: Checking containerd config path..."

CONFIG_FILE="/etc/containerd/config.toml"

if [ ! -f "$CONFIG_FILE" ]; then
  mkdir -p /etc/containerd
  containerd config default > "$CONFIG_FILE"
fi

CTRD_PS="$(ps -ef | grep '[c]ontainerd')"
echo "[INFO] containerd process: $CTRD_PS"

if echo "$CTRD_PS" | grep -q -- '--config'; then
  RUNTIME_CFG=$(echo "$CTRD_PS" | sed -n 's/.*--config[[:space:]]\+\([^[:space:]]\+\).*/\1/p')
  if [ "$RUNTIME_CFG" != "$CONFIG_FILE" ]; then
    echo "[FATAL] containerd is using config file $RUNTIME_CFG instead of $CONFIG_FILE"
    exit 1
  fi
fi

############################################
# 4. containerd：registry-cbu TLS skip verify（安全幂等）
############################################

echo "Step 4: Configuring registry-cbu TLS skip verify..."

NEED_RESTART=0
TLS_HEADER='[plugins."io.containerd.grpc.v1.cri".registry.configs."registry-cbu.huawei.com".tls]'

if grep -qF "$TLS_HEADER" "$CONFIG_FILE"; then
  echo "[OK] registry-cbu TLS configuration already exists, skip modification"
else
  cat >>"$CONFIG_FILE" <<'EOF'

[plugins."io.containerd.grpc.v1.cri".registry.configs."registry-cbu.huawei.com".tls]
  insecure_skip_verify = true
EOF
  NEED_RESTART=1
fi

############################################
# 5. 重启 containerd（仅必要时）
############################################

if [ "$NEED_RESTART" = "1" ]; then
  echo "Restarting containerd..."
  systemctl restart containerd
  sleep 2
  systemctl is-active --quiet containerd || {
    journalctl -u containerd -n 80 --no-pager
    exit 1
  }
else
  echo "containerd already configured, skip restart."
fi

############################################
# 6. 集群级：Device Plugin（仅 DO_RESTART=1）
############################################

if [ "$DO_RESTART" != "1" ]; then
  echo "Step 6: Skip cluster-level install (DO_RESTART=0)"
else
  echo "Step 6: Applying Device Plugin (cluster-level)..."

  # 6.1 查找 Helm tgz 包（版本权威来源）
  TAR_FILE=$(ls "$SCRIPT_DIR"/modelarts-device-plugin-*.tgz 2>/dev/null | sort -u)

  if [ -z "$TAR_FILE" ]; then
    echo "[FATAL] Device Plugin Helm package not found"
    exit 1
  fi

  if [ "$(echo "$TAR_FILE" | wc -l)" -ne 1 ]; then
    echo "[FATAL] Multiple Device Plugin Helm packages found:"
    echo "$TAR_FILE"
    exit 1
  fi

  echo "Using Device Plugin Helm package: $TAR_FILE"

  # 6.2 从 tgz 文件名中推导镜像 tag（不关心具体版本号）
  DP_IMAGE_TAG=$(basename "$TAR_FILE" \
    | sed -n 's/modelarts-device-plugin-\(.*\)\.tgz/\1/p')

  if [ -z "$DP_IMAGE_TAG" ]; then
    echo "[FATAL] Failed to extract Device Plugin image tag from tgz name"
    exit 1
  fi

  IMAGE="registry-cbu.huawei.com/modelarts-rse/modelarts-device-plugin:${DP_IMAGE_TAG}"

  echo "Using Device Plugin image: $IMAGE"

  # 6.3 解压 Helm chart
  rm -rf "$SCRIPT_DIR/modelarts-device-plugin"
  tar -zxf "$TAR_FILE" -C "$SCRIPT_DIR"

  cd "$SCRIPT_DIR/modelarts-device-plugin"

  # 6.4 注入 image & community k8s 配置
  sed -i "s|image:.*|image: $IMAGE|" values.yaml
  sed -i "s|isCommunityK8s:.*|isCommunityK8s: true|" values.yaml

  # 6.5 Helm 安装 / 升级
  helm upgrade --install madp \
    "$SCRIPT_DIR/modelarts-device-plugin" \
    -n kube-system
fi

############################################
# 7. 状态检查
############################################

echo "Step 7: Verification..."

kubectl get nodes "$NODE_NAME" -o json | grep huawei.com/ascend || true
kubectl get pods -n kube-system | grep -i device-plugin || true

echo "Done."