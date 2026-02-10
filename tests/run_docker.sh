#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <docker_image>"
  exit 1
fi

IMAGE_NAME="$1"

# 定义容器名称及对应的ASCEND设备信息
declare -A CONTAINERS=(
  ["DT_1"]="0,1"
  ["DT_2"]="2,3"
  ["DT_3"]="4,5"
  ["DT_4"]="6,7"
  ["DT_5"]="8,9,10,11"
  ["DT_6"]="12,13"
  ["DT_7"]="14,15"
)

echo "[INFO] Starting containers (using array and loop)"

for CONTAINER_NAME in "${!CONTAINERS[@]}"; do
  ASCEND_DEVICES="${CONTAINERS[$CONTAINER_NAME]}"
  
  echo "[INFO] Starting ${CONTAINER_NAME}"
  echo "       ASCEND_RT_VISIBLE_DEVICES=${ASCEND_DEVICES}"
  
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "[WARN] Container ${CONTAINER_NAME} already exists. Removing..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  fi

  docker run -it -d --shm-size=500g \
    -e PYTHONHASHSEED=123 \
    -e ASCEND_RT_VISIBLE_DEVICES="${ASCEND_DEVICES}" \
    -e http_proxy \
    -e https_proxy \
    --privileged=true \
    -u root \
    -w /home \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
    -v /data:/data/ \
    -v /tmp:/tmp \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -e http_proxy="${http_proxy:-}" \
    -e https_proxy="${https_proxy:-}" \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}"
done

echo "[INFO] All containers started."