#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/ut_config.sh"

if [ $# -ne 1 ]; then
  echo "Usage: $0 <docker_image>"
  exit 1
fi

IMAGE_NAME="$1"

echo "[INFO] Starting containers: ${CONTAINER_NAMES[*]}"
echo "[INFO] Image: ${IMAGE_NAME}"
echo "[INFO] shm-size: ${DOCKER_SHM_SIZE}"

for CONTAINER_NAME in "${CONTAINER_NAMES[@]}"; do
  ASCEND_DEVICES="${CONTAINER_DEVICES[$CONTAINER_NAME]:-}"
  if [[ -z "${ASCEND_DEVICES}" ]]; then
    echo "[ERROR] Missing devices mapping for ${CONTAINER_NAME} in ut_config.sh"
    exit 1
  fi

  echo "[INFO] Starting ${CONTAINER_NAME}"
  echo "       ASCEND_RT_VISIBLE_DEVICES=${ASCEND_DEVICES}"

  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "[WARN] Container ${CONTAINER_NAME} already exists. Removing..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  fi

  docker run -it -d --shm-size="${DOCKER_SHM_SIZE}" \
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