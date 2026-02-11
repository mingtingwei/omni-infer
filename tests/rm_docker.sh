#!/usr/bin/env bash
set -euo pipefail

# 定义要删除的容器名称
CONTAINERS=(
  "DT_1"
  "DT_2"
  "DT_3"
  "DT_4"
  "DT_5"
  "DT_6"
  "DT_7"
)

echo "[INFO] Removing containers: ${CONTAINERS[*]}"

for CONTAINER_NAME in "${CONTAINERS[@]}"; do
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "[INFO] Removing container ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  else
    echo "[WARN] Container ${CONTAINER_NAME} does not exist. Skipping."
  fi
done

echo "[INFO] Done."