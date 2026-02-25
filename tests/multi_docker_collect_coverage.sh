# 先生成python覆盖率 再生成C覆盖率

#!/usr/bin/env bash
set -euo pipefail

containers=(DT_1 DT_2 DT_3 DT_4 DT_5 DT_6 DT_7)
MERGE_CONTAINER="${containers[0]}"



HOST_OUT="${1:?HOST_OUT is required}" # 宿主机 覆盖率报告路径
HOST_OMNI_DIR="${2:?HOST_OMNI_DIR is required}" # 宿主机 omniinfer路径
OMNI_IN_CONTAINER="${3:?OMNI_IN_CONTAINER is required}" # 容器内 omniinfer路径
MERGE_DIR_IN_CONTAINER="${4:-/workspace/MERGE_DIR_IN_CONTAINER}" # 容器内 合并覆盖率的临时路径




mkdir -p "${HOST_OUT}"
rm -rf "${HOST_OUT:?}/"*
rm -rf "${HOST_OUT:?}/".??* 2>/dev/null || true

# python覆盖率相关路径

COV_IN_CONTAINER="${OMNI_IN_CONTAINER}/tests/.coverage"
REPORTS_IN_CONTAINER="${OMNI_IN_CONTAINER}/tests/reports/coverage"
HOST_GIT_DIR="${HOST_OMNI_DIR}/.git"


# C proxy覆盖率相关路径
HOST_NGINX_DIR="${HOST_OUT}/nginx_from_dockers"
HOST_PROXY_REPORT_DIR="${HOST_OUT}/coverage/proxy_report"

# 容器内：各 DT_x 的 nginx build 目录（来源）
SRC_NGINX_IN_CONTAINER="${OMNI_IN_CONTAINER}/omni/accelerators/sched/nginx-1.28.0"

# 容器内：拷回去的目标 sched 目录（每个容器一样）
DST_SCHED_IN_CONTAINER="${OMNI_IN_CONTAINER}/coverage_check/omni/accelerators/sched"

# 在 DT_1 里跑 gcovr 的目录
DT1_RUN_DIR="${OMNI_IN_CONTAINER}/DT1_gcovr_path"

# gcovr 参数
GCOVR_ROOT="${OMNI_IN_CONTAINER}/omni/accelerators/sched/nginx-1.28.0"
GCOVR_OBJDIR="${OMNI_IN_CONTAINER}/coverage_check/omni/accelerators/sched"
GCOVR_FILTER="${OMNI_IN_CONTAINER}/omni/accelerators/sched/omni_proxy/.*\.c$"

mkdir -p "${HOST_NGINX_DIR}"
mkdir -p "${HOST_PROXY_REPORT_DIR}"

echo "HOST_NGINX_DIR  = ${HOST_NGINX_DIR}"
echo "HOST_PROXY_REPORT_DIR = ${HOST_PROXY_REPORT_DIR}"
echo

# python覆盖率生成过程
# echo "===== STEP 1.1: 从各容器拷 coverage 到宿主机（当前目录） ====="
# 清理旧文件，避免 combine 混进历史

cd "${HOST_OUT}"

rm -f "${HOST_OUT}"/.coverage.*

for c in "${containers[@]}"; do
  echo "[INFO] docker cp ${c}:${COV_IN_CONTAINER} -> ${HOST_OUT}.coverage.${c}"
  docker cp "${c}:${COV_IN_CONTAINER}" "${HOST_OUT}.coverage.${c}"
done

echo
# echo "===== STEP 1.2: 把各容器 coverage 文件送进汇总容器 ${MERGE_CONTAINER}:${MERGE_DIR_IN_CONTAINER} ====="
docker exec "${MERGE_CONTAINER}" /bin/bash -c "
  set -e
  mkdir -p '${MERGE_DIR_IN_CONTAINER}'
  rm -f '${MERGE_DIR_IN_CONTAINER}/.coverage.'*
  rm -f '${MERGE_DIR_IN_CONTAINER}/.coverage'
"

for c in "${containers[@]}"; do
  echo "[INFO] docker cp ${HOST_OUT}.coverage.${c} -> ${MERGE_CONTAINER}:${MERGE_DIR_IN_CONTAINER}/.coverage.${c}"
  docker cp "${HOST_OUT}.coverage.${c}" "${MERGE_CONTAINER}:${MERGE_DIR_IN_CONTAINER}/.coverage.${c}"
done

echo
# echo "===== STEP 1.3: 在 ${MERGE_CONTAINER} 里 coverage combine（生成 ${MERGE_DIR_IN_CONTAINER}/.coverage） ====="
docker exec "${MERGE_CONTAINER}" /bin/bash -c "
  set -e
  cd '${MERGE_DIR_IN_CONTAINER}'
  ls -la
  coverage combine
  ls -la
"

echo
# echo "===== STEP 1.4: 把合并后的 .coverage 覆盖回 omniinfer/tests/.coverage（用于 collect_coverage.sh） ====="
docker exec "${MERGE_CONTAINER}" /bin/bash -c "
  set -e
  cd '${OMNI_IN_CONTAINER}/tests'
  # 备份原来的（可选，但建议留着）
  if [ -f .coverage ]; then
    cp -f .coverage .coverage.bak.\$(date +%Y%m%d_%H%M%S) || true
  fi
  cp -f '${MERGE_DIR_IN_CONTAINER}/.coverage' ./.coverage
  ls -la .coverage*
"

echo
# echo "===== STEP 1.5: 确保 ${MERGE_CONTAINER} 里 ${OMNI_IN_CONTAINER}/.git 存在 ====="
echo "[INFO] docker cp host:${HOST_GIT_DIR} -> ${MERGE_CONTAINER}:${OMNI_IN_CONTAINER}/.git"
docker cp "${HOST_GIT_DIR}" "${MERGE_CONTAINER}:${OMNI_IN_CONTAINER}/.git"

echo
# echo "===== STEP 1.6: 在 ${MERGE_CONTAINER} 里生成 python 覆盖率报告 ====="
docker exec "${MERGE_CONTAINER}" /bin/bash -c "
  set -e
  cd '${OMNI_IN_CONTAINER}'
  COVERAGE_FILE='${OMNI_IN_CONTAINER}/tests/.coverage' bash tests/collect_coverage.sh '${OMNI_IN_CONTAINER}'
"

echo
# echo "===== STEP 1.7: 拷贝报告回宿主机 ${HOST_OUT} ====="
rm -rf "${HOST_OUT}/coverage"
docker cp "${MERGE_CONTAINER}:${REPORTS_IN_CONTAINER}" "${HOST_OUT}/coverage"

# C proxy覆盖率生成过程
# echo "===== STEP 2.1: docker cp (DT_1~DT_4 -> 宿主机) ====="
rm -rf "${HOST_NGINX_DIR}/nginx-1.28.0-"*

for c in "${containers[@]}"; do
  echo "[INFO] docker cp from ${c}"
  rm -rf "${HOST_NGINX_DIR}/nginx-1.28.0-${c}"
  docker cp \
    "${c}:${SRC_NGINX_IN_CONTAINER}" \
    "${HOST_NGINX_DIR}/nginx-1.28.0-${c}"
done

echo
# echo "===== STEP 2.2: docker cp (宿主机 -> ${MERGE_CONTAINER} 的 sched 下，集中放) ====="

echo "[INFO] ${MERGE_CONTAINER}: clean and prepare sched dir"
docker exec "${MERGE_CONTAINER}" /bin/bash -c "
  set -e
  rm -rf '${DST_SCHED_IN_CONTAINER}'
  mkdir -p '${DST_SCHED_IN_CONTAINER}'
"

for c in "${containers[@]}"; do
  echo "[INFO] copy nginx-1.28.0-${c} -> ${MERGE_CONTAINER}"

  docker cp \
    "${HOST_NGINX_DIR}/nginx-1.28.0-${c}" \
    "${MERGE_CONTAINER}:${DST_SCHED_IN_CONTAINER}"
done


echo
# echo "===== STEP 2.3: 在 DT_1 执行 gcovr (输出到 DT_1:${DT1_RUN_DIR}) ====="
docker exec "${MERGE_CONTAINER}" /bin/bash -c "
  set -e
  mkdir -p '${DT1_RUN_DIR}'
  cd '${DT1_RUN_DIR}'

  gcovr \
    --gcov-ignore-errors=no_working_dir_found \
    --root '${GCOVR_ROOT}' \
    --object-directory '${GCOVR_OBJDIR}' \
    --filter '${GCOVR_FILTER}' \
    --html --html-details \
    --html=proxy_coverage.merged.html \
    --xml proxy_coverage.merged.xml \
    --txt-summary
"

echo
# echo "===== STEP 2.4: 把报告从 DT_1 拷回宿主机 ====="
rm -rf "${HOST_PROXY_REPORT_DIR}"
mkdir -p "${HOST_PROXY_REPORT_DIR}"
docker cp "${MERGE_CONTAINER}:${DT1_RUN_DIR}/." "${HOST_PROXY_REPORT_DIR}/"

echo
# echo "===== CLEANUP: 删除宿主机 nginx_from_dockers ====="
rm -rf "${HOST_NGINX_DIR}"



echo
# echo "===== DONE ====="
echo "[INFO] Host coverage dir: ${HOST_OUT}/coverage"
ls -la "${HOST_OUT}/coverage" | sed -n '1,80p'
