
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/install_logs"
mkdir -p "${LOG_DIR}"

OMNI_ROOT="${1:?Usage: $0 <omniinfer_root_dir>}"
CONTAINER_OMNI_ROOT="/workspace/omniinfer"
CONFIG_PATH="${2:-}"


HOST_DURATIONS_DIR="${OMNI_ROOT}/tests/test_durations_from_dockers"
CONTAINER_DURATIONS_DIR="${CONTAINER_OMNI_ROOT}/tests/test_durations_from_dockers"
mkdir -p "${HOST_DURATIONS_DIR}"

pids=()

bash "${OMNI_ROOT}/tests/download_config.sh" "${CONFIG_PATH:-}"

# ==============================
# 📦 定义每个容器的专属测试参数
# ==============================
# 容器用例切分策略配置
# DT_1-4 运行 omni-proxy相关用例，其中DT_1-3配置用例文件，DT_4运行DT_1-3之外的proxy用例
# DT_5运行 pytest mark npu_2cards_4dies的双卡用例
# DT_6-7 按用例执行时间，等分其余用例
OMNI="/workspace/omniinfer"
UNIT_TESTS="$OMNI/tests/unit_tests"
PROXY_DIR="$UNIT_TESTS/accelerators"
DURATIONS="$OMNI/tests/test_durations_unit.json"

# DT_1-4配置
DT1_TESTS=(
  "$PROXY_DIR/test_api_server.py"
  "$PROXY_DIR/test_chunked_prefill_scheduler.py"
  "$PROXY_DIR/test_proxy.py"
  "$UNIT_TESTS/api"
) # DT_1

ALL_SELECT_PROXY_TESTS=(
  "${DT1_TESTS[@]}"
  "$PROXY_DIR/test_proxy_group.py"     # DT_2
  "$PROXY_DIR/test_proxy_reload.py"    # DT_3
)

DT1_RUN_LIST=$(IFS=' '; echo "${DT1_TESTS[*]}")

IGNORE_ALL_ACCEL=""
for t in "${ALL_SELECT_PROXY_TESTS[@]}"; do
  IGNORE_ALL_ACCEL+=" --ignore $t"
done

# DT_6-7 配置
COMMON_SPLITTED_ARGS="$UNIT_TESTS --ignore $PROXY_DIR --ignore $UNIT_TESTS/api -m \"not(npu_2cards_4dies)\" --splits 2 --splitting-algorithm least_duration --durations-path $DURATIONS"

# ==============================
# 📦 容器测试参数
# ==============================
declare -A CONTAINER_TEST_ARGS=(
  [DT_1]="$DT1_RUN_LIST"
  [DT_2]="${ALL_SELECT_PROXY_TESTS[4]}"
  [DT_3]="${ALL_SELECT_PROXY_TESTS[5]}"
  [DT_4]="$PROXY_DIR$IGNORE_ALL_ACCEL"
  [DT_5]='-m "npu_2cards_4dies"'
  [DT_6]="$COMMON_SPLITTED_ARGS --group 1"
  [DT_7]="$COMMON_SPLITTED_ARGS --group 2"
)

# ==============================
# 🔁 启动所有容器任务
# ==============================
for CONTAINER_NAME in "${!CONTAINER_TEST_ARGS[@]}"; do
  echo "[INFO] Syncing and setting env in ${CONTAINER_NAME}"

  # 获取参数字符串（注意：这里不展开，原样传入）
  TEST_ARGS_STR="${CONTAINER_TEST_ARGS[$CONTAINER_NAME]}"

  # 构造容器内执行的命令（注意变量转义！）
  docker exec "${CONTAINER_NAME}" /bin/bash -c "
    . ~/.bashrc
    set -e
    mkdir -p ${CONTAINER_OMNI_ROOT}
    rm -rf ${CONTAINER_OMNI_ROOT}/*
    rm -rf ${CONTAINER_OMNI_ROOT}/.??* || true
    cp -r ${OMNI_ROOT}/* ${CONTAINER_OMNI_ROOT}/
    cp -r ${OMNI_ROOT}/.git ${CONTAINER_OMNI_ROOT}/
    yum install -y libuuid-devel

    mkdir -p ${CONTAINER_DURATIONS_DIR}
    export PYTHONPATH=/workspace/omniinfer/infer_engines/vllm:/workspace/omniinfer:\$PYTHONPATH

    bash /workspace/omniinfer/tests/run_tests.sh --skip-cov-collect --durations-out "${CONTAINER_DURATIONS_DIR}/test_durations_${CONTAINER_NAME}.json" ${TEST_ARGS_STR}
  " > "${LOG_DIR}/${CONTAINER_NAME}.log" 2>&1 &

  pids+=($!)
done

# ==============================
# 🕒 等待并收集结果
# ==============================
echo "[INFO] Waiting for all containers to finish..."

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

# ==============================
# Collect per-container durations and merge
# ==============================
echo "[INFO] Collecting per-container test durations..."
for c in "${!CONTAINER_TEST_ARGS[@]}"; do
  src="${CONTAINER_DURATIONS_DIR}/test_durations_${c}.json"
  dst="${HOST_DURATIONS_DIR}/test_durations_${c}.json"
  if docker exec "${c}" /bin/bash -c "test -f ${src}"; then
    echo "[INFO] docker cp ${c}:${src} -> ${dst}"
    docker cp "${c}:${src}" "${dst}"
  else
    echo "[WARN] durations json not found in ${c}: ${src}"
  fi
done

MERGED_JSON="${HOST_DURATIONS_DIR}/merged_test_durations.json"
MERGE_SCRIPT="${OMNI_ROOT}/tests/ut_CI_check/ut_CI_merge_test_durations_json.py"
if [[ -f "${MERGE_SCRIPT}" ]]; then
  args=("--out" "${MERGED_JSON}")
  for f in "${HOST_DURATIONS_DIR}"/test_durations_*.json; do
    if [[ -f "${f}" ]]; then
      args+=("--container-json" "${f}")
    fi
  done
  if [[ ${#args[@]} -gt 2 ]]; then
    python3 "${MERGE_SCRIPT}" "${args[@]}"
  else
    echo "[WARN] No durations json files to merge in ${HOST_DURATIONS_DIR}"
  fi
else
  echo "[WARN] merge script not found: ${MERGE_SCRIPT}"
fi

bash "${OMNI_ROOT}/tests/multi_docker_collect_coverage.sh" \
  "${OMNI_ROOT}/tests/reports" \
  "${OMNI_ROOT}" \
  "/workspace/omniinfer"

exit "${fail}"