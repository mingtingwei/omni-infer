#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/ut_config.sh"

LOG_DIR="${SCRIPT_DIR}/install_logs"
mkdir -p "${LOG_DIR}"

OMNI_ROOT="${1:?Usage: $0 <omniinfer_root_dir> [config_path]}"
CONFIG_PATH="${2:-}"

HOST_DURATIONS_DIR="${OMNI_ROOT}/tests/test_durations_from_dockers"
mkdir -p "${HOST_DURATIONS_DIR}"

pids=()

bash "${OMNI_ROOT}/tests/download_config.sh" "${CONFIG_PATH:-}"

# ------------------------------
# Start all container tasks
# ------------------------------
for CONTAINER_NAME in "${CONTAINER_NAMES[@]}"; do
  TEST_ARGS_STR="${CONTAINER_TEST_ARGS[$CONTAINER_NAME]:-}"
  if [[ -z "${TEST_ARGS_STR}" ]]; then
    echo "[ERROR] Missing test args for ${CONTAINER_NAME} in ut_config.sh"
    exit 1
  fi

  echo "[INFO] Syncing and setting env in ${CONTAINER_NAME}"
  echo "[INFO]   test args: ${TEST_ARGS_STR}"

  docker exec "${CONTAINER_NAME}" /bin/bash -c "
    set -e
    . ~/.bashrc || true

    mkdir -p '${CONTAINER_OMNI_ROOT}'
    rm -rf '${CONTAINER_OMNI_ROOT:?}'/*
    rm -rf '${CONTAINER_OMNI_ROOT}'/.??* || true

    cp -r '${OMNI_ROOT}'/* '${CONTAINER_OMNI_ROOT}'/
    cp -r '${OMNI_ROOT}'/.git '${CONTAINER_OMNI_ROOT}'/

    yum install -y libuuid-devel

    mkdir -p '${CONTAINER_DURATIONS_DIR}'
    export PYTHONPATH='${CONTAINER_OMNI_ROOT}/infer_engines/vllm:${CONTAINER_OMNI_ROOT}':\$PYTHONPATH

    bash '${CONTAINER_OMNI_ROOT}/tests/run_tests.sh' --skip-cov-collect \
      --durations-out '${CONTAINER_DURATIONS_DIR}/test_durations_${CONTAINER_NAME}.json' \
      ${TEST_ARGS_STR}
  " > "${LOG_DIR}/${CONTAINER_NAME}.log" 2>&1 &

  pids+=($!)
done

# ------------------------------
# Wait & collect status
# ------------------------------
echo "[INFO] Waiting for all containers to finish..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

# ------------------------------
# Collect per-container durations and merge
# ------------------------------
echo "[INFO] Collecting per-container test durations..."
for c in "${CONTAINER_NAMES[@]}"; do
  src="${CONTAINER_DURATIONS_DIR}/test_durations_${c}.json"
  dst="${HOST_DURATIONS_DIR}/test_durations_${c}.json"
  if docker exec "${c}" /bin/bash -c "test -f '${src}'"; then
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
    [[ -f "${f}" ]] && args+=("--container-json" "${f}")
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
  "${CONTAINER_OMNI_ROOT}"

exit "${fail}"