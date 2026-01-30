#!/usr/bin/env bash
set -euo pipefail

export COVERAGE_RCFILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.coveragerc"

pip install pytest-cov diff-cover beautifulsoup4 gcovr

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# shellcheck source=tests/utils.sh
source "${SCRIPT_DIR}/utils.sh"

target="all"
reports_dir=""
extra_args=()
CONFIG_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json-path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --unit)
      target="unit"
      shift
      ;;
    --integrated|--integration)
      target="integrated"
      shift
      ;;
    --reports-dir)
      reports_dir="$2"
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

TARGET_DIR="${SCRIPT_DIR}/unit_tests/accelerators/mock_model"

if [[ -n "${CONFIG_PATH}" ]]; then
    mkdir -p "${TARGET_DIR}"
    cp "${CONFIG_PATH}"/*.json "${TARGET_DIR}/"
    cp "${CONFIG_PATH}"/*.txt "${TARGET_DIR}/"
else
    bash "${SCRIPT_DIR}/download_config.sh"
fi

cd ${ROOT_DIR}
echo "[INFO] git status"
git status
echo "[INFO] git branch"
git branch --show-current && git log -5 --pretty=%s

cd ${ROOT_DIR}/omni/accelerators/sched/omni_proxy/
pkill -9 nginx || true
bash build.sh --skip-extras -c
unset http_proxy
unset https_proxy

target_path=()
report_name="pytest-all.xml"

unit_path="${SCRIPT_DIR}/unit_tests"
integrated_path="${SCRIPT_DIR}/integrated_tests"

case "${target}" in
  unit)
    target_path="${unit_path}"
    report_name="pytest-unit.xml"
    ;;
  integrated)
    target_path="${integrated_path}"
    report_name="pytest-integrated.xml"
    ;;
  all)
    target_path=("${unit_path}" "${integrated_path}")
    ;;
  *)
    log_warn "Unknown target '${target}', defaulting to all tests."
    ;;
esac

cmd=(
  pytest 
  --tb=long -v
  "${target_path[@]}"
  --cov
  "${extra_args[@]}"
)

if [[ -n "${reports_dir}" ]]; then
  mkdir -p "${reports_dir}"
  report_file="${reports_dir}/${report_name}"
  cmd+=(--junitxml "${report_file}")
  log_info "JUnit report will be written to ${report_file}"
fi

LOG_DIR="${ROOT_DIR}/tests/logs"
LOG_FILE="${LOG_DIR}/run_tests.log"

mkdir -p "${LOG_DIR}"

set +e
( cd "${ROOT_DIR}" && stdbuf -oL -eL "${cmd[@]}" ) 2>&1 | tee "${LOG_FILE}"
set -e

collect_coverage_reports() {
  local root_dir="$1"

  echo "[INFO] Collecting coverage reports..."

  cd "${root_dir}"

  # 1. 生成基础 coverage.xml
  coverage xml

  # 2. omni 覆盖率报告
  OMNI_INCLUDE="*/omni/layers/*,*/omni/adaptors/*,*/omni/models/*"

  mkdir -p coverage/omni_report
  coverage html --include="${OMNI_INCLUDE}" -d coverage/omni_report
  coverage xml  --include="${OMNI_INCLUDE}" -o coverage/omni_report/coverage_omni.xml

  # 3. vllm 覆盖率报告
  mkdir -p coverage/vllm_report
  coverage html --include="*/infer_engines/vllm/*" -d coverage/vllm_report
  coverage xml --include="*/infer_engines/vllm/*" -o coverage/vllm_report/coverage_vllm.xml

  # 4. 修正 vllm 路径（diff-cover 使用）
  sed -i 's|infer_engines/vllm/vllm|vllm|g' coverage/vllm_report/coverage_vllm.xml

  # 5. patch 覆盖率报告
  mkdir -p coverage/patch_report
  (cd infer_engines/vllm && git diff > "${root_dir}/combine.patch")

  diff-cover coverage/vllm_report/coverage_vllm.xml \
    --diff-file combine.patch \
    --format html:coverage/patch_report/patch_coverage.html

  python tests/patch_diff.py

  # 6. proxy 覆盖率报告
  mkdir -p coverage/proxy_report
  bash "tests/unit_tests/accelerators/gen_proxy_cov.sh"

  rm -rf coverage/proxy_report/*
  mv proxy_report/* coverage/proxy_report/

  # 7. 收集日志和最终结果
  mkdir -p tests/logs/proxy_logs
  mkdir -p tests/reports

  mv *.log tests/logs/proxy_logs 2>/dev/null || true
  mv coverage tests/reports/

  echo "[INFO] Coverage reports collected successfully."
}

collect_coverage_reports "${ROOT_DIR}"