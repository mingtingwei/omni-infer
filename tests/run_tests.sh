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
else
    bash "${SCRIPT_DIR}/download_config.sh"
fi

cd ${ROOT_DIR}/omni/accelerators/sched/omni_proxy/
bash build.sh --skip-extras -c
unset http_proxy
unset https_proxy

marker_args=()
target_path="${SCRIPT_DIR}"
report_name="pytest-all.xml"

case "${target}" in
  unit)
    target_path="${SCRIPT_DIR}/unit_tests"
    report_name="pytest-unit.xml"
    ;;
  integrated)
    marker_args=(-m "integrated and not gpu")
    target_path="${SCRIPT_DIR}/integrated_tests"
    report_name="pytest-integrated.xml"
    ;;
  all)
    marker_args=(-m "not gpu")
    ;; # run everything except GPU-tagged cases
  *)
    log_warn "Unknown target '${target}', defaulting to all tests."
    ;;
esac

cmd=(
  pytest 
  "${marker_args[@]}" 
  --tb=no -v
  "${target_path}"
  --cov
  "${extra_args[@]}"
)

if [[ -n "${reports_dir}" ]]; then
  mkdir -p "${reports_dir}"
  report_file="${reports_dir}/${report_name}"
  cmd+=(--junitxml "${report_file}")
  log_info "JUnit report will be written to ${report_file}"
fi

# 1. 执行 pytest
set +e
(cd "${ROOT_DIR}" && "${cmd[@]}")
set -e

cd "${ROOT_DIR}"
coverage xml

# 2.omni 报告
coverage html --include="*/omni/*" -d coverage_html/omni
coverage xml --include="*/omni/*" -o coverage.xml.omni

# 3.vllm 报告
coverage html --include="*/infer_engines/vllm/*" -d coverage_html/vllm
coverage xml --include="*/infer_engines/vllm/*" -o coverage.xml.vllm

# 4. 替换 coverage.xml 中的路径
sed -i 's|infer_engines/vllm/vllm|vllm|g' "coverage.xml"

# 5.生成diff报告
cd infer_engines/vllm
git diff > ${ROOT_DIR}/combine.patch
cd ${ROOT_DIR}
diff-cover coverage.xml --diff-file combine.patch --format html:coverage_html/patch_coverage.html
python tests/patch_diff.py

# 6.生成proxy报告
bash "tests/unit_tests/accelerators/gen_proxy_cov.sh"
rm -rf coverage_html/proxy_report/*
mv proxy_report/ coverage_html/proxy_report
mv *.log coverage_html/proxy_report