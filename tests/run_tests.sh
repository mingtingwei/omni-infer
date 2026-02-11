#!/usr/bin/env bash
set -euo pipefail

export COVERAGE_RCFILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.coveragerc"

# 安装必要依赖（兼容单机 & 多容器）
pip install pytest-cov diff-cover beautifulsoup4 gcovr pytest-shard pytest-split numpy==1.26 transformers==4.53.2 xgrammar==0.1.19

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# shellcheck source=tests/utils.sh
source "${SCRIPT_DIR}/utils.sh"

target="all"
reports_dir=""
CONFIG_PATH=""


# 是否跳过 coverage 收集
skip_cov_collect=false

# 某些配置下 可能需要source_bashrc
source_bashrc=false

# 透传给 pytest 的参数
pytest_args=()

durations_out=""


# ==============================
# 🔍 参数解析
# ==============================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --json-path)
      CONFIG_PATH="$2"; shift 2 ;;
    --unit)
      target="unit"; shift ;;
    --integrated|--integration)
      target="integrated"; shift ;;
    --reports-dir)
      reports_dir="$2"; shift 2 ;;

    --skip-cov-collect)
      skip_cov_collect=true
      shift ;;

    --source-bashrc)
      source_bashrc=true
      shift ;;      

    --durations-out)
      durations_out="$2"; shift 2 ;;

    # 其他全部透传给 pytest
    *)
      pytest_args+=("$1"); shift ;;
  esac
done

bash "${SCRIPT_DIR}/download_config.sh" "${CONFIG_PATH:-}"

cd "${ROOT_DIR}"

echo "[INFO] git status"
GIT_PAGER=cat git status

echo "[INFO] git branch"
GIT_PAGER=cat git branch --show-current
GIT_PAGER=cat git log -5 --pretty=%s

cd ${ROOT_DIR}/omni/accelerators/sched/omni_proxy/
pkill -9 nginx || true
bash build.sh --skip-extras -c
unset http_proxy
unset https_proxy

unit_path="${SCRIPT_DIR}/unit_tests"
integrated_path="${SCRIPT_DIR}/integrated_tests"

target_path=()
report_name="pytest-all.xml"

case "${target}" in
  unit) report_name="pytest-unit.xml" ;;
  integrated) report_name="pytest-integrated.xml" ;;
  all) report_name="pytest-all.xml" ;;
esac

has_explicit_target=0
prev_was_ignore=0

for a in "${pytest_args[@]}"; do
  # 如果这是 --ignore，本身不是目标，但它的下一个参数也不是目标
  if [[ "${a}" == "--ignore" ]]; then
    prev_was_ignore=1
    continue
  fi

  # 跳过 --ignore 后面的那个参数
  if [[ "${prev_was_ignore}" -eq 1 ]]; then
    prev_was_ignore=0
    continue
  fi

  # 1) nodeid，一定是测试目标
  if [[ "${a}" == *"::"* ]]; then
    has_explicit_target=1
    break
  fi

  # 2) .py 文件（相对或绝对）
  if [[ "${a}" == *.py ]]; then
    has_explicit_target=1
    break
  fi

  # 3) tests 下的目录（相对或绝对）
  if [[ "${a}" == tests/* ]] || [[ "${a}" == */tests/* ]] || [[ "${a}" == */tests ]]; then
    has_explicit_target=1
    break
  fi
done

if [[ "${has_explicit_target}" -eq 1 ]]; then
  target_path=()
else
  case "${target}" in
    unit) target_path=("${unit_path}") ;;
    integrated) target_path=("${integrated_path}") ;;
    all) target_path=("${unit_path}" "${integrated_path}") ;;
  esac
fi

if [[ "${source_bashrc}" == true ]]; then
  set +u
  source ~/.bashrc
  set -u
fi

cmd=(pytest --tb=long -v)

if [[ -n "${durations_out}" ]]; then
  cmd+=( -p ut_CI_check.ut_CI_durations_plugin --durations-out "${durations_out}" )
fi
if [[ "${#target_path[@]}" -gt 0 ]]; then
  cmd+=("${target_path[@]}")
fi

if [[ "${#pytest_args[@]}" -gt 0 ]]; then
  cmd+=("${pytest_args[@]}")
fi

cmd+=(--cov="${ROOT_DIR}")


if [[ -n "${reports_dir}" ]]; then
  mkdir -p "${reports_dir}"
  report_file="${reports_dir}/${report_name}"
  cmd+=(--junitxml "${report_file}")
  log_info "JUnit report will be written to ${report_file}"
fi

# ==============================
# ▶️ 执行测试
# ==============================
LOG_DIR="${ROOT_DIR}/tests/logs"
LOG_FILE="${LOG_DIR}/run_tests.log"
mkdir -p "${LOG_DIR}"

echo "[INFO] About to run:"
printf '  %q ' "${cmd[@]}"
echo

set +e
( cd "${ROOT_DIR}/tests" && stdbuf -oL -eL "${cmd[@]}" ) 2>&1 | tee "${LOG_FILE}"
exit_code=$?
set -e

# ==============================
# 📊 收集覆盖率
# ==============================
if [[ "${skip_cov_collect}" == false ]]; then
  cp ${ROOT_DIR}/tests/.coverage ${ROOT_DIR} 2>/dev/null || true
  COV_SCRIPT="${SCRIPT_DIR}/collect_coverage.sh"
  source "${COV_SCRIPT}" "${ROOT_DIR}"
else
  echo "[INFO] --skip-cov-collect specified, skipping coverage collection"
fi

exit "${exit_code}"