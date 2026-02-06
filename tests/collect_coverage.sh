#!/usr/bin/env bash
set -euo pipefail

# 默认行为：收集 proxy 报告
IGNORE_PROXY=false

# 解析参数
if [[ $# -eq 1 ]]; then
  REPO_ROOT="$1"
elif [[ $# -eq 2 && "$2" == "--ignore-proxy" ]]; then
  REPO_ROOT="$1"
  IGNORE_PROXY=true
else
  echo "Usage: $0 <repo_root_dir> [--ignore-proxy]"
  exit 1
fi

ROOT_DIR="$(cd "${REPO_ROOT}" && pwd)"

echo "[INFO] Collecting coverage reports..."
echo "[INFO] Root dir: ${ROOT_DIR}"
echo "[INFO] Ignore proxy report: ${IGNORE_PROXY}"

cd "${ROOT_DIR}"

# 1. 生成基础 coverage.xml
echo "[INFO] Generating base coverage.xml"
coverage xml

# 2. omni 覆盖率报告
OMNI_INCLUDE="*/omni/layers/*,*/omni/adaptors/*,*/omni/models/*"

echo "[INFO] Generating omni coverage report"
mkdir -p coverage/omni_report
coverage html --include="${OMNI_INCLUDE}" -d coverage/omni_report
coverage xml  --include="${OMNI_INCLUDE}" -o coverage/omni_report/coverage_omni.xml

# 3. vllm 覆盖率报告
echo "[INFO] Generating vllm coverage report"
mkdir -p coverage/vllm_report
coverage html --include="*/infer_engines/vllm/*" -d coverage/vllm_report
coverage xml  --include="*/infer_engines/vllm/*" -o coverage/vllm_report/coverage_vllm.xml

# 4. 修正 vllm 路径（diff-cover 使用）
echo "[INFO] Fixing vllm paths for diff-cover"
sed -i 's|infer_engines/vllm/vllm|vllm|g' coverage/vllm_report/coverage_vllm.xml

# 5. patch 覆盖率报告
echo "[INFO] Generating patch coverage report"
mkdir -p coverage/patch_report

(
  cd infer_engines/vllm
  git diff > "${ROOT_DIR}/combine.patch"
)

diff-cover coverage/vllm_report/coverage_vllm.xml \
  --diff-file combine.patch \
  --format html:coverage/patch_report/patch_coverage.html

python tests/patch_diff.py

# 6. proxy 覆盖率报告（可选）
if [[ "${IGNORE_PROXY}" == false ]]; then
  echo "[INFO] Generating proxy coverage report"
  mkdir -p coverage/proxy_report

  bash tests/unit_tests/accelerators/gen_proxy_cov.sh

  rm -rf coverage/proxy_report/*
  mv proxy_report/* coverage/proxy_report/
fi

# 7. 收集日志和最终结果
echo "[INFO] Collecting logs and final artifacts"

mkdir -p tests/logs/proxy_logs
mkdir -p tests/reports

rm -rf tests/reports/*
mv coverage tests/reports/
mv *.log tests/logs/proxy_logs 2>/dev/null || true

echo "[INFO] Coverage reports collected successfully."