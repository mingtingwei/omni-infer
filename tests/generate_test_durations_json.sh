#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash generate_test_durations_json.sh --unit -o /path/test_durations_unit.json
#   bash generate_test_durations_json.sh --integrated -o /path/test_durations_integrated.json
#   bash generate_test_durations_json.sh -o /path/test_durations_all.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # tests/
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

target="all"
OUT="test_durations.json"
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --unit)
      target="unit"
      shift
      ;;
    --integrated|--integration)
      target="integrated"
      shift
      ;;
    -o|--out)
      OUT="$2"
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

unit_path="${SCRIPT_DIR}/unit_tests"
integrated_path="${SCRIPT_DIR}/integrated_tests"

target_path=()
case "${target}" in
  unit) target_path=("${unit_path}") ;;
  integrated) target_path=("${integrated_path}") ;;
  all) target_path=("${unit_path}" "${integrated_path}") ;;
  *) echo "[ERROR] unknown target: ${target}" >&2; exit 1 ;;
esac

# 临时 pytest 插件：收集每条用例 call 阶段耗时，写成 {nodeid: seconds}
PLUGIN="${SCRIPT_DIR}/_durations_plugin.py"
cat > "${PLUGIN}" <<'PY'
import json, os
from collections import defaultdict

D = defaultdict(float)

def pytest_runtest_logreport(report):
    # 把同一个 nodeid 的 setup/call/teardown 都累加
    D[report.nodeid] += float(getattr(report, "duration", 0.0))

def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("TEST_DURATIONS_OUT")
    if not out:
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(D), f, indent=4, sort_keys=True)
PY

export TEST_DURATIONS_OUT="${OUT}"

set +e
# 关键：在 tests/ 目录下跑，这样 -p _durations_plugin 能 import 到同目录插件文件
( cd "${SCRIPT_DIR}" && pytest -sv "${target_path[@]}" -p _durations_plugin "${extra_args[@]}" )
rc=$?
set -e

rm -f "${PLUGIN}" || true

if [[ -f "${OUT}" ]]; then
  echo "[INFO] durations json written to: ${OUT}"
else
  echo "[ERROR] durations json not generated (pytest rc=${rc})" >&2
fi

exit "${rc}"
