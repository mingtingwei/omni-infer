#!/bin/bash
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="$(cd "$WORK_DIR/../../../" && pwd)"
echo "TOP_DIR is ${TOP_DIR}"
REPORT_PATH="proxy_report"

mkdir -p ${REPORT_PATH}
rm -rf ./${REPORT_PATH}/*
gcovr --gcov-ignore-errors=no_working_dir_found --root ${TOP_DIR}/omni/accelerators/sched/nginx-1.28.0 --filter "${TOP_DIR}/omni/accelerators/sched/omni_proxy/.*\.c$" --html --html-details --html=./${REPORT_PATH}/coverage_report.html  --xml --xml=./${REPORT_PATH}/coverage_report.xml --txt-summary
tar czf proxy_cov.tar.gz ./${REPORT_PATH}