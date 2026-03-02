#!/usr/bin/env bash
# Central UT configuration for multi-docker runs.
# This file is intended to be sourced by other scripts in this directory.

set -euo pipefail

# ------------------------------
# Container list (order matters)
# ------------------------------
export CONTAINER_NAMES=(
  "DT_1"
  "DT_2"
  "DT_3"
  "DT_4"
  "DT_5"
  "DT_6"
  "DT_7"
)

# ------------------------------
# Device assignment per container
# ------------------------------
# NOTE: associative array must be declared in the same shell that uses it (source this file).
declare -gA CONTAINER_DEVICES=(
  ["DT_1"]="0,1"
  ["DT_2"]="2,3"
  ["DT_3"]="4,5"
  ["DT_4"]="6,7,8,9"
  ["DT_5"]="10,11"
  ["DT_6"]="12,13"
  ["DT_7"]="14,15"
)

# ------------------------------
# In-container paths
# ------------------------------
export CONTAINER_OMNI_ROOT_DEFAULT="/workspace/omniinfer"

# If caller wants to override, they can export CONTAINER_OMNI_ROOT before sourcing.
export CONTAINER_OMNI_ROOT="${CONTAINER_OMNI_ROOT:-$CONTAINER_OMNI_ROOT_DEFAULT}"

export OMNI="$CONTAINER_OMNI_ROOT"
export UNIT_TESTS="$OMNI/tests/unit_tests"
export PROXY_DIR="$UNIT_TESTS/accelerators"
export DURATIONS="$OMNI/tests/test_durations_unit.json"
export CONTAINER_DURATIONS_DIR="$CONTAINER_OMNI_ROOT/tests/test_durations_from_dockers"

# ------------------------------
# Test selection / splitting rules
# ------------------------------
# DT_1-3: proxy-related tests
# DT_4 : pytest mark npu_2cards_4dies (2-cards tests)
# DT_5-7: remaining tests split by durations into 3 groups

# DT_1 list
DT1_TESTS=(
  "$PROXY_DIR/test_proxy.py"
  "$PROXY_DIR/test_proxy_reload.py"
)

# DT_2 list
DT2_TESTS=(
  "$PROXY_DIR/test_api_server.py"
  "$PROXY_DIR/test_proxy_load_balance.py"
  "$UNIT_TESTS/api"
)

# Join helper
_join_by_space() { local IFS=' '; echo "$*"; }

export DT1_RUN_LIST="$(_join_by_space "${DT1_TESTS[@]}")"
export DT2_RUN_LIST="$(_join_by_space "${DT2_TESTS[@]}")"

# Build ignores for DT_3: run all proxy tests except DT_1/DT_2 assigned ones
IGNORE_ALL_ACCEL=""
for t in "${DT1_TESTS[@]}" "${DT2_TESTS[@]}"; do
  IGNORE_ALL_ACCEL+=" --ignore $t"
done
export IGNORE_ALL_ACCEL

# Split config for DT_5-7 (quotes left to caller to avoid over-escaping)
export COMMON_SPLITTED_ARGS="$UNIT_TESTS --ignore $PROXY_DIR --ignore $UNIT_TESTS/api -m \"not(npu_2cards_4dies)\" --splits 3 --splitting-algorithm least_duration --durations-path $DURATIONS"

# Container -> pytest args mapping
declare -gA CONTAINER_TEST_ARGS=(
  ["DT_1"]="$DT1_RUN_LIST"
  ["DT_2"]="$DT2_RUN_LIST"
  ["DT_3"]="$PROXY_DIR$IGNORE_ALL_ACCEL"
  ["DT_4"]='-m "npu_2cards_4dies"'
  ["DT_5"]="$COMMON_SPLITTED_ARGS --group 1"
  ["DT_6"]="$COMMON_SPLITTED_ARGS --group 2"
  ["DT_7"]="$COMMON_SPLITTED_ARGS --group 3"
)

# ------------------------------
# Common docker run options (used by run_docker.sh)
# ------------------------------
export DOCKER_SHM_SIZE="${DOCKER_SHM_SIZE:-500g}"

# Minimal helper: check container name exists in config
ut_has_container() {
  local name="$1"
  for c in "${CONTAINER_NAMES[@]}"; do
    [[ "$c" == "$name" ]] && return 0
  done
  return 1
}