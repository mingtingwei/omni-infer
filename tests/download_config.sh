#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MOCK_DIR="${SCRIPT_DIR}/unit_tests/accelerators/mock_model"

echo "[INFO] Preparing mock model files in ${MOCK_DIR}"

mkdir -p "${MOCK_DIR}"

BASE_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main"

wget -q --show-progress --no-check-certificate "${BASE_URL}/config.json" \
  -O "${MOCK_DIR}/config.json"

wget -q --show-progress --no-check-certificate "${BASE_URL}/tokenizer.json" \
  -O "${MOCK_DIR}/tokenizer.json"

wget -q --show-progress --no-check-certificate "${BASE_URL}/tokenizer_config.json" \
  -O "${MOCK_DIR}/tokenizer_config.json"

wget -q --show-progress --no-check-certificate "${BASE_URL}/vocab.json" \
  -O "${MOCK_DIR}/vocab.json"

wget -q --show-progress --no-check-certificate "${BASE_URL}/merges.txt" \
  -O "${MOCK_DIR}/merges.txt"

echo "[INFO] Mock model files downloaded successfully."
