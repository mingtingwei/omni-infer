#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOCK_DIR="${SCRIPT_DIR}/unit_tests/accelerators/mock_model"

CONFIG_PATH="${1:-}"

echo "[INFO] Preparing mock model files in ${MOCK_DIR}"

has_json=false
has_txt=false

if [[ -d "${MOCK_DIR}" ]]; then
  if compgen -G "${MOCK_DIR}"/*.json > /dev/null; then
    has_json=true
  fi
  if compgen -G "${MOCK_DIR}"/*.txt > /dev/null; then
    has_txt=true
  fi
fi

if [[ "${has_json}" == true && "${has_txt}" == true ]]; then
  echo "[INFO] Mock model config already exists, skip setup."
  exit 0
fi

mkdir -p "${MOCK_DIR}"

if [[ -n "${CONFIG_PATH}" ]]; then
  echo "[INFO] Copying mock model config from ${CONFIG_PATH}"

  cp "${CONFIG_PATH}"/*.json "${MOCK_DIR}/"
  cp "${CONFIG_PATH}"/*.txt  "${MOCK_DIR}/"

  echo "[INFO] Mock model config copied successfully."
  exit 0
fi

echo "[INFO] No local config provided, downloading default mock model config..."

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