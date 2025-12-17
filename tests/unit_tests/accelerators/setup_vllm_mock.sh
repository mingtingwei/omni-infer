#!/bin/bash
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="$(cd "$WORK_DIR/../../../" && pwd)"
echo "TOP_DIR is ${TOP_DIR}"

#download model json file
cd ${WORK_DIR}
mkdir -p mock_model/
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json -O ./mock_model/config.json
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer.json -O ./mock_model/tokenizer.json
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer_config.json -O ./mock_model/tokenizer_config.json
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/vocab.json -O ./mock_model/vocab.json

#install nginx
pkill nginx
cd ${TOP_DIR}/omni/accelerators/sched/omni_proxy/
bash build.sh --skip-extras -c
pip3 install gcovr

#install vllm
cd ${TOP_DIR}/infer_engines/
if [ ! -d "./vllm" ] || [ ! -d "./vllm/.git" ]; then
  echo "=====================installing vllm====================="
  git clone https://github.com/vllm-project/vllm.git
fi
sh bash_install_code.sh
cd ${TOP_DIR}/infer_engines/vllm
SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e .

#install omniinfer
cd ${TOP_DIR}
pip3 install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip3 install torch-npu==2.6.0
pip install -e .
