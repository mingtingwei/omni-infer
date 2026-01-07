export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_ENABLE_MC2=0
export USING_LCCL_COM=0

export OMNI_USE_PANGU=1
export FORCE_ENABLE_CHUNK_PREFILL=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HCCL_RDMA_TIMEOUT=5

export ASCEND_GLOBAL_LOG_LEVEL=3
export CPU_AFFINITY_CONF=2
export VLLM_LOGGING_LEVEL=INFO
export OMNI_REUSE_PREFILLED_TOKENS=1
export OMNI_SKIP_DECODE_TOKENIZE=1
export HCCL_BUFFSIZE=1000
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0

export PROFILER_TOKEN_THRESHOLD=2000

python start_api_servers.py \
    --num-servers 1 \
    --model-path /data/weights/7BV5_VL \
    --master-ip 127.0.0.1 \
    --base-api-port 8088 \
    --master-port 62345 \
    --tp 1 \
    --max-model-len 32768 \
    --served-model-name openpangu_vl \
    --log-dir ./qwen_vl \
    --gpu-util 0.9 \
    --extra-args "--max-num-batched-tokens 32768 --max-num-seqs 144" \
    --no-enable-prefix-caching \
    --additional-config '{"graph_model_compile_config":{"level":1, "use_ge_graph_cached":false, "block_num_floating_range":50, "decode_gear_list":[64]}, "enable_hybrid_graph_mode": true}' \


