import os
import json
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union
from unittest.mock import MagicMock, Mock, patch

import torch
from vllm.config import CacheConfig, ParallelConfig, CompilationLevel, CompilationConfig, SchedulerConfig, DeviceConfig, SpeculativeConfig
from vllm.platforms import current_platform
from omni.adaptors.vllm.compilation.compile_config import NPUCompilationConfig

default_config_path = os.path.normpath(os.path.join(os.path.abspath(__file__), '../configs'))

def creat_vllm_config(base_config):
    model_config = SimpleNamespace(
        hf_config=base_config,
        hf_text_config=base_config,
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
        use_mla=True,
        quant_config=None,
        max_model_len=2048,
        is_attention_free=False,
        disable_cascade_attn = True,
    )

    parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
        )
    cache_config = CacheConfig()
    cache_config.block_size = 128
    compilation_config = CompilationConfig()
    compilation_config.level = 0
    compilation_config.full_cuda_graph = False
    compilation_config.custom_ops = ['all']
    compilation_config.enabled_custom_ops = MagicMock(spec=Counter[str])
    compilation_config.disabled_custom_ops = MagicMock(spec=Counter[str])
    compilation_config.static_forward_context = {}
    npu_compilation_config = NPUCompilationConfig()
    npu_compilation_config.level = CompilationLevel.NO_COMPILATION
    npu_compilation_config.use_ge_graph_cached = False

    scheduler_config = SchedulerConfig()
    scheduler_config.max_num_seqs = 4
    scheduler_config.max_num_batched_tokens = 2048
    scheduler_config.max_batch_size = 4
    scheduler_config.preemption_mode = None
    scheduler_config.chunked_prefill_enabled = False


    device_config = DeviceConfig(device=current_platform.device_type)

    vllm_config = Mock()
    vllm_config.model_config = model_config
    vllm_config.cache_config = cache_config
    vllm_config.quant_config = None
    vllm_config.parallel_config = parallel_config
    vllm_config.compilation_config = compilation_config
    vllm_config.npu_compilation_config = npu_compilation_config
    vllm_config.scheduler_config = scheduler_config
    vllm_config.device_config = device_config
    vllm_config.speculative_config = None
    vllm_config.kv_transfer_config = None
    return vllm_config

@patch("vllm.config.SpeculativeConfig",
new_callable=lambda: MagicMock(spec=SpeculativeConfig))
def get_speculative_config(mock_speculative_config):
    mock_speculative_config.method = 'deepseek_mtp'
    mock_speculative_config.model = ''
    mock_speculative_config.num_speculative_tokens = 1
    return mock_speculative_config

def load_configs(
    base_path: Union[str, Path] = default_config_path,
    config_mode: Optional[str] = None,
    required_key: Optional[str] = None,
    encoding: str = "utf-8",
    verbose: bool = False
) -> Dict[str, Any]:
    base_path = Path(base_path)
    if not base_path.exists():
            raise FileNotFoundError(f"Config base path is not existed!: {base_path}")

    config_file = os.path.join(base_path, "{}.json".format(config_mode))
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"config_file is not existed!: {config_file}")
    
    with open(config_file, 'r', encoding=encoding) as f:
        config_data = json.load(f)

    if required_key not in config_data:
        error_msg = (
            f"Configuration file is missing required key: '{required_key}'。\n"
            f"Loaded configuration files: {config_file}\n"
            f"Keys available at the top level: {list(config_data.keys())}"
        )
        raise KeyError(error_msg)
        return None

    return config_data[required_key]