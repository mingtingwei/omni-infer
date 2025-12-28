from omni.adaptors.vllm.patches import model_patch
import pytest
import os
import gc
import torch
import time
import torch.multiprocessing as mp
import tempfile
from unittest.mock import patch

from vllm.platforms import current_platform
from vllm.config import CompilationLevel
from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.config import set_current_vllm_config

from omni.models.config_loader.loader import model_extra_config
from omni.quantization.compressed_tensors.compressed_tensors import AscendCompressedTensorsConfig
from .base import MockRunner
from .utils import creat_vllm_config, load_configs, get_speculative_config, three_sigma_filter
from .registry import HF_EXAMPLE_MODELS

class Test_e2e_models():
    def init_distributed(self, local_rank, world_size: int):
        with set_current_vllm_config(self.vllm_config, check_compile=False):
            init_distributed_environment(
                world_size=world_size,
                rank=local_rank,
                distributed_init_method=f"file://{self.temp_file_path}",
                local_rank=local_rank,
                backend="hccl",
            )
            initialize_model_parallel(world_size, 1)

    def _latency_test(self, num_tokens, model_type, threshold, test_count=100, redundancy=5):
        res = []
        for idx in range(test_count + redundancy):
            start_t = time.time()
            _ = self.mock_runner.forward_decode(num_tokens, self.vllm_config.scheduler_config.max_batch_size)
            end_t = time.time()
            if idx >= redundancy:
                res.append(end_t - start_t)
        filtered_times = three_sigma_filter(res)
        assert filtered_times <= float(threshold) * 1.05, f"Model:{model_type} the inference latency \
            {filtered_times} exceeds threshold {float(threshold) *1.05}"

    def _model_runner(self, local_rank: int, world_size: int, model_info, enable_graph):

        os.environ["GLOO_SOCKET_IFNAME"] = os.environ.get(
            "GLOO_SOCKET_IFNAME", "enp23s0f3"
        )
        os.environ["TP_SOCKET_IFNAME"] = os.environ.get(
            "TP_SOCKET_IFNAME", "enp23s0f3"
        )
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

        self.vllm_config = creat_vllm_config(model_info.hf_config)

        from omni.models.config_loader.loader import model_extra_config
        if model_info.model_cls.__name__ == "Qwen3MoeForCausalLM":
            model_extra_config.operator_opt_config.decode_moe_dispatch_combine = False
        # enable graph compile
        if enable_graph:
            self.vllm_config.npu_compilation_config.level = CompilationLevel.DYNAMO_AS_IS

        if self.vllm_config.model_config.hf_config.enable_quantization:
            required_key = self.vllm_config.model_config.hf_config.model_type
            quant_configs = load_configs(config_mode="quantization_config", required_key=required_key)
            self.vllm_config.model_config.hf_config.quantization_config = AscendCompressedTensorsConfig.from_config(quant_configs)

        if self.vllm_config.model_config.hf_config.enable_speculative:
            self.vllm_config.speculative_config = get_speculative_config()
            decode_bsz = 1 + self.vllm_config.speculative_config.num_speculative_tokens
            self.vllm_config.scheduler_config.max_batch_size *= decode_bsz

        decode_gear_list = [self.vllm_config.scheduler_config.max_batch_size]
        # Optional: update model_extra_config
        model_extra_config.task_config.decode_gear_list = decode_gear_list
        # Optional: update vllm_config
        self.vllm_config.npu_compilation_config.decode_gear_list = decode_gear_list
        torch.npu.set_device(local_rank)
        self.device = torch.device(f"{current_platform.device_type}:0")

        self.init_distributed(local_rank, world_size)

        self.mock_runner = MockRunner(self.vllm_config, self.device)

        # initialize model
        with model_info.init_patch_context():
            self.model = self.mock_runner.init_model(model_info.model_cls)
        assert isinstance(
            self.model,
            model_info.model_cls,
        )

        # profile run
        self.mock_runner._dummy_run(self.vllm_config.scheduler_config.max_batch_size)
        gc.collect()

        # init kv_cache
        self.mock_runner.init_kv_cache()

        # test forward dummy
        forward_results = self.mock_runner._dummy_run(self.vllm_config.scheduler_config.max_batch_size)
        print(f"{forward_results[0].shape=}, {forward_results[1].shape=}")
        if isinstance(forward_results, tuple):
            assert forward_results[0].shape == torch.Size([self.vllm_config.scheduler_config.max_batch_size, self.vllm_config.model_config.hf_config.hidden_size])
            assert forward_results[1].shape == torch.Size([self.vllm_config.scheduler_config.max_batch_size, self.vllm_config.model_config.hf_config.vocab_size])
        else:
            assert forward_results.shape == torch.Size([self.vllm_config.scheduler_config.max_batch_size, self.vllm_config.model_config.hf_config.hidden_size])
        # test forward prefill
        forward_results = self.mock_runner.forward_prefill(model_info.prompt_token_ids, self.vllm_config.scheduler_config.max_batch_size)
        if isinstance(forward_results, tuple):
            assert forward_results[0].shape == torch.Size([len(model_info.prompt_token_ids), self.vllm_config.model_config.hf_config.hidden_size])
            assert forward_results[1].shape == torch.Size([1, self.vllm_config.model_config.hf_config.vocab_size])
        else:
            assert forward_results.shape == torch.Size([len(model_info.prompt_token_ids), self.vllm_config.model_config.hf_config.hidden_size])
        # test forward decode
        num_tokens = 1 # without speculative tokens
        forward_results = self.mock_runner.forward_decode(num_tokens, self.vllm_config.scheduler_config.max_batch_size)
        print(f"forward_decode: {forward_results[0].shape=}, {forward_results[1].shape=}")
        if isinstance(forward_results, tuple):
            assert forward_results[0].shape == torch.Size([self.vllm_config.scheduler_config.max_batch_size, self.vllm_config.model_config.hf_config.hidden_size])
            assert forward_results[1].shape == torch.Size([self.vllm_config.scheduler_config.max_batch_size, self.vllm_config.model_config.hf_config.vocab_size])
        else:
            assert forward_results.shape == torch.Size([self.vllm_config.scheduler_config.max_batch_size, self.vllm_config.model_config.hf_config.hidden_size])

        if enable_graph:
            self._latency_test(num_tokens=num_tokens, 
                               model_type=self.vllm_config.model_config.hf_config.model_type, 
                               threshold=self.vllm_config.model_config.hf_config.decode_cost_time)

    @pytest.mark.parametrize("enable_graph", [False, True])
    @pytest.mark.parametrize("world_size", [1])
    @pytest.mark.parametrize("model_arch", HF_EXAMPLE_MODELS.get_supported_archs())
    def test_model(self, world_size: int, model_arch: str, enable_graph: bool):
        model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)

        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            self.temp_file_path = tfile.name

        try:
            mp.spawn(
                self._model_runner,
                args=(world_size, model_info, enable_graph),
                nprocs=world_size,
            )
        finally:
            if os.path.exists(self.temp_file_path):
                os.remove(self.temp_file_path)

