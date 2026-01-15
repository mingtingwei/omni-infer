import contextlib
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CompilationLevel

from omni.adaptors.vllm.worker.npu_model_runner import NPUModelRunner

torch_npu = pytest.importorskip("torch_npu")


class DummyModelConfig:
    def __init__(
        self,
        *,
        use_mla: bool = False,
        uses_mrope: bool = False,
        max_model_len: int = 8,
        head_size: int = 4,
        vocab_size: int = 32,
        num_layers: int = 2,
    ):
        self.model = "dummy"
        self.dtype = torch.float16
        self.use_mla = use_mla
        self.uses_mrope = uses_mrope
        self.max_model_len = max_model_len
        self.head_size = head_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.is_multimodal_model = False
        self.attention_chunk_size = 0
        self.disable_cascade_attn = False
        self.hf_config = SimpleNamespace(model_type="decoder")
        self.hf_text_config = SimpleNamespace()

    def get_num_attention_heads(self, parallel_config):
        return 1

    def get_hidden_size(self):
        return 4

    def get_head_size(self):
        return self.head_size

    def get_num_layers_by_block_type(self, parallel_config, block_type):
        return self.num_layers

    def get_vocab_size(self):
        return self.vocab_size


class DummyParallelConfig:
    def __init__(self, tp: int = 1, pp: int = 1, dp: int = 1, rank: int = 0):
        self.tensor_parallel_size = tp
        self.pipeline_parallel_size = pp
        self.data_parallel_size = dp
        self.rank = rank
        self.data_parallel_rank = 0
        self.world_size_across_dp = tp * pp * dp


class DummyCacheConfig:
    def __init__(self, block_size: int = 4, cache_dtype: str = "auto", swap_space_bytes: int = 2048):
        self.block_size = block_size
        self.cache_dtype = cache_dtype
        self.cpu_offload_gb = 0
        self.swap_space_bytes = swap_space_bytes


class DummySchedulerConfig:
    def __init__(
        self,
        *,
        max_num_batched_tokens: int = 16,
        max_num_seqs: int = 2,
        enable_chunked_prefill: bool = False,
        preemption_mode: str | None = None,
        num_step: int = 1,
    ):
        self.runner_type = "generate"
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.chunked_prefill_enabled = enable_chunked_prefill
        self.preemption_mode = preemption_mode
        self.num_step = num_step
        self.max_num_partial_prefills = 1
        self.long_prefill_token_threshold = 0
        self.multi_step_stream_outputs = True
        self.num_scheduler_steps = 1
        self.send_delta_data = False


class DummySpeculativeConfig:
    def __init__(self, num_speculative_tokens: int = 1, enable_adaptive: bool = False, method: str = "ngram"):
        self.num_speculative_tokens = num_speculative_tokens
        self.enable_adaptive = enable_adaptive
        self.method = method

    def use_eagle(self):
        return False


class DummyKVTransferConfig:
    def __init__(self, kv_role: str | None):
        self.kv_role = kv_role


class DummyCompilationConfig:
    def __init__(self, level: int = CompilationLevel.NO_COMPILATION):
        self.level = level
        self.cudagraph_capture_sizes: list[int] = []
        self.static_forward_context = {}


class DummyNPUCompilationConfig:
    def __init__(
        self,
        *,
        level: int = CompilationLevel.NO_COMPILATION,
        decode_gear_list: list[int] | None = None,
        use_ge_graph_cached: bool = False,
    ):
        self.level = level
        self.decode_gear_list = decode_gear_list
        self.use_ge_graph_cached = use_ge_graph_cached


class FakeSampler:
    def __init__(self, *_, **__):
        self.prepared = False

    def prepare_cache(self, *_, **__):
        self.prepared = True

    def __call__(self, *, logits, sampling_metadata):
        batch = sampling_metadata.batch_size if hasattr(sampling_metadata, "batch_size") else logits.shape[0]
        return SimpleNamespace(sampled_token_ids=torch.zeros((batch, 1), dtype=torch.int64, device=logits.device),
                               logprobs_tensors=None)


class FakeValidator:
    def __init__(self, *_, **__):
        pass

    def parse_output(self, tensor, vocab_size):
        return tensor.tolist()


class FakeDrafter:
    def __init__(self, *_, **__):
        self.loaded = False

    def load_model(self, model):
        self.loaded = True


@pytest.fixture(autouse=True)
def patch_runner_dependencies(monkeypatch):
    import importlib

    gpu_runner_mod = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    utils_mod = importlib.import_module("vllm.model_executor.models.utils")
    if not hasattr(gpu_runner_mod, "set_cpu_offload_max_bytes"):
        setattr(gpu_runner_mod, "set_cpu_offload_max_bytes", utils_mod.set_cpu_offload_max_bytes)

    monkeypatch.setattr(gpu_runner_mod, "NgramProposer", lambda *args, **kwargs: SimpleNamespace(), raising=False)
    monkeypatch.setattr(gpu_runner_mod, "MedusaProposer", lambda *args, **kwargs: SimpleNamespace(), raising=False)

    def safe_bind(kv_caches, forward_context, runner_kv_caches):
        runner_kv_caches.extend(kv_caches.values())
        if forward_context is None:
            return
        for layer, cache in kv_caches.items():
            if layer not in forward_context:
                forward_context[layer] = SimpleNamespace()
            forward_context[layer].kv_cache = [cache]

    monkeypatch.setattr("vllm.v1.utils.bind_kv_cache", safe_bind, raising=False)
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.bind_kv_cache", safe_bind, raising=False)
    monkeypatch.setattr(gpu_runner_mod, "compute_encoder_budget", lambda *_, **__: (0, 0))
    monkeypatch.setattr(gpu_runner_mod, "set_cpu_offload_max_bytes", lambda *_: None)

    dummy_props = SimpleNamespace(multi_processor_count=1)
    monkeypatch.setattr(gpu_runner_mod.torch.cuda, "get_device_properties", lambda device: dummy_props)
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.torch.cuda.get_device_properties",
                        lambda device: dummy_props)
    monkeypatch.setattr("vllm.utils.supports_dynamo", lambda: True)
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.set_forward_context",
                        lambda *_, **__: contextlib.nullcontext())
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.has_kv_transfer_group", lambda: False)


@pytest.fixture
def parallel_state(monkeypatch):
    import importlib

    class PPGroup:
        def __init__(self, is_last_rank=True):
            self.is_last_rank = is_last_rank
            self.is_first_rank = True

    pp_group = PPGroup(is_last_rank=True)
    dp_group = SimpleNamespace(world_size=1, cpu_group="cpu")

    gpu_runner_mod = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    parallel_state_mod = importlib.import_module("vllm.distributed.parallel_state")
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.get_pp_group", lambda: pp_group)
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.get_dp_group", lambda: dp_group)
    monkeypatch.setattr(gpu_runner_mod, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(gpu_runner_mod, "get_dp_group", lambda: dp_group, raising=False)
    monkeypatch.setattr(parallel_state_mod, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(parallel_state_mod, "get_dp_group", lambda: dp_group)
    return pp_group


@pytest.fixture
def sampler_and_drafter(monkeypatch):
    monkeypatch.setattr("omni.adaptors.vllm.sample.sampler.AscendSamplerV1", FakeSampler)
    monkeypatch.setattr("omni.adaptors.vllm.sample.validator.SimpleValidator", FakeValidator)
    monkeypatch.setattr("omni.adaptors.vllm.sample.validator.SparseRejectionSamplerValidator", FakeValidator)
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.PostDrafter", FakeDrafter)


def make_vllm_config(
    *,
    model_config: DummyModelConfig,
    cache_config: DummyCacheConfig,
    scheduler_config: DummySchedulerConfig,
    parallel_config: DummyParallelConfig,
    npu_compilation_config: DummyNPUCompilationConfig,
    spec_config: DummySpeculativeConfig | None,
    kv_role: str | None = None,
    additional_config: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        device_config=None,
        load_config=SimpleNamespace(),
        lora_config=None,
        speculative_config=spec_config,
        decoding_config=None,
        observability_config=None,
        prompt_adapter_config=None,
        quant_config=None,
        compilation_config=DummyCompilationConfig(),
        kv_transfer_config=DummyKVTransferConfig(kv_role) if kv_role else None,
        kv_events_config=None,
        additional_config=additional_config or {},
        instance_id="",
        npu_compilation_config=npu_compilation_config,
    )


def test_hybrid_chunk_prefill_graph_mode_npu(parallel_state, sampler_and_drafter, npu_device):
    torch.manual_seed(3)
    torch_npu.npu.manual_seed(3)

    model_cfg = DummyModelConfig()
    cache_cfg = DummyCacheConfig()
    sched_cfg = DummySchedulerConfig(max_num_batched_tokens=20, max_num_seqs=2, enable_chunked_prefill=True)
    parallel_cfg = DummyParallelConfig()
    npu_comp_cfg = DummyNPUCompilationConfig(level=CompilationLevel.PIECEWISE, decode_gear_list=[4, 8])

    vllm_cfg = make_vllm_config(
        model_config=model_cfg,
        cache_config=cache_cfg,
        scheduler_config=sched_cfg,
        parallel_config=parallel_cfg,
        npu_compilation_config=npu_comp_cfg,
        spec_config=None,
        kv_role=None,
        additional_config={},
    )

    runner = NPUModelRunner(vllm_cfg, npu_device)

    assert runner.is_hybrid_chunked_prefill_graph_mode
    assert runner.max_batch_size == runner.max_num_tokens

    print(f"NPU_DEVICE={npu_device}")
    print("HYBRID_CHUNK_PREFILL_NPU_OK")
