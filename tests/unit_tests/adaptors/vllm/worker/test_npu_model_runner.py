import types
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm.config import CompilationLevel
from vllm.v1.kv_cache_interface import (AttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor,
                                        SlidingWindowSpec)

from omni.adaptors.vllm.worker.npu_model_runner import (
    NPUModelRunner,
    _get_pad_size,
)
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.layers.attention.backend.attention_dummy_builder import (
    DummyAttentionMetadataBuilder,
)


def parse_ascend_devices():
    import os
    # Get the environment variable, default to empty string if not found
    env_val = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')
    
    if not env_val.strip():
        # Handle case where env var is missing or empty
        return 0, [0, 1]
    try:
        # Split by comma and convert to integers
        visible_die_list = [int(x.strip()) for x in env_val.split(',') if x.strip()]
        first_die_no = visible_die_list[0] if visible_die_list else None
    except ValueError as e:
        print(f"Error parsing ASCEND_RT_VISIBLE_DEVICES: {e}, using default values.")
        return 0, [0, 1]

    return first_die_no, visible_die_list

# ---- Lightweight test doubles -------------------------------------------------


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
        # Unused fields below but defined for compatibility with real config
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


class FakeKVTransferGroup:
    def __init__(self):
        self.registered = None
        self.finished = set()

    def register_kv_caches(self, caches, *_, **__):
        self.registered = caches

    def get_load_kv_failure_reqs(self):
        return None

    def clear_connector_metadata(self):
        pass


class FakeCacheEngine:
    def __init__(self, *_, **__):
        self.swap_in_calls = []
        self.swap_out_calls = []

    def swap_in(self, blocks):
        self.swap_in_calls.append(blocks)

    def swap_out(self, blocks):
        self.swap_out_calls.append(blocks)


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
        self.dummy_input = None

    def load_model(self, model):
        self.loaded = True

    def verify_and_prepare_inputs(self, *, input_ids, logits, sampling_metadata, spec_decode_metadata,
                                  num_prefills, num_decodes, chunk_next_tokens, chunk_next_indices):
        sampled = SimpleNamespace(sampled_token_ids=torch.zeros((input_ids.shape[0], 1), device=input_ids.device,
                                                                dtype=torch.int64),
                                  logprobs_tensors=None)
        return sampled, None, torch.tensor([0], device=input_ids.device)

    def propose(self, **kwargs):
        num_tokens = kwargs.get("num_tokens", 1)
        return torch.zeros((num_tokens, 1), dtype=torch.int64, device=kwargs["positions"].device)

    def prepare_dummy_input(self, input_ids):
        self.dummy_input = input_ids


class FakeRotaryEmb:
    @staticmethod
    def get_cos_sin(positions):
        return torch.zeros_like(positions), torch.zeros_like(positions)


class FakeLayer:
    def __init__(self):
        self.self_attn = SimpleNamespace(rotary_emb=FakeRotaryEmb())


class FakeModel(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.model = SimpleNamespace(start_layer=0, layers=[FakeLayer()])
        self.config = SimpleNamespace()
        self.hidden_size = hidden_size

    def get_input_embeddings(self, input_ids, mm_embeds=None):
        shape = (input_ids.shape[0], self.hidden_size)
        return torch.zeros(shape, device=input_ids.device, dtype=torch.float16)

    def forward(self, input_ids=None, positions=None, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        token_shape = inputs_embeds.shape[0] if inputs_embeds is not None else input_ids.shape[0]
        return torch.zeros((token_shape, self.hidden_size), device=positions.device, dtype=torch.float16)

    __call__ = forward


class RecordingDummyBuilder(DummyAttentionMetadataBuilder):
    def __init__(self, device):
        self.device = device
        self.calls: list[dict] = []
        self.runner = None
        self.mark_static_called = False

    def build(self, *, num_reqs, num_actual_tokens, max_query_len, **kwargs):
        call = dict(num_reqs=num_reqs, num_actual_tokens=num_actual_tokens,
                    max_query_len=max_query_len, **kwargs)
        self.calls.append(call)
        seq_lens = torch.full((num_reqs,), max_query_len, dtype=torch.int64,
                              device=self.device)
        metadata = SimpleNamespace(
            attn_state=getattr(self.runner, "attn_state", None),
            query_lens=seq_lens.clone(),
            seq_lens=seq_lens.clone(),
            slot_mapping=torch.zeros(num_actual_tokens, dtype=torch.int64,
                                     device=self.device),
            slot_indices=torch.zeros((num_actual_tokens, 2),
                                     dtype=torch.int64,
                                     device=self.device),
            decode=SimpleNamespace(
                seq_lens=seq_lens.clone(),
                block_table=torch.zeros((num_reqs, 1), dtype=torch.int64,
                                        device=self.device),
            ),
        )
        return metadata

    def build_dummy(self, *args, **kwargs):
        return SimpleNamespace()

    def mark_static_for_attn_metadata(self, *_, **__):
        self.mark_static_called = True

    def compute_logits(self, hidden_states, _):
        return torch.zeros((hidden_states.shape[0], 1), device=hidden_states.device, dtype=hidden_states.dtype)


# ---- Shared fixtures ---------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_gpu_runner_dependencies(monkeypatch):
    # Avoid heavy imports and device queries during init.
    import importlib

    gpu_runner_mod = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    utils_mod = importlib.import_module("vllm.model_executor.models.utils")
    if not hasattr(gpu_runner_mod, "set_cpu_offload_max_bytes"):
        setattr(gpu_runner_mod, "set_cpu_offload_max_bytes", utils_mod.set_cpu_offload_max_bytes)

    # Keep speculative dependencies lightweight.
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
    monkeypatch.setattr(
        gpu_runner_mod,
        "compute_encoder_budget",
        lambda model_config, scheduler_config, mm_registry: (0, 0),
    )
    monkeypatch.setattr(
        gpu_runner_mod,
        "set_cpu_offload_max_bytes",
        lambda *_: None,
    )
    dummy_props = SimpleNamespace(multi_processor_count=1)
    monkeypatch.setattr(
        gpu_runner_mod.torch.cuda,
        "get_device_properties",
        lambda device: dummy_props,
    )
    monkeypatch.setattr("vllm.utils.check_use_alibi", lambda *_: False)
    monkeypatch.setattr(gpu_runner_mod, "check_use_alibi", lambda *_: False)
    monkeypatch.setattr(
        "omni.adaptors.vllm.worker.npu_model_runner.torch.cuda.get_device_properties",
        lambda device: dummy_props,
    )
    monkeypatch.setattr("vllm.utils.supports_dynamo", lambda: True)
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.decode_h2d_trigger", lambda: None)
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.set_forward_context", lambda *_, **__: nullcontext())
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.has_kv_transfer_group", lambda: False)


@pytest.fixture
def kv_transfer_group(monkeypatch):
    group = FakeKVTransferGroup()
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.get_kv_transfer_group", lambda: group)
    return group


@pytest.fixture
def parallel_state(monkeypatch):
    import importlib

    class PPGroup:
        def __init__(self, is_last_rank=True):
            self.is_last_rank = is_last_rank

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
    # Swap out heavy sampler/drafter for quick fakes.
    monkeypatch.setattr("omni.adaptors.vllm.sample.sampler.AscendSamplerV1", FakeSampler)
    monkeypatch.setattr("omni.adaptors.vllm.sample.validator.SimpleValidator", FakeValidator)
    monkeypatch.setattr("omni.adaptors.vllm.sample.validator.SparseRejectionSamplerValidator", FakeValidator)
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.PostDrafter", FakeDrafter)


@pytest.fixture
def npu_device():
    first_die = parse_ascend_devices()[0]
    return torch.device(f"npu:{first_die}")


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


# ---- Tests -------------------------------------------------------------------


def test_init_spec_decode_and_default_gears(parallel_state, sampler_and_drafter, npu_device):
    # Ensure spec decode init derives batch sizes, gears, and drafter/sampler on last PP rank.
    model_cfg = DummyModelConfig()
    cache_cfg = DummyCacheConfig()
    sched_cfg = DummySchedulerConfig(max_num_batched_tokens=12, max_num_seqs=2)
    parallel_cfg = DummyParallelConfig()
    spec_cfg = DummySpeculativeConfig(num_speculative_tokens=2, enable_adaptive=False)
    npu_comp_cfg = DummyNPUCompilationConfig(level=CompilationLevel.NO_COMPILATION, decode_gear_list=None)

    vllm_cfg = make_vllm_config(
        model_config=model_cfg,
        cache_config=cache_cfg,
        scheduler_config=sched_cfg,
        parallel_config=parallel_cfg,
        npu_compilation_config=npu_comp_cfg,
        spec_config=spec_cfg,
        kv_role=None,
        additional_config={},
    )

    runner = NPUModelRunner(vllm_cfg, npu_device)

    assert runner.num_tokens_per_reqs_decode == 3
    assert runner.decode_max_num_tokens == runner.max_num_reqs * 3
    assert runner.decode_gear_list == [runner.max_batch_size]
    assert runner.max_batch_size == sched_cfg.max_num_seqs * (1 + spec_cfg.num_speculative_tokens)
    # Ensure sampler/drafter constructed on last PP rank.
    assert hasattr(runner, "sampler")
    assert hasattr(runner, "drafter")


def test_init_requires_gears_for_adaptive_spec(parallel_state, sampler_and_drafter, npu_device):
    # Adaptive speculative decoding without decode_gear_list should raise.
    model_cfg = DummyModelConfig()
    cache_cfg = DummyCacheConfig()
    sched_cfg = DummySchedulerConfig(max_num_batched_tokens=8, max_num_seqs=2)
    parallel_cfg = DummyParallelConfig()
    spec_cfg = DummySpeculativeConfig(num_speculative_tokens=1, enable_adaptive=True)
    npu_comp_cfg = DummyNPUCompilationConfig(level=CompilationLevel.NO_COMPILATION, decode_gear_list=None)

    vllm_cfg = make_vllm_config(
        model_config=model_cfg,
        cache_config=cache_cfg,
        scheduler_config=sched_cfg,
        parallel_config=parallel_cfg,
        npu_compilation_config=npu_comp_cfg,
        spec_config=spec_cfg,
        kv_role=None,
        additional_config={},
    )

    with pytest.raises(RuntimeError):
        NPUModelRunner(vllm_cfg, npu_device)


def test_init_hybrid_chunked_prefill_graph_mode(parallel_state, sampler_and_drafter, npu_device):
    # Hybrid chunked-prefill graph mode toggles max_batch_size to max_num_tokens.
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
    assert runner.max_batch_size == sched_cfg.max_num_batched_tokens


def test_init_training_flags_and_c8_kv_dtype(monkeypatch, parallel_state, sampler_and_drafter, npu_device):
    # Training save flags and C8 kv_cache dtype selection respect kv_role and mla flag.
    from omni.models.config_loader import loader

    model_cfg = DummyModelConfig(use_mla=False)
    cache_cfg = DummyCacheConfig()
    sched_cfg = DummySchedulerConfig()
    parallel_cfg = DummyParallelConfig()
    npu_comp_cfg = DummyNPUCompilationConfig(level=CompilationLevel.NO_COMPILATION)

    # enable training dump path
    monkeypatch.setenv("TRAINING_DATA_SAVE_PATH", "/tmp/training")
    monkeypatch.setenv("TRAINING_DATA_TOKEN_THRESHOLD", "16")
    monkeypatch.setattr(loader.model_extra_config.operator_opt_config, "enable_c8", True)

    vllm_cfg_consumer = make_vllm_config(
        model_config=model_cfg,
        cache_config=cache_cfg,
        scheduler_config=sched_cfg,
        parallel_config=parallel_cfg,
        npu_compilation_config=npu_comp_cfg,
        spec_config=None,
        kv_role="kv_consumer",
        additional_config={},
    )
    runner_consumer = NPUModelRunner(vllm_cfg_consumer, npu_device)
    assert runner_consumer.save_token_ids
    assert not runner_consumer.save_hidden_states
    assert runner_consumer.kv_cache_dtype == torch.int8

    vllm_cfg_producer = make_vllm_config(
        model_config=model_cfg,
        cache_config=cache_cfg,
        scheduler_config=sched_cfg,
        parallel_config=parallel_cfg,
        npu_compilation_config=npu_comp_cfg,
        spec_config=None,
        kv_role="kv_producer",
        additional_config={},
    )
    runner_producer = NPUModelRunner(vllm_cfg_producer, npu_device)
    assert runner_producer.save_hidden_states
    assert not runner_producer.save_token_ids


def test_initialize_kv_cache_registers_and_binds(
    monkeypatch, kv_transfer_group, parallel_state, sampler_and_drafter, npu_device
):
    # KV cache init allocates tensors, binds to runner/forward context, and registers with transfer group.
    model_cfg = DummyModelConfig()
    cache_cfg = DummyCacheConfig(block_size=2)
    sched_cfg = DummySchedulerConfig()
    parallel_cfg = DummyParallelConfig()
    npu_comp_cfg = DummyNPUCompilationConfig(level=CompilationLevel.NO_COMPILATION)

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

    bind_calls = []

    def binder(kv_caches, forward_context, runner_kv_caches):
        bind_calls.append(kv_caches)
        runner_kv_caches.extend(kv_caches.values())
        if forward_context is None:
            return
        for layer, cache in kv_caches.items():
            if layer not in forward_context:
                forward_context[layer] = SimpleNamespace()
            forward_context[layer].kv_cache = [cache]

    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.bind_kv_cache", binder)
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.has_kv_transfer_group", lambda: True)
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.has_kv_transfer_group", lambda: True)
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.CacheEngine", FakeCacheEngine)

    runner = NPUModelRunner(vllm_cfg, npu_device)
    backend = SimpleNamespace(
        get_kv_cache_shape=lambda num_blocks, block_size, num_kv_heads, head_size, *args: (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        ),
        init_kv_cache_each_layer=lambda shape, dtype, device, model_config, enable_graph: torch.zeros(shape, dtype=dtype, device="cpu"),
    )
    runner.attn_backends = [backend]
    runner.attn_metadata_builders = [SimpleNamespace()]
    monkeypatch.setattr(runner, "initialize_attn_backend", lambda cfg: None)

    spec = AttentionSpec(
        block_size=cache_cfg.block_size,
        num_kv_heads=1,
        head_size=model_cfg.head_size,
        dtype=torch.float16,
        use_mla=model_cfg.use_mla,
    )
    tensor_size = spec.page_size_bytes * 2
    layer_name = "layers.0.attn"
    kv_cfg = KVCacheConfig(
        num_blocks=2,
        tensors={layer_name: KVCacheTensor(size=tensor_size)},
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)],
    )

    runner.initialize_kv_cache(kv_cfg)

    assert runner.kv_caches, "kv_caches should be bound to runner"
    assert kv_transfer_group.registered is not None
    assert bind_calls, "bind_kv_cache should be invoked when NO_NPU_MOCK is unset"


def test_prepare_kv_cache_swaps(monkeypatch, parallel_state, sampler_and_drafter, npu_device):
    # Swap in/out hooks delegate to cache engine when blocks are provided.
    model_cfg = DummyModelConfig()
    cache_cfg = DummyCacheConfig()
    sched_cfg = DummySchedulerConfig()
    parallel_cfg = DummyParallelConfig()
    npu_comp_cfg = DummyNPUCompilationConfig()

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
    engine = FakeCacheEngine()
    runner.cache_engine = engine

    scheduler_output = SimpleNamespace(blocks_to_swap_in=[[1, 2]], blocks_to_swap_out=[[3]])

    runner._prepare_kv_cache(scheduler_output)
    assert engine.swap_in_calls[0] == [[1, 2]]
    assert engine.swap_out_calls[0] == [[3]]


def test_get_max_token_num_allreduce(monkeypatch, parallel_state, sampler_and_drafter, npu_device):
    # _get_max_token_num picks global max via all_reduce when DP enabled.
    model_cfg = DummyModelConfig()
    cache_cfg = DummyCacheConfig()
    sched_cfg = DummySchedulerConfig()
    parallel_cfg = DummyParallelConfig(dp=2)
    npu_comp_cfg = DummyNPUCompilationConfig()

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
    runner.decode_gear_list = [2, 6]

    monkeypatch.setattr(
        "omni.adaptors.vllm.worker.npu_model_runner.dist.all_reduce",
        lambda tensor, group, op: tensor.fill_(6),
    )

    result = runner._get_max_token_num(is_enable_dp=True, num_tokens=4)
    assert result == 6


def test_get_closest_gear_error(parallel_state, sampler_and_drafter, npu_device):
    # _get_closest_gear raises when requested token count exceeds max gear.
    model_cfg = DummyModelConfig()
    cache_cfg = DummyCacheConfig()
    sched_cfg = DummySchedulerConfig()
    parallel_cfg = DummyParallelConfig()
    npu_comp_cfg = DummyNPUCompilationConfig()

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
    runner.decode_gear_list = [1, 2]

    with pytest.raises(ValueError):
        runner._get_closest_gear(5)


def test_simple_prepare_inputs_calls_flashattn(monkeypatch, parallel_state, sampler_and_drafter, npu_device):
    # _simple_prepare_inputs hits flashattn advance when accepted_num is provided.
    model_cfg = DummyModelConfig()
    cache_cfg = DummyCacheConfig(block_size=2)
    sched_cfg = DummySchedulerConfig(max_num_seqs=1)
    parallel_cfg = DummyParallelConfig()
    npu_comp_cfg = DummyNPUCompilationConfig()

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
    # initialize kv cache to create block tables and input batch
    backend = SimpleNamespace(
        get_kv_cache_shape=lambda num_blocks, block_size, num_kv_heads, head_size, *args: (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        ),
        init_kv_cache_each_layer=lambda shape, dtype, device, model_config, enable_graph: torch.zeros(shape, dtype=dtype, device="cpu"),
    )
    runner.attn_backends = [backend]
    runner.attn_metadata_builders = [SimpleNamespace(mark_static_for_attn_metadata=lambda *_: None)]
    monkeypatch.setattr(runner, "initialize_attn_backend", lambda cfg: None)

    spec = AttentionSpec(
        block_size=cache_cfg.block_size,
        num_kv_heads=1,
        head_size=model_cfg.head_size,
        dtype=torch.float16,
        use_mla=False,
    )
    tensor_size = spec.page_size_bytes * 1
    layer_name = "layers.0.attn"
    kv_cfg = KVCacheConfig(
        num_blocks=1,
        tensors={layer_name: KVCacheTensor(size=tensor_size)},
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)],
    )
    runner.initialize_kv_cache(kv_cfg)
    runner.input_batch.req_id_to_index = {"req0": 0}

    total_tokens = 2
    attn_metadata = {
        layer_name: SimpleNamespace(
            attn_state=None,
            seq_lens=torch.ones(total_tokens, dtype=torch.int64, device=npu_device),
            slot_mapping=torch.zeros(runner.max_num_tokens, dtype=torch.int64, device=npu_device),
        )
    }
    runner.model = FakeModel(hidden_size=runner.hidden_size).to(npu_device)

    calls = []
    monkeypatch.setattr(
        "torch_npu.npu_advance_step_flashattn",
        lambda **kwargs: calls.append(kwargs),
    )

    positions = torch.zeros(total_tokens, dtype=torch.int64, device=npu_device)
    cached_token = torch.zeros((1, 1), dtype=torch.float16, device=npu_device)
    cached_spec = torch.ones((1, 1), dtype=torch.float16, device=npu_device)
    accepted_num = torch.zeros(1, dtype=torch.int64, device=npu_device)

    runner._simple_prepare_inputs(attn_metadata, positions, cached_token, cached_spec, accepted_num)
    assert calls, "torch_npu.npu_advance_step_flashattn should be invoked when accepted_num is provided"


def test_get_pad_size_respects_tp(monkeypatch):
    # _get_pad_size accounts for tensor parallel world size.
    monkeypatch.setattr("omni.adaptors.vllm.worker.npu_model_runner.get_tensor_model_parallel_world_size", lambda: 2)
    assert _get_pad_size(3) == 1


@pytest.mark.parametrize(
    "num_scheduled_tokens,num_computed_tokens,decode_gears,expected_state,expected_pad,tp_world_size",
    [
        ({"req0": 1, "req1": 1}, [0, 0], [4, 8], AscendAttentionState.DecodeOnly, 2, 1),
        ({"req0": 2, "req1": 1}, [0, 0], [1], AscendAttentionState.PrefillNoCache, 1, 2),
    ],
)
def test_prepare_inputs_branches(
    monkeypatch,
    parallel_state,
    sampler_and_drafter,
    npu_device,
    num_scheduled_tokens,
    num_computed_tokens,
    decode_gears,
    expected_state,
    expected_pad,
    tp_world_size,
):
    # _prepare_inputs handles decode vs prefill paths, padding, and metadata builder hooks.
    monkeypatch.setattr(
        "omni.adaptors.vllm.worker.npu_model_runner.get_tensor_model_parallel_world_size",
        lambda: tp_world_size,
    )
    model_cfg = DummyModelConfig(max_model_len=8)
    model_cfg.disable_cascade_attn = True
    cache_cfg = DummyCacheConfig(block_size=2)
    sched_cfg = DummySchedulerConfig(
        max_num_batched_tokens=8, max_num_seqs=len(num_scheduled_tokens)
    )
    parallel_cfg = DummyParallelConfig()
    npu_comp_cfg = DummyNPUCompilationConfig(
        level=CompilationLevel.PIECEWISE, decode_gear_list=decode_gears
    )

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
    runner.cascade_attn_enabled = False

    backend = SimpleNamespace(
        get_kv_cache_shape=lambda num_blocks, block_size, num_kv_heads, head_size, *args: (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        ),
        init_kv_cache_each_layer=lambda shape, dtype, device, model_config, enable_graph: torch.zeros(
            shape, dtype=dtype, device="cpu"
        ),
    )
    runner.attn_backends = [backend]
    runner.attn_metadata_builders = [SimpleNamespace()]
    monkeypatch.setattr(runner, "initialize_attn_backend", lambda cfg: None)

    layer_name = "layers.0.attn"
    spec = AttentionSpec(
        block_size=cache_cfg.block_size,
        num_kv_heads=1,
        head_size=model_cfg.head_size,
        dtype=torch.float16,
        use_mla=model_cfg.use_mla,
    )
    kv_cfg = KVCacheConfig(
        num_blocks=2,
        tensors={layer_name: KVCacheTensor(size=spec.page_size_bytes)},
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)],
    )
    runner.initialize_kv_cache(kv_cfg)

    builder = RecordingDummyBuilder(npu_device)
    builder.runner = runner
    runner.attn_metadata_builders = [builder]

    req_ids = list(num_scheduled_tokens.keys())
    runner.input_batch._req_ids = req_ids
    runner.input_batch.req_id_to_index = {rid: idx for idx, rid in enumerate(req_ids)}
    num_reqs = len(req_ids)
    runner.input_batch.num_computed_tokens_cpu[:num_reqs] = num_computed_tokens
    for idx, req_id in enumerate(req_ids):
        total_for_req = num_computed_tokens[idx] + num_scheduled_tokens[req_id]
        runner.input_batch.token_ids_cpu[idx, :total_for_req] = (
            torch.arange(total_for_req, dtype=torch.int64).numpy() + (idx + 1) * 10
        )

    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[0],
    )

    attn_metadata, graph_pad_size, sample_indices, positions, spec_decode_metadata = runner._prepare_inputs(
        scheduler_output
    )

    assert runner.attn_state == expected_state
    assert graph_pad_size == expected_pad
    assert spec_decode_metadata is None
    assert builder.calls and builder.calls[0]["graph_pad_size"] == expected_pad
    assert builder.mark_static_called is (expected_state == AscendAttentionState.DecodeOnly)

    total_tokens = scheduler_output.total_num_scheduled_tokens
    expected_positions = []
    expected_input_ids = []
    cu_tokens = []
    cum = 0
    for idx, rid in enumerate(req_ids):
        for offset in range(num_scheduled_tokens[rid]):
            position = num_computed_tokens[idx] + offset
            expected_positions.append(position)
            expected_input_ids.append((idx + 1) * 10 + position)
        cum += num_scheduled_tokens[rid]
        cu_tokens.append(cum)
    assert positions.cpu().tolist()[:total_tokens] == expected_positions
    assert positions.numel() == total_tokens + expected_pad
    assert runner.input_ids[:total_tokens].cpu().tolist() == expected_input_ids
    assert sample_indices.cpu().tolist() == [tok - 1 for tok in cu_tokens]

    first_metadata = attn_metadata[layer_name]
    assert first_metadata.attn_state == expected_state
    assert first_metadata.decode.seq_lens.shape[0] == num_reqs
