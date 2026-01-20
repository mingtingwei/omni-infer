import contextlib
import importlib
import sys
import types
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture()
def attention_backend_module(monkeypatch):
    def _install_module(name):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    sys.modules.pop("omni.layers.attention.backend.attention", None)
    # Ensure module cleanup is reversible for downstream tests.
    monkeypatch.delitem(sys.modules, "omni.layers", raising=False)

    torch_npu_stub = _install_module("torch_npu")
    torch_npu_stub.npu_scatter_pa_kv_cache = Mock()
    torch_npu_stub.npu_fused_infer_attention_score_v2 = Mock(return_value=(torch.zeros(1),))

    torchair_stub = _install_module("torchair")
    torchair_stub.ops = types.SimpleNamespace(npu_fused_infer_attention_score_v2=Mock(return_value=(torch.zeros(1),)))
    torchair_stub.scope = types.SimpleNamespace(
        npu_stream_switch=lambda *_args, **_kwargs: contextlib.nullcontext(),
        npu_wait_tensor=lambda tensor, *_args, **_kwargs: tensor,
    )

    _install_module("vllm")
    _install_module("vllm.attention")
    _install_module("vllm.attention.backends")
    _install_module("vllm.model_executor")
    _install_module("vllm.model_executor.layers")
    _install_module("vllm.model_executor.models")

    abstract_mod = _install_module("vllm.attention.backends.abstract")
    abstract_mod.AttentionBackend = type("AttentionBackend", (), {})
    abstract_mod.AttentionImpl = type("AttentionImpl", (), {})
    abstract_mod.AttentionLayer = type("AttentionLayer", (), {})
    abstract_mod.AttentionType = type("AttentionType", (), {"DECODER": "decoder"})

    utils_mod = _install_module("vllm.attention.backends.utils")
    utils_mod.PAD_SLOT_ID = -1
    utils_mod.CommonAttentionState = type("CommonAttentionState", (), {})

    rot_mod = _install_module("vllm.model_executor.layers.rotary_embedding")
    rot_mod.DynamicNTKScalingRotaryEmbedding = type("DynamicNTKScalingRotaryEmbedding", (), {})

    model_utils_mod = _install_module("vllm.model_executor.models.utils")
    model_utils_mod.extract_layer_index = lambda *_args, **_kwargs: 0

    forward_mod = _install_module("vllm.forward_context")
    forward_mod.ForwardContext = type("ForwardContext", (), {})
    forward_mod.get_forward_context = lambda: types.SimpleNamespace(attn_metadata=None, virtual_engine=0)

    utils_core_mod = _install_module("vllm.utils")
    utils_core_mod.direct_register_custom_op = lambda *_args, **_kwargs: None
    utils_core_mod.supports_dynamo = lambda: False
    utils_core_mod.is_pin_memory_available = lambda: False

    dist_mod = _install_module("vllm.distributed")
    dist_mod.get_pp_group = lambda: types.SimpleNamespace(world_size=1)

    _install_module("vllm.v1")
    _install_module("vllm.v1.core")
    _install_module("vllm.v1.core.sched")
    sched_mod = _install_module("vllm.v1.core.sched.output")
    sched_mod.SchedulerOutput = type("SchedulerOutput", (), {})
    _install_module("vllm.v1.worker")
    input_mod = _install_module("vllm.v1.worker.gpu_input_batch")
    input_mod.InputBatch = type("InputBatch", (), {})
    kv_mod = _install_module("vllm.v1.kv_cache_interface")
    kv_mod.AttentionSpec = type("AttentionSpec", (), {})
    block_mod = _install_module("vllm.v1.worker.block_table")
    block_mod.BlockTable = type("BlockTable", (), {})

    platforms_mod = _install_module("vllm.platforms")
    platforms_mod.current_platform = types.SimpleNamespace(device_type="cpu")

    config_mod = _install_module("vllm.config")

    class CompilationLevel:
        NO_COMPILATION = 0

    config_mod.CompilationLevel = CompilationLevel

    class _DummySchedulerConfig:
        enable_chunked_prefill = False

    class _DummyNpuCompilationConfig:
        def __init__(self, level=CompilationLevel.NO_COMPILATION):
            self.level = level

    class _DummyVllmConfig:
        def __init__(self):
            self.npu_compilation_config = _DummyNpuCompilationConfig()
            self.scheduler_config = _DummySchedulerConfig()
            self.kv_transfer_config = None
            self.additional_config = {}

    config_mod.get_current_vllm_config = lambda: _DummyVllmConfig()

    # Avoid patch_all side effects when omni.layers __init__ imports model_patch.
    model_patch_stub = _install_module("omni.adaptors.vllm.patches.model_patch")
    model_patch_stub.patch_all = lambda: None

    rotary_stub = _install_module("omni.layers.rotary_embedding")
    rotary_stub.QwenMRotaryEmbedding = type("QwenMRotaryEmbedding", (), {})
    rotary_stub.MRotaryEmbeddingInterleaved = type("MRotaryEmbeddingInterleaved", (), {})

    config_loader_stub = _install_module("omni.models.config_loader.loader")
    config_loader_stub.model_extra_config = types.SimpleNamespace(
        operator_opt_config=types.SimpleNamespace(
            enable_c8=False,
            use_tnd_pa=False,
            use_omni_cache=False,
        ),
        task_config=types.SimpleNamespace(decode_gear_list=[]),
    )

    module = importlib.import_module("omni.layers.attention.backend.attention")
    module.AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE = torch.zeros((1, 1), dtype=torch.bool)
    module.model_extra_config = types.SimpleNamespace(
        operator_opt_config=types.SimpleNamespace(use_tnd_pa=False, enable_c8=False)
    )
    return module


def test_forward_sink_attention_pads_block_tables_and_scatter(attention_backend_module, monkeypatch):
    module = attention_backend_module

    captured = {}
    scatter_calls = []

    orig_ones = module.torch.ones

    def _safe_ones(*args, **kwargs):
        kwargs.pop("device", None)
        return orig_ones(*args, **kwargs)

    monkeypatch.setattr(module.torch, "ones", _safe_ones)

    def _fake_scatter(_key, _value, _k_cache, _v_cache, slots):
        scatter_calls.append(slots.clone() if torch.is_tensor(slots) else slots)

    def _fake_fused(*_args, **kwargs):
        captured["block_table"] = kwargs.get("block_table")
        captured["actual_seq_kvlen"] = kwargs.get("actual_seq_kvlen")
        return (torch.zeros((1, 2, 16)),)

    module.torch_npu.npu_scatter_pa_kv_cache = _fake_scatter
    module.torch_npu.npu_fused_infer_attention_score_v2 = _fake_fused

    impl = module.AscendAttentionBackendImpl(
        num_heads=2,
        head_size=16,
        scale=1.0,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    layer = types.SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)

    query = torch.zeros((1, 32))
    key = torch.zeros((1, 32))
    value = torch.zeros((1, 32))

    kv_cache_key = torch.zeros((1, 2, 1, 128, 16))
    kv_cache_value = torch.zeros((1, 2, 1, 128, 16))

    attn_metadata = types.SimpleNamespace(
        attn_state=module.AscendAttentionState.PrefillNoCache,
        slot_mapping=torch.tensor([1], dtype=torch.int32),
        block_tables=torch.tensor([[5, 6]], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        query_lens=torch.tensor([1], dtype=torch.int32),
    )

    sink_key = torch.zeros((128, 2, 16))
    sink_value = torch.zeros((128, 2, 16))

    output = impl.forward_sink_attention(
        layer,
        query,
        key,
        value,
        (kv_cache_key, kv_cache_value),
        attn_metadata,
        sink_key=sink_key,
        sink_value=sink_value,
    )

    assert output.shape == (1, 32)
    assert captured["block_table"].shape[1] == attn_metadata.block_tables.shape[1] + 1
    assert torch.equal(captured["block_table"][:, 0], torch.zeros(1, dtype=attn_metadata.block_tables.dtype))
    assert torch.equal(captured["actual_seq_kvlen"], attn_metadata.seq_lens + 128)

    expected_slots = torch.arange(0, 128, dtype=torch.int32)
    assert any(torch.equal(slots, expected_slots) for slots in scatter_calls)
