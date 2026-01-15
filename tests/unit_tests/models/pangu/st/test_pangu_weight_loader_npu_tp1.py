import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn import Parameter

torch_npu = pytest.importorskip("torch_npu")

import vllm.distributed.parallel_state as parallel_state

import omni.models.pangu.pangu_pro_moe_v2.pangu_moe_v2 as pangu_mod
from omni.layers.linear import AscendMergedColumnParallelLinear


class _DummyGroup:
    def __init__(self, world_size=1, rank_in_group=0):
        self.world_size = world_size
        self.rank_in_group = rank_in_group


class _FusedMoEStub:
    @staticmethod
    def make_expert_params_mapping(*args, **kwargs):
        return []


@pytest.mark.npu
def test_load_weights_on_npu_real_loader_tp1(npu_device, monkeypatch):
    tp_group = _DummyGroup(world_size=1, rank_in_group=0)
    monkeypatch.setattr(pangu_mod, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(pangu_mod, "FusedMoE", _FusedMoEStub)
    monkeypatch.setattr(pangu_mod, "is_pp_missing_parameter", lambda name, self: False)
    monkeypatch.setattr(pangu_mod, "logger", SimpleNamespace(warning_once=Mock()))

    model = pangu_mod.PanguProMoEV2ForCausalLM.__new__(pangu_mod.PanguProMoEV2ForCausalLM)
    model.config = types.SimpleNamespace(num_experts=0)
    model.model = types.SimpleNamespace(end_layer=1)

    gate_up_module = AscendMergedColumnParallelLinear(
        input_size=2,
        output_sizes=[2, 2],
        bias=False,
        tp_size=tp_group.world_size,
        tp_rank=tp_group.rank_in_group,
        params_dtype=torch.float16,
    ).to(npu_device)
    gate_up_weight = gate_up_module.weight

    key_scale = Parameter(torch.zeros((1,), device=npu_device, dtype=torch.float16))
    value_scale = Parameter(torch.zeros((1,), device=npu_device, dtype=torch.float16))
    kv_scale = Parameter(torch.zeros((1,), device=npu_device, dtype=torch.float16))
    sink_key = Parameter(torch.zeros((2, 2), device=npu_device, dtype=torch.float16))
    sink_value = Parameter(torch.zeros((2, 2), device=npu_device, dtype=torch.float16))

    params = {
        "model.layers.0.self_attn.attn.key_antiquant_scale": key_scale,
        "model.layers.0.self_attn.attn.value_antiquant_scale": value_scale,
        "model.layers.0.self_attn.attn.kv_scale": kv_scale,
        "model.layers.0.self_attn.param_sink_key": sink_key,
        "model.layers.0.self_attn.param_sink_value": sink_value,
        "model.layers.0.mlp.gate_up_proj.weight": gate_up_weight,
    }

    def _named_parameters():
        yield from params.items()

    model.named_parameters = _named_parameters  # type: ignore[assignment]

    w_gate = torch.tensor([[1.0, 2.0], [3.0, 4.0]],
                          device=npu_device,
                          dtype=torch.float16)
    w_up = torch.tensor([[5.0, 6.0], [7.0, 8.0]],
                        device=npu_device,
                        dtype=torch.float16)
    w_k = torch.tensor([1.5], device=npu_device, dtype=torch.float16)
    w_v = torch.tensor([2.5], device=npu_device, dtype=torch.float16)
    w_kv = torch.tensor([3.5], device=npu_device, dtype=torch.float16)
    w_sink_k = torch.arange(4, device=npu_device, dtype=torch.float16).reshape(2, 2)
    w_sink_v = (torch.arange(4, device=npu_device, dtype=torch.float16) + 10).reshape(2, 2)

    weights = [
        ("model.layers.0.mlp.gate_proj.weight", w_gate),
        ("model.layers.0.mlp.up_proj.weight", w_up),
        ("model.layers.0.self_attn.k_proj.kv_cache_scale", w_k),
        ("model.layers.0.self_attn.v_proj.kv_cache_scale", w_v),
        ("model.layers.0.self_attn.kv_scale", w_kv),
        ("model.layers.0.self_attn.param_sink_key", w_sink_k),
        ("model.layers.0.self_attn.param_sink_value", w_sink_v),
    ]

    loaded = pangu_mod.PanguProMoEV2ForCausalLM.load_weights(model, weights)

    assert "model.layers.0.mlp.gate_up_proj.weight" in loaded
    assert "model.layers.0.self_attn.attn.key_antiquant_scale" in loaded
    assert "model.layers.0.self_attn.attn.value_antiquant_scale" in loaded
    assert "model.layers.0.self_attn.attn.kv_scale" in loaded
    assert "model.layers.0.self_attn.param_sink_key" in loaded
    assert "model.layers.0.self_attn.param_sink_value" in loaded

    assert getattr(sink_key, "is_2_dims", False)
    assert getattr(sink_value, "is_2_dims", False)
    assert getattr(key_scale, "is_2_dims", False)
    assert getattr(value_scale, "is_2_dims", False)
    assert getattr(kv_scale, "is_2_dims", False)

    assert torch.equal(key_scale.data, w_k.to(dtype=key_scale.dtype))
    assert torch.equal(value_scale.data, w_v.to(dtype=value_scale.dtype))
    assert torch.equal(kv_scale.data, w_kv.to(dtype=kv_scale.dtype))
    assert torch.equal(sink_key.data, w_sink_k.to(dtype=sink_key.dtype))
    assert torch.equal(sink_value.data, w_sink_v.to(dtype=sink_value.dtype))
    assert torch.equal(gate_up_weight.data[:2, :], w_gate.to(dtype=gate_up_weight.dtype))
    assert torch.equal(gate_up_weight.data[2:, :], w_up.to(dtype=gate_up_weight.dtype))

    print(f"NPU_DEVICE={npu_device}")
    print("sink params loaded")
    print("WEIGHT_LOADER_NPU_OK")
