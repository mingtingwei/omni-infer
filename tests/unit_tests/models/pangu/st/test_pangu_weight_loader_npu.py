import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn import Parameter

torch_npu = pytest.importorskip("torch_npu")

import omni.models.pangu.pangu_pro_moe_v2.pangu_moe_v2 as pangu_mod


class _DummyGroup:
    def __init__(self, world_size=1, rank_in_group=0):
        self.world_size = world_size
        self.rank_in_group = rank_in_group


class _FusedMoEStub:
    @staticmethod
    def make_expert_params_mapping(*args, **kwargs):
        return []


def _fake_weight_loader(param, weight, *_args, **_kwargs):
    if isinstance(weight, torch.Tensor):
        param.data.copy_(weight.to(device=param.device, dtype=param.dtype))
    else:
        param.data.copy_(torch.tensor(weight, device=param.device, dtype=param.dtype))


def test_load_weights_on_npu(npu_device, monkeypatch):
    monkeypatch.setattr(pangu_mod, "get_tp_group", lambda: _DummyGroup())
    monkeypatch.setattr(pangu_mod, "FusedMoE", _FusedMoEStub)
    monkeypatch.setattr(pangu_mod, "default_weight_loader", _fake_weight_loader)
    monkeypatch.setattr(pangu_mod, "sharded_weight_loader", lambda dim: _fake_weight_loader)
    monkeypatch.setattr(pangu_mod, "is_pp_missing_parameter", lambda name, self: False)
    monkeypatch.setattr(pangu_mod, "logger", SimpleNamespace(warning_once=Mock()))

    torch.manual_seed(1)
    torch_npu.npu.manual_seed(1)

    model = pangu_mod.PanguProMoEV2ForCausalLM.__new__(pangu_mod.PanguProMoEV2ForCausalLM)
    model.config = types.SimpleNamespace(num_experts=0)
    model.model = types.SimpleNamespace(end_layer=1)

    key_scale = Parameter(torch.zeros((1,), device=npu_device, dtype=torch.float16))
    key_scale.weight_loader = Mock(side_effect=_fake_weight_loader)
    value_scale = Parameter(torch.zeros((1,), device=npu_device, dtype=torch.float16))
    value_scale.weight_loader = Mock(side_effect=_fake_weight_loader)
    kv_scale = Parameter(torch.zeros((1,), device=npu_device, dtype=torch.float16))
    kv_scale.weight_loader = Mock(side_effect=_fake_weight_loader)
    sink_key = Parameter(torch.zeros((2, 2), device=npu_device, dtype=torch.float16))
    sink_value = Parameter(torch.zeros((2, 2), device=npu_device, dtype=torch.float16))

    params = {
        "model.layers.0.self_attn.attn.key_antiquant_scale": key_scale,
        "model.layers.0.self_attn.attn.value_antiquant_scale": value_scale,
        "model.layers.0.self_attn.attn.kv_scale": kv_scale,
        "model.layers.0.self_attn.param_sink_key": sink_key,
        "model.layers.0.self_attn.param_sink_value": sink_value,
    }

    def _named_parameters():
        for name, param in params.items():
            yield name, param

    model.named_parameters = _named_parameters  # type: ignore[assignment]

    weights = [
        ("model.layers.0.self_attn.k_proj.kv_cache_scale", torch.randn((1,), device=npu_device, dtype=torch.float16)),
        ("model.layers.0.self_attn.v_proj.kv_cache_scale", torch.randn((1,), device=npu_device, dtype=torch.float16)),
        ("model.layers.0.self_attn.kv_scale", torch.randn((1,), device=npu_device, dtype=torch.float16)),
        ("model.layers.0.self_attn.param_sink_key", torch.randn((2, 2), device=npu_device, dtype=torch.float16)),
        ("model.layers.0.self_attn.param_sink_value", torch.randn((2, 2), device=npu_device, dtype=torch.float16)),
    ]

    loaded = pangu_mod.PanguProMoEV2ForCausalLM.load_weights(model, weights)

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

    assert sink_key.device == npu_device
    assert sink_value.device == npu_device
    assert key_scale.weight_loader.call_count >= 1
    assert value_scale.weight_loader.call_count >= 1
    assert kv_scale.weight_loader.call_count >= 1

    print(f"NPU_DEVICE={npu_device}")
    print("sink params loaded")
    print("WEIGHT_LOADER_NPU_OK")
