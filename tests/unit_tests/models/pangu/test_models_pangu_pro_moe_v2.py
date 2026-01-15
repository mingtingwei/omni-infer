import contextlib
import importlib
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn import Parameter


class _DummyGroup:
    def __init__(self, world_size=1, rank_in_group=0):
        self.world_size = world_size
        self.rank_in_group = rank_in_group


class _FusedMoEStub:
    @staticmethod
    def make_expert_params_mapping(*args, **kwargs):
        return []


def _install_torch_npu_stub(stack: contextlib.ExitStack) -> None:
    try:
        import torch_npu  # noqa: F401
        return
    except Exception:
        stub = types.ModuleType("torch_npu")
        stub.npu_format_cast = lambda tensor, *_args, **_kwargs: tensor  # type: ignore[attr-defined]
        stub.npu_prefetch = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
        stack.enter_context(patch.dict(sys.modules, {"torch_npu": stub}))

        if not hasattr(torch, "npu"):
            npu_ns = types.SimpleNamespace(config=types.SimpleNamespace(allow_internal_format=False))
            stack.enter_context(patch.object(torch, "npu", npu_ns, create=True))
        else:
            if not hasattr(torch.npu, "config"):  # type: ignore[attr-defined]
                cfg_ns = types.SimpleNamespace(allow_internal_format=False)
                stack.enter_context(patch.object(torch.npu, "config", cfg_ns, create=True))  # type: ignore[attr-defined]
            stack.enter_context(
                patch.object(torch.npu.config, "allow_internal_format", False, create=True)  # type: ignore[attr-defined]
            )


def _install_torchair_stub(stack: contextlib.ExitStack) -> None:
    try:
        import torchair  # noqa: F401
        return
    except Exception:
        stub = types.ModuleType("torchair")
        stub.scope = types.SimpleNamespace(
            npu_stream_switch=lambda *_args, **_kwargs: contextlib.nullcontext(),
            npu_wait_tensor=lambda tensor, *_args, **_kwargs: tensor,
        )
        stub.ops = types.SimpleNamespace()
        stack.enter_context(patch.dict(sys.modules, {"torchair": stub}))


class TestPanguProMoEV2LoadWeights(unittest.TestCase):
    _MODULE = "omni.models.pangu.pangu_pro_moe_v2.pangu_moe_v2"

    def setUp(self):
        super().setUp()
        self._stack = contextlib.ExitStack()
        _install_torch_npu_stub(self._stack)
        _install_torchair_stub(self._stack)
        self._module_preexisted = self._MODULE in sys.modules
        self.M = importlib.import_module(self._MODULE)

        def _maybe_unimport():
            if not self._module_preexisted:
                sys.modules.pop(self._MODULE, None)

        self._stack.callback(_maybe_unimport)

    def tearDown(self):
        self._stack.close()
        super().tearDown()

    def test_load_weights_remaps_kv_scales_and_sink_params(self):
        m = self.M.PanguProMoEV2ForCausalLM.__new__(self.M.PanguProMoEV2ForCausalLM)
        m.config = types.SimpleNamespace(num_experts=0)
        m.model = types.SimpleNamespace(end_layer=1)

        key_scale = Parameter(torch.empty(1))
        key_scale.weight_loader = Mock()
        value_scale = Parameter(torch.empty(1))
        value_scale.weight_loader = Mock()
        kv_scale = Parameter(torch.empty(1))
        kv_scale.weight_loader = Mock()
        sink_key = Parameter(torch.empty(1))
        sink_value = Parameter(torch.empty(1))

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

        m.named_parameters = _named_parameters  # type: ignore[assignment]

        sharded_loader = Mock()

        def _sharded_weight_loader(dim):
            self.assertEqual(dim, -2)
            return sharded_loader

        def _set_weight_attrs(param, attrs):
            for key, val in attrs.items():
                setattr(param, key, val)

        with patch.object(self.M, "get_tp_group", return_value=_DummyGroup(), create=True), \
             patch.object(self.M, "FusedMoE", _FusedMoEStub, create=True), \
             patch.object(self.M, "default_weight_loader", Mock(), create=True), \
             patch.object(self.M, "sharded_weight_loader", side_effect=_sharded_weight_loader, create=True), \
             patch.object(self.M, "is_pp_missing_parameter", Mock(return_value=False), create=True), \
             patch.object(self.M, "set_weight_attrs", side_effect=_set_weight_attrs, create=True), \
             patch.object(self.M, "logger", types.SimpleNamespace(warning_once=Mock()), create=True):
            weights = [
                ("model.layers.0.self_attn.k_proj.kv_cache_scale", torch.tensor([2.0])),
                ("model.layers.0.self_attn.v_proj.kv_cache_scale", torch.tensor([3.0])),
                ("model.layers.0.self_attn.kv_scale", torch.tensor([1.0])),
                ("model.layers.0.self_attn.param_sink_key", torch.randn(2, 2)),
                ("model.layers.0.self_attn.param_sink_value", torch.randn(2, 2)),
            ]
            loaded = self.M.PanguProMoEV2ForCausalLM.load_weights(m, weights)

        self.assertIn("model.layers.0.self_attn.attn.key_antiquant_scale", loaded)
        self.assertIn("model.layers.0.self_attn.attn.value_antiquant_scale", loaded)
        self.assertIn("model.layers.0.self_attn.attn.kv_scale", loaded)
        self.assertIn("model.layers.0.self_attn.param_sink_key", loaded)
        self.assertIn("model.layers.0.self_attn.param_sink_value", loaded)
        self.assertTrue(hasattr(sink_key, "is_2_dims"))
        self.assertTrue(hasattr(sink_value, "is_2_dims"))
        self.assertTrue(hasattr(key_scale, "is_2_dims"))
        self.assertTrue(hasattr(value_scale, "is_2_dims"))
        self.assertTrue(hasattr(kv_scale, "is_2_dims"))
        self.assertGreaterEqual(sharded_loader.call_count, 2)

    def test_moe_block_forward_selects_prefill_and_decode_paths(self):
        block = self.M.PanguProMoEV2MoEBlock.__new__(self.M.PanguProMoEV2MoEBlock)
        block.is_init_gate = True
        block._forward_prefill_norm = Mock(return_value="prefill")
        block._forward_decode_norm = Mock(return_value="decode")

        hidden_states = torch.zeros(1, 1)
        residual = torch.zeros(1, 1)

        attn_prefill = types.SimpleNamespace(attn_state=object())
        attn_decode = types.SimpleNamespace(attn_state=self.M.AscendAttentionState.DecodeOnly)

        out = block.forward(hidden_states, residual, attn_prefill, layer_id=0, is_hybrid_chunked_prefill_graph_mode=False)
        self.assertEqual(out, "prefill")

        out = block.forward(hidden_states, residual, attn_decode, layer_id=0, is_hybrid_chunked_prefill_graph_mode=False)
        self.assertEqual(out, "decode")

        out = block.forward(hidden_states, residual, attn_prefill, layer_id=0, is_hybrid_chunked_prefill_graph_mode=True)
        self.assertEqual(out, "decode")

        out = block.forward(hidden_states, residual, None, layer_id=0, is_hybrid_chunked_prefill_graph_mode=False)
        self.assertEqual(out, "prefill")


if __name__ == "__main__":
    unittest.main()
