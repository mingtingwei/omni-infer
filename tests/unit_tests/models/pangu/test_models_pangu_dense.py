import os
import sys
import types
import contextlib
import pytest
import unittest
from unittest import mock
from unittest.mock import Mock, patch, MagicMock
import importlib

import torch
from torch import nn
from torch.nn import Parameter

# --- Make import robust in non-NPU unit-test envs (isolated, best-effort stubs) ---
#
# IMPORTANT: Do NOT mutate process-global state permanently at import time.
# This module uses pytest's xunit-style hooks to:
#   1) (optionally) provide light stubs for torch_npu / torch.npu *only* for importing the SUT,
#   2) restore all global state in teardown_module so other test files are not affected.
#
# This keeps the UT highly isolated and prevents cross-test contamination.
PANGU_DENSE = None  # will be imported in setup_module()

_ISOLATION_STATE = {}


def setup_module():
    global PANGU_DENSE, _ISOLATION_STATE

    # ---- snapshot global state we may touch ----
    _ISOLATION_STATE = {
        "sys_has_torch_npu": "torch_npu" in sys.modules,
        "sys_torch_npu_obj": sys.modules.get("torch_npu"),
        "torch_has_npu_attr": hasattr(torch, "npu"),
        "torch_npu_attr_obj": getattr(torch, "npu", None),
        "pangu_dense_in_sysmodules": "omni.models.pangu.pangu_dense" in sys.modules,
        "pangu_dense_obj": sys.modules.get("omni.models.pangu.pangu_dense"),
    }

    # Snapshot torch.npu.config / allow_internal_format if torch.npu exists
    if _ISOLATION_STATE["torch_has_npu_attr"]:
        npu_obj = torch.npu  # type: ignore[attr-defined]
        _ISOLATION_STATE["npu_has_config"] = hasattr(npu_obj, "config")
        _ISOLATION_STATE["npu_config_obj"] = getattr(npu_obj, "config", None)
        if _ISOLATION_STATE["npu_has_config"]:
            cfg = npu_obj.config  # type: ignore[attr-defined]
            _ISOLATION_STATE["cfg_has_allow_internal_format"] = hasattr(cfg, "allow_internal_format")
            _ISOLATION_STATE["cfg_allow_internal_format_value"] = getattr(cfg, "allow_internal_format", None)
        else:
            _ISOLATION_STATE["cfg_has_allow_internal_format"] = False
            _ISOLATION_STATE["cfg_allow_internal_format_value"] = None
    else:
        _ISOLATION_STATE["npu_has_config"] = False
        _ISOLATION_STATE["npu_config_obj"] = None
        _ISOLATION_STATE["cfg_has_allow_internal_format"] = False
        _ISOLATION_STATE["cfg_allow_internal_format_value"] = None

    # ---- provide minimal stubs only when the module truly does not exist ----
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("torch_npu")
        stub.npu_prefetch = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        sys.modules["torch_npu"] = stub
        _ISOLATION_STATE["created_torch_npu_stub"] = True
    else:
        _ISOLATION_STATE["created_torch_npu_stub"] = False

    # Some projects rely on torch.npu existing at import time.
    # Create only what is missing, and restore it all in teardown_module.
    if not hasattr(torch, "npu"):
        torch.npu = types.SimpleNamespace(  # type: ignore[attr-defined]
            config=types.SimpleNamespace(allow_internal_format=False)
        )
        _ISOLATION_STATE["created_torch_npu_attr"] = True
        _ISOLATION_STATE["created_npu_config"] = True
        _ISOLATION_STATE["created_allow_internal_format"] = True
    else:
        _ISOLATION_STATE["created_torch_npu_attr"] = False
        # ensure config exists
        if not hasattr(torch.npu, "config"):  # type: ignore[attr-defined]
            torch.npu.config = types.SimpleNamespace(allow_internal_format=False)  # type: ignore[attr-defined]
            _ISOLATION_STATE["created_npu_config"] = True
            _ISOLATION_STATE["created_allow_internal_format"] = True
        else:
            _ISOLATION_STATE["created_npu_config"] = False
            if not hasattr(torch.npu.config, "allow_internal_format"):  # type: ignore[attr-defined]
                torch.npu.config.allow_internal_format = False  # type: ignore[attr-defined]
                _ISOLATION_STATE["created_allow_internal_format"] = True
            else:
                _ISOLATION_STATE["created_allow_internal_format"] = False

    # ---- import SUT (or reuse existing) ----
    if _ISOLATION_STATE["pangu_dense_in_sysmodules"]:
        PANGU_DENSE = _ISOLATION_STATE["pangu_dense_obj"]
    else:
        PANGU_DENSE = importlib.import_module("omni.models.pangu.pangu_dense")


def teardown_module():
    global PANGU_DENSE, _ISOLATION_STATE

    # ---- unload SUT if we imported it (avoid leaking a stubbed import to others) ----
    if _ISOLATION_STATE and not _ISOLATION_STATE.get("pangu_dense_in_sysmodules", False):
        with contextlib.suppress(Exception):
            sys.modules.pop("omni.models.pangu.pangu_dense", None)
    PANGU_DENSE = None

    # ---- restore torch_npu in sys.modules ----
    if _ISOLATION_STATE.get("sys_has_torch_npu", False):
        with contextlib.suppress(Exception):
            sys.modules["torch_npu"] = _ISOLATION_STATE.get("sys_torch_npu_obj")
    else:
        with contextlib.suppress(Exception):
            sys.modules.pop("torch_npu", None)

    # ---- restore torch.npu (and any attrs we added) ----
    if _ISOLATION_STATE.get("torch_has_npu_attr", False):
        with contextlib.suppress(Exception):
            torch.npu = _ISOLATION_STATE.get("torch_npu_attr_obj")  # type: ignore[attr-defined]

        # restore / remove config as appropriate
        if not _ISOLATION_STATE.get("npu_has_config", False):
            with contextlib.suppress(Exception):
                if hasattr(torch.npu, "config"):  # type: ignore[attr-defined]
                    delattr(torch.npu, "config")  # type: ignore[attr-defined]
        else:
            with contextlib.suppress(Exception):
                torch.npu.config = _ISOLATION_STATE.get("npu_config_obj")  # type: ignore[attr-defined]

            with contextlib.suppress(Exception):
                cfg = torch.npu.config  # type: ignore[attr-defined]
                if _ISOLATION_STATE.get("cfg_has_allow_internal_format", False):
                    setattr(cfg, "allow_internal_format", _ISOLATION_STATE.get("cfg_allow_internal_format_value"))
                else:
                    if hasattr(cfg, "allow_internal_format"):
                        delattr(cfg, "allow_internal_format")
    else:
        with contextlib.suppress(Exception):
            if hasattr(torch, "npu"):
                delattr(torch, "npu")

    _ISOLATION_STATE = {}


class _DummyPPGroup:
    def __init__(self, is_first_rank: bool, is_last_rank: bool):
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank


class _DummyAttnMeta:
    def __init__(self, cos=None, sin=None, attn_state=None):
        self.cos = cos
        self.sin = sin
        self.attn_state = attn_state


class _DummyRotaryWithGetCosSin:
    def __init__(self, cos, sin):
        self._cos = cos
        self._sin = sin
        self.get_cos_sin = MagicMock(return_value=(cos, sin))


class _DummyLayer(nn.Module):
    """A light-weight stand-in for PANGU_DENSE.PanguEmbeddedDecoderLayer.
    Must match the call signature used by PANGU_DENSE.PanguEmbeddedModel.forward.
    """

    def __init__(self, layer_name: str, rotary_emb):
        super().__init__()
        self.layer_name = layer_name
        self.self_attn = types.SimpleNamespace(rotary_emb=rotary_emb)
        self.calls = []

    def forward(self, positions, hidden_states, residual, cos, sin, next_layer=None):
        self.calls.append(
            {
                "positions": positions,
                "hidden_states": hidden_states,
                "residual": residual,
                "cos": cos,
                "sin": sin,
                "has_next_layer": next_layer is not None,
            }
        )
        # Ensure residual becomes non-None after the first layer, like real decoder does.
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        # Make it deterministic and shape-preserving.
        return hidden_states + 1, residual + 2


class TestModelsPanguDense(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # When running under unittest directly, pytest xunit hooks may not run.
        # Ensure the SUT is imported for this module.
        global PANGU_DENSE
        if PANGU_DENSE is None:
            setup_module()
        self.assertIsNotNone(PANGU_DENSE)

    def tearDown(self):
        super().tearDown()

    # ---------------------------------------------------------------------
    # PANGU_DENSE.PanguEmbeddedForCausalLM.forward: contract forwarding (design intent)
    # ---------------------------------------------------------------------
    def test_pangu_embedded_for_causal_lm_forward_forwards_args_in_model_forward_signature_order(self):
        """Design intent: PANGU_DENSE.PanguEmbeddedForCausalLM.forward should forward args
        in the same order as PANGU_DENSE.PanguEmbeddedModel.forward:
          (input_ids, positions, intermediate_tensors, attn_metadata, inputs_embeds)
        This test will FAIL if the order is swapped (revealing a real bug).
        """
        lm = PANGU_DENSE.PanguEmbeddedForCausalLM.__new__(PANGU_DENSE.PanguEmbeddedForCausalLM)
        # nn.Module init is not required for this test because we only call forward
        # and rely on self.model to be callable.
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
        intermediate_tensors = {"hidden_states": torch.randn(1, 4), "residual": torch.randn(1, 4)}
        attn_metadata = _DummyAttnMeta(cos=torch.randn(1), sin=torch.randn(1))
        inputs_embeds = torch.randn(1, 3, 4)

        lm.model = MagicMock(return_value=torch.randn(1, 4))

        out = lm.forward(
            input_ids=input_ids,
            positions=positions,
            kv_caches=None,
            attn_metadata=attn_metadata,
            selected_indices=None,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        self.assertTrue(torch.is_tensor(out))
        # Critical: check argument order against the model.forward signature.
        lm.model.assert_called_once()
        called_args, called_kwargs = lm.model.call_args

        # 允许 keyword forwarding（推荐）
        if called_kwargs:
            self.assertIs(called_kwargs["intermediate_tensors"], intermediate_tensors)
            self.assertIs(called_kwargs["attn_metadata"], attn_metadata)
            self.assertIs(called_kwargs["inputs_embeds"], inputs_embeds)
        else:
            # 也允许正确的 positional forwarding
            self.assertIs(called_args[2], intermediate_tensors)
            self.assertIs(called_args[3], attn_metadata)
            self.assertIs(called_args[4], inputs_embeds)



    # ---------------------------------------------------------------------
    # PANGU_DENSE.PanguEmbeddedDecoderLayer.forward dispatch
    # ---------------------------------------------------------------------
    def test_pangu_embedded_decoder_layer_forward_dispatches_by_enable_flashcomm_flag(self):
        layer = PANGU_DENSE.PanguEmbeddedDecoderLayer.__new__(PANGU_DENSE.PanguEmbeddedDecoderLayer)
        # Avoid heavy init; we only test dispatch behavior.
        positions = torch.tensor([[0, 1]], dtype=torch.long)
        hidden_states = torch.randn(2, 4)
        residual = None
        cos = torch.randn(1)
        sin = torch.randn(1)

        layer.forward_flashcomm = MagicMock(return_value=(torch.randn(2, 4), torch.randn(2, 4)))
        layer.forward_norm = MagicMock(return_value=(torch.randn(2, 4), torch.randn(2, 4)))

        layer.enable_flashcomm = True
        _ = layer.forward(positions, hidden_states, residual, cos, sin, next_layer=None)
        layer.forward_flashcomm.assert_called_once()
        layer.forward_norm.assert_not_called()

        layer.forward_flashcomm.reset_mock()
        layer.forward_norm.reset_mock()

        layer.enable_flashcomm = False
        _ = layer.forward(positions, hidden_states, residual, cos, sin, next_layer=None)
        layer.forward_norm.assert_called_once()
        layer.forward_flashcomm.assert_not_called()

    # ---------------------------------------------------------------------
    # PANGU_DENSE.PanguEmbeddedAttention.forward rotary call signature branches
    # ---------------------------------------------------------------------
    def test_pangu_embedded_attention_forward_uses_qwenmrotary_3arg_signature_when_rotary_type_matches(self):
        """If rotary_emb type matches (QwenMRotaryEmbedding / MRotaryEmbeddingInterleaved),
        it should be called as rotary(positions, q, k) (no cos/sin).
        """
        attn = PANGU_DENSE.PanguEmbeddedAttention.__new__(PANGU_DENSE.PanguEmbeddedAttention)
        attn.q_size = 2
        attn.kv_size = 1

        # qkv_proj returns tensor that can be split to q,k,v
        qkv = torch.randn(2, attn.q_size + 2 * attn.kv_size)  # (2, 4)
        attn.qkv_proj = MagicMock(return_value=(qkv, None))

        class DummyQwenMRotary:
            def __init__(self):
                self.calls = []

            def __call__(self, positions, q, k):
                self.calls.append((positions, q, k))
                return q, k

        dummy_rotary = DummyQwenMRotary()
        # Patch module-level class reference so `type(self.rotary_emb) in [...]` matches.
        with patch.object(PANGU_DENSE, "QwenMRotaryEmbedding", DummyQwenMRotary), patch.object(
            PANGU_DENSE, "MRotaryEmbeddingInterleaved", type("DummyMRI", (), {})
        ):
            attn.rotary_emb = dummy_rotary
            attn.attn = MagicMock(return_value=torch.randn(2, 4))
            attn.o_proj = MagicMock(return_value=(torch.randn(2, 4), None))

            positions = torch.tensor([[0, 1]], dtype=torch.long)
            hidden_states = torch.randn(2, 8)
            out = attn.forward(positions, hidden_states, cos=None, sin=None, x_transform="AG", reduce_type="RS", next_layer=None)

            self.assertTrue(torch.is_tensor(out))
            self.assertEqual(len(dummy_rotary.calls), 1)
            # Ensure 3-arg signature (positions, q, k) is used
            self.assertEqual(len(dummy_rotary.calls[0]), 3)

    def test_pangu_embedded_attention_forward_uses_default_5arg_rotary_signature_with_cos_sin(self):
        """Otherwise rotary_emb should be called as rotary(positions, q, k, cos, sin)."""
        attn = PANGU_DENSE.PanguEmbeddedAttention.__new__(PANGU_DENSE.PanguEmbeddedAttention)
        attn.q_size = 2
        attn.kv_size = 1

        qkv = torch.randn(2, attn.q_size + 2 * attn.kv_size)  # (2, 4)
        attn.qkv_proj = MagicMock(return_value=(qkv, None))

        class DummyOtherRotary:
            def __init__(self):
                self.calls = []

            def __call__(self, positions, q, k, cos, sin):
                self.calls.append((positions, q, k, cos, sin))
                return q, k

        dummy_rotary = DummyOtherRotary()
        attn.rotary_emb = dummy_rotary
        attn.attn = MagicMock(return_value=torch.randn(2, 4))
        attn.o_proj = MagicMock(return_value=(torch.randn(2, 4), None))

        positions = torch.tensor([[0, 1]], dtype=torch.long)
        hidden_states = torch.randn(2, 8)
        cos = torch.randn(1)
        sin = torch.randn(1)
        out = attn.forward(positions, hidden_states, cos=cos, sin=sin, x_transform=None, reduce_type="AR", next_layer=[])

        self.assertTrue(torch.is_tensor(out))
        self.assertEqual(len(dummy_rotary.calls), 1)
        self.assertEqual(len(dummy_rotary.calls[0]), 5)

    # ---------------------------------------------------------------------
    # PANGU_DENSE.PanguEmbeddedModel.forward: PP behavior + attn_metadata handling
    # ---------------------------------------------------------------------
    def test_pangu_embedded_model_forward_first_rank_prefers_inputs_embeds_over_embed_tokens(self):
        """First PP rank: if inputs_embeds is provided, embed_tokens should not be used."""
        model = PANGU_DENSE.PanguEmbeddedModel.__new__(PANGU_DENSE.PanguEmbeddedModel)
        nn.Module.__init__(model)

        hidden_size = 4
        positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
        inputs_embeds = torch.randn(1, 3, hidden_size)

        model.embed_tokens = MagicMock(return_value=torch.randn(1, 3, hidden_size))
        model.start_layer = 0
        model.end_layer = 1

        dummy_rotary = _DummyRotaryWithGetCosSin(cos=torch.randn(1), sin=torch.randn(1))
        layer0 = _DummyLayer(layer_name="model.layers.0.self_attn.attn", rotary_emb=dummy_rotary)
        model.layers = nn.ModuleList([layer0])

        model.aux_hidden_state_layers = tuple()
        model.norm = MagicMock(return_value=(torch.randn(1, 3, hidden_size), None))

        with patch.object(PANGU_DENSE, "get_pp_group", return_value=_DummyPPGroup(True, True)):
            out = model.forward(
                input_ids=None,
                positions=positions,
                intermediate_tensors=None,
                attn_metadata=None,
                inputs_embeds=inputs_embeds,
            )

        # embed_tokens should be bypassed
        model.embed_tokens.assert_not_called()
        self.assertTrue(torch.is_tensor(out))
        self.assertEqual(len(layer0.calls), 1)
        self.assertTrue(torch.allclose(layer0.calls[0]["hidden_states"], inputs_embeds))

    def test_pangu_embedded_model_forward_not_first_rank_consumes_intermediate_tensors_and_returns_intermediate_when_not_last_rank(self):
        """Non-first PP rank must use intermediate_tensors, and non-last rank returns IntermediateTensors-like object."""
        model = PANGU_DENSE.PanguEmbeddedModel.__new__(PANGU_DENSE.PanguEmbeddedModel)
        nn.Module.__init__(model)

        hidden_size = 4
        positions = torch.tensor([[0, 1, 2]], dtype=torch.long)

        hs_in = torch.randn(1, 3, hidden_size)
        res_in = torch.randn(1, 3, hidden_size)
        intermediate_tensors = {"hidden_states": hs_in, "residual": res_in}

        model.embed_tokens = MagicMock()  # must not be called
        model.start_layer = 0
        model.end_layer = 1

        dummy_rotary = _DummyRotaryWithGetCosSin(cos=torch.randn(1), sin=torch.randn(1))
        layer0 = _DummyLayer(layer_name="model.layers.0.self_attn.attn", rotary_emb=dummy_rotary)
        model.layers = nn.ModuleList([layer0])
        model.aux_hidden_state_layers = tuple()

        # Make IntermediateTensors construction lightweight for unit tests.
        with patch.object(PANGU_DENSE, "get_pp_group", return_value=_DummyPPGroup(False, False)), patch.object(
            PANGU_DENSE, "IntermediateTensors", side_effect=lambda d: d
        ):
            out = model.forward(
                input_ids=torch.tensor([[1, 2, 3]]),
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                attn_metadata=None,
                inputs_embeds=None,
            )

        model.embed_tokens.assert_not_called()
        self.assertIsInstance(out, dict)
        self.assertIn("hidden_states", out)
        self.assertIn("residual", out)
        self.assertEqual(out["hidden_states"].shape, hs_in.shape)
        self.assertEqual(out["residual"].shape, res_in.shape)

    def test_pangu_embedded_model_forward_attn_metadata_dict_uses_cos_sin_from_metadata_and_skips_get_cos_sin(self):
        """When attn_metadata is dict, it should pick first entry and use its cos/sin."""
        model = PANGU_DENSE.PanguEmbeddedModel.__new__(PANGU_DENSE.PanguEmbeddedModel)
        nn.Module.__init__(model)

        hidden_size = 4
        positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        model.embed_tokens = MagicMock(return_value=torch.randn(1, 3, hidden_size))
        model.start_layer = 0
        model.end_layer = 1

        dummy_rotary = _DummyRotaryWithGetCosSin(cos=torch.randn(1), sin=torch.randn(1))
        layer0 = _DummyLayer(layer_name="model.layers.0.self_attn.attn", rotary_emb=dummy_rotary)
        model.layers = nn.ModuleList([layer0])
        model.aux_hidden_state_layers = tuple()
        model.norm = MagicMock(return_value=(torch.randn(1, 3, hidden_size), None))

        meta_cos = torch.randn(1)
        meta_sin = torch.randn(1)
        attn_metadata = {"any_key": _DummyAttnMeta(cos=meta_cos, sin=meta_sin)}

        with patch.object(PANGU_DENSE, "get_pp_group", return_value=_DummyPPGroup(True, True)):
            _ = model.forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                attn_metadata=attn_metadata,
                inputs_embeds=None,
            )

        # Should not compute cos/sin from rotary when metadata provides them.
        dummy_rotary.get_cos_sin.assert_not_called()
        self.assertEqual(len(layer0.calls), 1)
        self.assertTrue(torch.equal(layer0.calls[0]["cos"], meta_cos))
        self.assertTrue(torch.equal(layer0.calls[0]["sin"], meta_sin))

    def test_pangu_embedded_model_forward_returns_aux_hidden_states_when_configured(self):
        """If aux_hidden_state_layers is set and last rank, forward should return (hidden_states, aux_hidden_states)."""
        model = PANGU_DENSE.PanguEmbeddedModel.__new__(PANGU_DENSE.PanguEmbeddedModel)
        nn.Module.__init__(model)

        hidden_size = 4
        positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        model.embed_tokens = MagicMock(return_value=torch.randn(1, 3, hidden_size))
        model.start_layer = 0
        model.end_layer = 3  # three layers

        dummy_rotary = _DummyRotaryWithGetCosSin(cos=torch.randn(1), sin=torch.randn(1))
        layer0 = _DummyLayer(layer_name="model.layers.0.self_attn.attn", rotary_emb=dummy_rotary)
        layer1 = _DummyLayer(layer_name="model.layers.1.self_attn.attn", rotary_emb=dummy_rotary)
        layer2 = _DummyLayer(layer_name="model.layers.2.self_attn.attn", rotary_emb=dummy_rotary)
        model.layers = nn.ModuleList([layer0, layer1, layer2])

        # Ensure idx=2 is safe: residual is already tensor by then.
        model.aux_hidden_state_layers = (2,)
        model.norm = MagicMock(side_effect=lambda hs, res, y_transform=None: (hs, None))

        with patch.object(PANGU_DENSE, "get_pp_group", return_value=_DummyPPGroup(True, True)):
            out = model.forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=None,
                attn_metadata=None,
                inputs_embeds=None,
            )

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        hidden_states, aux_states = out
        self.assertTrue(torch.is_tensor(hidden_states))
        self.assertIsInstance(aux_states, list)
        self.assertEqual(len(aux_states), 1)
        self.assertEqual(aux_states[0].shape, hidden_states.shape)

    # ---------------------------------------------------------------------
    # PANGU_DENSE.PanguEmbeddedModel.load_weights: stacked mapping + skip rules + cache scale
    # ---------------------------------------------------------------------
    def _make_minimal_model_for_load_weights(self):
        m = PANGU_DENSE.PanguEmbeddedModel.__new__(PANGU_DENSE.PanguEmbeddedModel)
        nn.Module.__init__(m)

        # Minimal attributes used by load_weights
        m.quant_config = None

        # Create minimal parameter tree that produces names matching mapping replacements.
        m.layers = nn.ModuleList([nn.Module()])
        m.layers[0].self_attn = nn.Module()
        m.layers[0].self_attn.qkv_proj = nn.Module()
        m.layers[0].self_attn.qkv_proj.weight = Parameter(torch.empty(1))
        m.layers[0].self_attn.qkv_proj.weight.weight_loader = MagicMock()

        m.layers[0].mlp = nn.Module()
        m.layers[0].mlp.gate_up_proj = nn.Module()
        m.layers[0].mlp.gate_up_proj.weight = Parameter(torch.empty(1))
        m.layers[0].mlp.gate_up_proj.weight.weight_loader = MagicMock()

        # For quant cache scale path
        m.layers[0].self_attn.kv_scale = Parameter(torch.empty(1))
        m.layers[0].self_attn.kv_scale.weight_loader = MagicMock()

        return m

    def test_pangu_embedded_model_load_weights_stacked_mapping_and_skip_and_cache_scale(self):
        m = self._make_minimal_model_for_load_weights()

        # quant_config.get_cache_scale(name) -> scale_name in params_dict
        class DummyQuantCfg:
            def get_cache_scale(self, name):
                if "kv_cache_scale" in name:
                    return "layers.0.self_attn.kv_scale"
                return None

        m.quant_config = DummyQuantCfg()

        weights = [
            # stacked mapping: q/k/v -> qkv_proj with shard_id "q"/"k"/"v"
            ("language_model.layers.0.self_attn.q_proj.weight", torch.randn(1)),
            ("layers.0.self_attn.k_proj.weight", torch.randn(1)),
            ("layers.0.self_attn.v_proj.weight", torch.randn(1)),
            # stacked mapping: gate/up -> gate_up_proj shard_id 0/1
            ("layers.0.mlp.gate_proj.weight", torch.randn(1)),
            ("layers.0.mlp.up_proj.weight", torch.randn(1)),
            # quant cache scale path
            ("layers.0.self_attn.kv_cache_scale", torch.randn(1)),
            # skip rules
            ("visual.encoder.weight", torch.randn(1)),
            ("layers.0.self_attn.rotary_emb.inv_freq", torch.randn(1)),
            ("layers.0.self_attn.rotary_emb.cos_cached", torch.randn(1)),
            ("layers.0.self_attn.rotary_emb.sin_cached", torch.randn(1)),
        ]

        with patch.object(PANGU_DENSE, "is_pp_missing_parameter", return_value=False), patch.object(
            PANGU_DENSE, "maybe_remap_kv_scale_name", side_effect=lambda name, params_dict: name
        ), patch.object(PANGU_DENSE, "default_weight_loader", side_effect=lambda p, w: None):
            loaded = m.load_weights(weights)

        # qkv_proj weight_loader should have been called 3 times with shard ids q/k/v
        qkv_loader = m.layers[0].self_attn.qkv_proj.weight.weight_loader
        self.assertEqual(qkv_loader.call_count, 3)
        shard_ids = [call.args[2] for call in qkv_loader.call_args_list]  # (param, loaded_weight, shard_id)
        self.assertCountEqual(shard_ids, ["q", "k", "v"])

        # gate_up_proj should have shard ids 0 and 1
        gate_loader = m.layers[0].mlp.gate_up_proj.weight.weight_loader
        self.assertEqual(gate_loader.call_count, 2)
        gate_shards = [call.args[2] for call in gate_loader.call_args_list]
        self.assertCountEqual(gate_shards, [0, 1])

        # kv_scale should be loaded via quant cache scale path (no shard_id)
        kv_scale_loader = m.layers[0].self_attn.kv_scale.weight_loader
        kv_scale_loader.assert_called_once()
        self.assertIn("layers.0.self_attn.kv_scale", loaded)

        # Skip rules: should not appear in loaded set
        self.assertNotIn("visual.encoder.weight", loaded)
        self.assertNotIn("layers.0.self_attn.rotary_emb.inv_freq", loaded)
        self.assertNotIn("layers.0.self_attn.rotary_emb.cos_cached", loaded)
        self.assertNotIn("layers.0.self_attn.rotary_emb.sin_cached", loaded)

    # ---------------------------------------------------------------------
    # should_use_eager_mode: dict unwrap + DecodeOnly handling
    # ---------------------------------------------------------------------
    def test_pangu_embedded_for_causal_lm_should_use_eager_mode_handles_none_and_dict(self):
        lm = PANGU_DENSE.PanguEmbeddedForCausalLM.__new__(PANGU_DENSE.PanguEmbeddedForCausalLM)

        # Minimal model stub for layer_name lookup.
        layer_stub = types.SimpleNamespace(layer_name="model.layers.0.self_attn.attn")
        lm.model = types.SimpleNamespace(layers=[layer_stub], start_layer=0)

        # No metadata => eager
        self.assertTrue(lm.should_use_eager_mode(attn_metadata=None))

        # Dict metadata: should pick entry by layer_name
        class DummyAscendState:
            DecodeOnly = object()

        with patch.object(PANGU_DENSE, "AscendAttentionState", DummyAscendState):
            md_decode = _DummyAttnMeta(attn_state=DummyAscendState.DecodeOnly)
            md_prefill = _DummyAttnMeta(attn_state=object())
            self.assertFalse(
                lm.should_use_eager_mode(attn_metadata={layer_stub.layer_name: md_decode})
            )
            self.assertTrue(
                lm.should_use_eager_mode(attn_metadata={layer_stub.layer_name: md_prefill})
            )


if __name__ == "__main__":
    unittest.main()
