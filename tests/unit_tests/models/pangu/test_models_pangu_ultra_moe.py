import os
import sys
import types
import contextlib
import unittest
from unittest.mock import Mock, patch, MagicMock
import importlib
import inspect

import torch
from torch import nn
from torch.nn import Parameter

# -------------------------------------------------------------
# Global isolation helpers (install stubs only within this module)
# -------------------------------------------------------------
_MISSING = object()
_ISO_STATE = {
    "torch_npu_orig": _MISSING,
    "torch_npu_stub": None,
    "torch_npu_had": False,
    "torch_npu_added": False,
    "torch_npu_attr_had": False,
    "torch_npu_attr_orig": _MISSING,
    "torch_npu_attr_stub": None,
    "torch_npu_attr_added": False,
}


def _install_npu_stubs_if_needed():
    """Install minimal torch_npu / torch.npu stubs ONLY when missing, and record state for restore.

    Critical isolation rule:
    - Never mutate an existing real torch.npu (e.g., do NOT touch allow_internal_format).
    - Only add stubs when absent.
    """
    # ---- sys.modules["torch_npu"] ----
    _ISO_STATE["torch_npu_had"] = "torch_npu" in sys.modules
    _ISO_STATE["torch_npu_orig"] = sys.modules.get("torch_npu", _MISSING)

    if not _ISO_STATE["torch_npu_had"]:
        try:
            import torch_npu  # noqa: F401
        except Exception:
            stub = types.ModuleType("torch_npu")
            stub.npu_prefetch = lambda *args, **kwargs: None  # type: ignore[attr-defined]
            sys.modules["torch_npu"] = stub
            _ISO_STATE["torch_npu_stub"] = stub
            _ISO_STATE["torch_npu_added"] = True

    # ---- torch.npu attribute ----
    _ISO_STATE["torch_npu_attr_had"] = hasattr(torch, "npu")
    _ISO_STATE["torch_npu_attr_orig"] = getattr(torch, "npu", _MISSING)

    if not _ISO_STATE["torch_npu_attr_had"]:
        npu_stub = types.SimpleNamespace(
            config=types.SimpleNamespace(allow_internal_format=False),
            Stream=lambda *args, **kwargs: None,
            current_stream=lambda: None,
            stream=lambda s: contextlib.nullcontext(),
        )
        torch.npu = npu_stub  # type: ignore[attr-defined]
        _ISO_STATE["torch_npu_attr_stub"] = npu_stub
        _ISO_STATE["torch_npu_attr_added"] = True


def _restore_npu_stubs():
    """Restore sys.modules / torch attributes to their original state (best-effort, identity-safe)."""
    # Restore torch.npu
    if _ISO_STATE.get("torch_npu_attr_added"):
        # Only delete if it's still our stub
        cur = getattr(torch, "npu", _MISSING)
        if cur is _ISO_STATE.get("torch_npu_attr_stub"):
            try:
                delattr(torch, "npu")
            except Exception:
                pass
    else:
        # If we never added, do not touch existing real torch.npu
        pass

    # Restore sys.modules["torch_npu"]
    if _ISO_STATE.get("torch_npu_added"):
        # Only remove if still our stub
        cur = sys.modules.get("torch_npu", None)
        if cur is _ISO_STATE.get("torch_npu_stub"):
            sys.modules.pop("torch_npu", None)
    else:
        # If existed before, never touch
        pass

    # reset runtime markers (avoid cascading issues if re-imported)
    _ISO_STATE["torch_npu_stub"] = None
    _ISO_STATE["torch_npu_added"] = False
    _ISO_STATE["torch_npu_attr_stub"] = None
    _ISO_STATE["torch_npu_attr_added"] = False


# Delay business-module import until setUpModule to avoid collection-time side effects
PANGU_ULTRA_MOE = None


def setUpModule():  # noqa: N802
    global PANGU_ULTRA_MOE
    _install_npu_stubs_if_needed()
    PANGU_ULTRA_MOE = importlib.import_module("omni.models.pangu.pangu_ultra_moe")


def tearDownModule():  # noqa: N802
    global PANGU_ULTRA_MOE
    # Best-effort: drop our reference (avoid leaking stub-captured globals)
    PANGU_ULTRA_MOE = None
    _restore_npu_stubs()


# ---------------------------------------------------------------------
# Light-weight dummies for patching / stubbing
# ---------------------------------------------------------------------
class _DummyTPGroup:
    world_size = 1
    rank_in_group = 0

    def all_gather(self, x, dim=0):
        return x

    def reduce_scatter(self, x):
        return x


class _DummyOptCfg:
    def __init__(self, enable_prefill_micro_batch=False, merge_qkv=False, use_super_kernel=False):
        self.enable_prefill_micro_batch = enable_prefill_micro_batch
        self.merge_qkv = merge_qkv
        self.use_super_kernel = use_super_kernel


class _DummyExtraCfg:
    def __init__(self, opt_cfg=None):
        self.operator_opt_config = opt_cfg or _DummyOptCfg()
        self.parall_config = types.SimpleNamespace(redundancy_shared_expert_num=0)


class _DummyPrefill:
    def __init__(self, seq_lens=None, query_lens=None, block_table=None):
        self.seq_lens = seq_lens or []
        self.query_lens = query_lens or []
        self.block_table = block_table
        self.seq_qlen_group = None
        self.seq_kvlen_group = None


class _DummyAttnMeta:
    def __init__(self, prefill=None):
        self.prefill = prefill


class _DummyMeta:
    """AttentionMetadata-like minimal object used by split_attn_metadata_index/refresh_metadata."""

    def __init__(self, slot_mapping: torch.Tensor, prefill: _DummyPrefill):
        self.slot_mapping = slot_mapping
        self.prefill = prefill


class TestModelsPanguUltraMOE(unittest.TestCase):
    def setUp(self):
        super().setUp()
        if PANGU_ULTRA_MOE is None:
            raise RuntimeError("PANGU_ULTRA_MOE not initialized; setUpModule() didn't run?")

    def tearDown(self):
        super().tearDown()

    # ---------------------------------------------------------------
    # 1) ParallelPanguUltraMoEMLP: activation gate (design intent)
    # ---------------------------------------------------------------
    def test_parallel_pangu_ultra_moe_mlp_init_requires_silu_activation(self):
        """Only 'silu' supported; others should fail-fast."""
        DummyLinear = type(
            "DummyLinear",
            (),
            {
                "__init__": lambda self, *a, **k: None,
                "forward": lambda self, x: (x, None),
                "weight_scale": None,
            },
        )

        with patch.object(PANGU_ULTRA_MOE, "AscendMergedColumnParallelLinear", DummyLinear), \
             patch.object(PANGU_ULTRA_MOE, "AscendRowParallelLinear", DummyLinear), \
             patch.object(PANGU_ULTRA_MOE, "get_mlp_tp_group", return_value=_DummyTPGroup()), \
             patch.object(PANGU_ULTRA_MOE, "SiluAndMul", lambda: (lambda x, quant_symbol=None: x)):
            with self.assertRaises(ValueError):
                PANGU_ULTRA_MOE.ParallelPanguUltraMoEMLP(
                    hidden_size=16,
                    intermediate_size=32,
                    hidden_act="gelu",  # not supported
                    quant_config=None,
                    prefix="x",
                )

    # ---------------------------------------------------------------
    # 2) DecoderLayer: MoE vs Dense selection logic
    # ---------------------------------------------------------------
    def test_decoder_layer_selects_moe_or_dense_mlp_by_config_and_layer_idx(self):
        """Structure gate: select MoE or Dense depending on config + layer_idx."""
        class DummyAttn:
            def __init__(self, *a, **k):
                pass

        class DummyMoE:
            def __init__(self, *a, **k):
                pass

        class DummyDense:
            def __init__(self, *a, **k):
                pass

        class DummyNorm:
            def __init__(self, *a, **k):
                pass

        cfg = types.SimpleNamespace(
            hidden_size=16,
            intermediate_size=32,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            num_attention_heads=2,
            attention_qk_dim=8,
            attention_qk_rope_dim=8,
            attention_v_dim=8,
            attention_kv_lora_dim=4,
            num_hidden_layers=8,
            num_routed_experts=8,
            num_dense_layers=3,
            moe_layer_freq=2,
        )

        with patch.object(PANGU_ULTRA_MOE, "DeepseekMLA", DummyAttn), \
             patch.object(PANGU_ULTRA_MOE, "DeepseekMoE", DummyMoE), \
             patch.object(PANGU_ULTRA_MOE, "ParallelPanguUltraMoEMLP", DummyDense), \
             patch.object(PANGU_ULTRA_MOE, "RMSNorm", DummyNorm):
            # layer_idx=4 >= 3 and 4%2==0 => MoE
            layer_moe = PANGU_ULTRA_MOE.PanguUltraMoEDecoderLayer(
                cfg, prefix="model.layers.4", cache_config=None, quant_config=None
            )
            self.assertTrue(layer_moe.is_moe)
            self.assertIsInstance(layer_moe.mlp, DummyMoE)

            # layer_idx=5 => Dense
            layer_dense = PANGU_ULTRA_MOE.PanguUltraMoEDecoderLayer(
                cfg, prefix="model.layers.5", cache_config=None, quant_config=None
            )
            self.assertFalse(layer_dense.is_moe)
            self.assertIsInstance(layer_dense.mlp, DummyDense)

    # ---------------------------------------------------------------
    # 3) DecoderLayer.forward_mlp: dict attn_metadata unwrap by layer_name
    # ---------------------------------------------------------------
    def test_decoder_layer_forward_mlp_uses_metadata_from_dict_by_layer_name(self):
        layer = PANGU_ULTRA_MOE.PanguUltraMoEDecoderLayer.__new__(PANGU_ULTRA_MOE.PanguUltraMoEDecoderLayer)
        layer.layer_name = "model.layers.0.self_attn.attn"
        layer.is_moe = False

        layer.post_attention_layernorm = MagicMock(side_effect=lambda hs, res: (hs, res))
        layer.post_mlp_layernorm = MagicMock(side_effect=lambda hs: hs)

        captured = {}

        def _mlp(hs, res, attn_metadata, *a, **k):
            captured["attn_metadata"] = attn_metadata
            return hs, res

        layer.mlp = MagicMock(side_effect=_mlp)

        md_target = _DummyAttnMeta(prefill=None)
        md_other = _DummyAttnMeta(prefill=None)
        attn_metadata = {layer.layer_name: md_target, "other": md_other}

        hs = torch.randn(3, 16)
        res = torch.randn(3, 16)
        out_hs, out_res = layer.forward_mlp(hs, attn_metadata, res)

        self.assertTrue(torch.is_tensor(out_hs))
        self.assertTrue(torch.is_tensor(out_res))
        self.assertIs(captured["attn_metadata"], md_target)

    # ---------------------------------------------------------------
    # 4) DecoderLayer.forward_mlp: MoE tuple output merge (shared+routed)
    # ---------------------------------------------------------------
    def test_decoder_layer_forward_mlp_moe_tuple_output_is_merged(self):
        dummy_extra = _DummyExtraCfg(_DummyOptCfg(use_super_kernel=False))
        with patch.object(PANGU_ULTRA_MOE, "model_extra_config", dummy_extra):
            layer = PANGU_ULTRA_MOE.PanguUltraMoEDecoderLayer.__new__(PANGU_ULTRA_MOE.PanguUltraMoEDecoderLayer)
            layer.layer_name = "model.layers.0.self_attn.attn"
            layer.is_moe = True
            layer.post_attention_layernorm = MagicMock(side_effect=lambda hs, res: (hs, res))

            # MoE returns (shared, routed)
            def _moe(hs, res, attn_metadata, layer_id=None, next_attention_weights=None, comm_group=None):
                return (hs, hs), res

            layer.mlp = MagicMock(side_effect=_moe)

            hs = torch.randn(2, 16)
            res = torch.randn(2, 16)
            md = _DummyAttnMeta(prefill=None)  # decode

            out_hs, out_res = layer.forward_mlp(hs, md, res, layer_id=0)

            self.assertTrue(torch.is_tensor(out_hs))
            self.assertTrue(torch.is_tensor(out_res))
            self.assertEqual(out_hs.shape, hs.shape)

    # ---------------------------------------------------------------
    # 5) PanguUltraMoEModel.forward: route to micro-batch vs normal
    # ---------------------------------------------------------------
    def test_model_forward_routes_to_micro_batch_only_when_enabled_and_multi_seq_prefill(self):
        model = PANGU_ULTRA_MOE.PanguUltraMoEModel.__new__(PANGU_ULTRA_MOE.PanguUltraMoEModel)

        # bind method & provide attrs used for key resolution
        model.prefix = "model.layers"
        model.postfix = ".self_attn.attn"
        model.get_layer_attn_metadata = PANGU_ULTRA_MOE.PanguUltraMoEModel.get_layer_attn_metadata.__get__(model)

        model.forward_micro_batch = MagicMock(return_value="MICRO")
        model.forward_normal = MagicMock(return_value="NORMAL")

        md_key = "model.layers.0.self_attn.attn"
        md = _DummyAttnMeta(prefill=_DummyPrefill(seq_lens=[2, 2], query_lens=[2, 2], block_table=None))
        attn_metadata = {md_key: md}

        # Enabled => MICRO
        with patch.object(PANGU_ULTRA_MOE, "model_extra_config", _DummyExtraCfg(_DummyOptCfg(enable_prefill_micro_batch=True))):
            out = model.forward(
                input_ids=torch.tensor([1, 2, 3]),
                positions=torch.tensor([0, 1, 2]),
                kv_caches=None,
                attn_metadata=attn_metadata,
                intermediate_tensors=None,
                max_num_tokens=None,
                lm_head=None,
            )
        self.assertEqual(out, "MICRO")
        model.forward_micro_batch.assert_called_once()
        model.forward_normal.assert_not_called()

        # Disabled => NORMAL
        model.forward_micro_batch.reset_mock()
        model.forward_normal.reset_mock()
        with patch.object(PANGU_ULTRA_MOE, "model_extra_config", _DummyExtraCfg(_DummyOptCfg(enable_prefill_micro_batch=False))):
            out2 = model.forward(
                input_ids=torch.tensor([1, 2, 3]),
                positions=torch.tensor([0, 1, 2]),
                kv_caches=None,
                attn_metadata=attn_metadata,
                intermediate_tensors=None,
                max_num_tokens=None,
                lm_head=None,
            )
        self.assertEqual(out2, "NORMAL")
        model.forward_normal.assert_called_once()
        model.forward_micro_batch.assert_not_called()

    # ---------------------------------------------------------------
    # 6) partition_list: non-empty split + sane split_index
    # ---------------------------------------------------------------
    def test_partition_list_returns_two_non_empty_parts_and_valid_split_index(self):
        model = PANGU_ULTRA_MOE.PanguUltraMoEModel.__new__(PANGU_ULTRA_MOE.PanguUltraMoEModel)
        model.partition_list = PANGU_ULTRA_MOE.PanguUltraMoEModel.partition_list.__get__(model)

        left, right, split_idx = model.partition_list([3, 1, 2], pos=6)
        self.assertTrue(len(left) > 0)
        self.assertTrue(len(right) > 0)
        self.assertEqual(sorted(left + right), sorted([3, 1, 2]))
        self.assertTrue(1 <= split_idx <= 2)

    # ---------------------------------------------------------------
    # 7) split_attn_metadata_index + refresh_metadata: deepcopy + padding + field updates
    # ---------------------------------------------------------------
    def test_split_attn_metadata_index_pads_slot_mapping_and_updates_prefill_fields_without_mutating_original(self):
        model = PANGU_ULTRA_MOE.PanguUltraMoEModel.__new__(PANGU_ULTRA_MOE.PanguUltraMoEModel)
        model.index_batch = PANGU_ULTRA_MOE.PanguUltraMoEModel.index_batch.__get__(model)
        model.pad_tensor = PANGU_ULTRA_MOE.PanguUltraMoEModel.pad_tensor.__get__(model)
        model.refresh_metadata = PANGU_ULTRA_MOE.PanguUltraMoEModel.refresh_metadata.__get__(model)
        model.split_attn_metadata_index = PANGU_ULTRA_MOE.PanguUltraMoEModel.split_attn_metadata_index.__get__(model)

        prefill = _DummyPrefill(seq_lens=[2, 1], query_lens=[2, 1], block_table=torch.tensor([0]))
        metadata = _DummyMeta(slot_mapping=torch.tensor([10, 11, 12], dtype=torch.int64), prefill=prefill)

        def _fake_group_request_list(seq_lens, query_lens, block_table, max_num_tokens):
            # return (seq_kvlen_group, seq_qlen_group, _)
            return [seq_lens], [query_lens], None

        with patch.object(PANGU_ULTRA_MOE, "group_request_list", side_effect=_fake_group_request_list):
            out = model.split_attn_metadata_index(
                metadata=metadata,
                is_local_stream=True,
                split_idx=2,      # slot_mapping => [10,11]
                pad_size=1,       # pad one element with -1
                max_num_tokens=128,
            )

        # original unchanged
        self.assertTrue(torch.equal(metadata.slot_mapping, torch.tensor([10, 11, 12])))

        # output padded & updated
        self.assertEqual(out.slot_mapping.shape[0], 3)
        self.assertEqual(int(out.slot_mapping[-1].item()), -1)
        self.assertEqual(out.prefill.seq_lens, [2])
        self.assertEqual(out.prefill.query_lens, [2])
        self.assertEqual(out.prefill.seq_qlen_group, [[2]])
        self.assertEqual(out.prefill.seq_kvlen_group, [[2]])

    # ---------------------------------------------------------------
    # 8) compute_lmhead: selected_indices branch (shape guarding)
    # ---------------------------------------------------------------
    def test_compute_lmhead_flattens_and_index_selects_when_selected_indices_provided(self):
        lm = PANGU_ULTRA_MOE.PanguUltraMoEForCausalLM.__new__(PANGU_ULTRA_MOE.PanguUltraMoEForCausalLM)

        captured = {}

        def _lm_head(x, *args, **kwargs):
            captured["shape"] = tuple(x.shape)
            return torch.randn(x.shape[0], 10), None

        lm.lm_head = MagicMock(side_effect=_lm_head)

        hidden_states = torch.randn(1, 3, 16)  # flatten => (3,16)
        selected_indices = torch.tensor([0, 2], dtype=torch.int64)  # => (2,16)

        sig = inspect.signature(lm.compute_lmhead)
        kwargs = dict(hidden_states=hidden_states, selected_indices=selected_indices)

        # 兼容不同版本签名：只在存在时才传
        if "reduce_type" in sig.parameters:
            kwargs["reduce_type"] = "AR"
        if "x_transform" in sig.parameters:
            kwargs["x_transform"] = None

        logits, _ = lm.compute_lmhead(**kwargs)

        self.assertTrue(torch.is_tensor(logits))
        self.assertEqual(captured["shape"], (2, 16))

    # ---------------------------------------------------------------
    # 9) should_use_eager_mode: None / dict unwrap / prefill
    # ---------------------------------------------------------------
    def test_should_use_eager_mode_true_for_missing_metadata_or_prefill_false_for_decode(self):
        lm = PANGU_ULTRA_MOE.PanguUltraMoEForCausalLM.__new__(PANGU_ULTRA_MOE.PanguUltraMoEForCausalLM)

        layer_stub = types.SimpleNamespace(layer_name="model.layers.0.self_attn.attn")
        lm.model = types.SimpleNamespace(layers=[layer_stub], start_layer=0)

        self.assertTrue(lm.should_use_eager_mode(attn_metadata=None))

        md_decode = _DummyAttnMeta(prefill=None)
        self.assertFalse(lm.should_use_eager_mode(attn_metadata={layer_stub.layer_name: md_decode}))

        md_prefill = _DummyAttnMeta(prefill=_DummyPrefill(seq_lens=[1], query_lens=[1], block_table=None))
        self.assertTrue(lm.should_use_eager_mode(attn_metadata={layer_stub.layer_name: md_prefill}))

    # ---------------------------------------------------------------
    # 10) load_weights: skip rotary + stacked mapping (gate/up -> gate_up)
    # ---------------------------------------------------------------
    def test_load_weights_skips_rotary_inv_freq_and_applies_stacked_param_mapping(self):
        dummy_extra = _DummyExtraCfg(_DummyOptCfg(merge_qkv=False))

        with patch.object(PANGU_ULTRA_MOE, "model_extra_config", dummy_extra), \
             patch.object(PANGU_ULTRA_MOE, "is_pp_missing_parameter", return_value=False), \
             patch.object(PANGU_ULTRA_MOE, "default_weight_loader", side_effect=lambda p, w: None), \
             patch.object(PANGU_ULTRA_MOE, "FusedMoE") as _fake_fused_moe:

            _fake_fused_moe.make_expert_params_mapping.return_value = []

            lm = PANGU_ULTRA_MOE.PanguUltraMoEForCausalLM.__new__(PANGU_ULTRA_MOE.PanguUltraMoEForCausalLM)
            lm.config = types.SimpleNamespace(
                architectures=["PanguUltraMoEForCausalLM"],
                num_mtp_layers=0,
                num_hidden_layers=1,
                num_routed_experts=0,
            )

            gate_param = Parameter(torch.empty(1))
            gate_param.weight_loader = MagicMock()
            lm.named_parameters = lambda: [("model.layers.0.mlp.gate_up_proj.weight", gate_param)]

            weights = [
                ("model.layers.0.mlp.gate_proj.weight", torch.randn(1)),
                ("model.layers.0.mlp.up_proj.weight", torch.randn(1)),
                ("model.layers.0.self_attn.rotary_emb.inv_freq", torch.randn(1)),  # must skip
            ]

            loaded = lm.load_weights(weights)

        self.assertEqual(gate_param.weight_loader.call_count, 2)
        shard_ids = [c.args[2] for c in gate_param.weight_loader.call_args_list]
        self.assertCountEqual(shard_ids, [0, 1])

        self.assertNotIn("model.layers.0.self_attn.rotary_emb.inv_freq", loaded)
        self.assertIn("model.layers.0.mlp.gate_up_proj.weight", loaded)


if __name__ == "__main__":
    unittest.main()
