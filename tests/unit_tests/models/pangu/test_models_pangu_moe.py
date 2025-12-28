import sys
import types
import contextlib
import unittest
import importlib
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.nn import Parameter


# -------------------------
# Helpers / test doubles
# -------------------------
class _DummyGroup:
    def __init__(self, world_size=1, rank_in_group=0):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.device_group = self

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("Only dim=0 expected in these UT paths.")
        return x.repeat((self.world_size, 1))


class _DummyRMSNorm(nn.Module):
    """Behaves like vLLM RMSNorm in this module usage:
    - called as norm(x) in decoder when residual is None
    - called as norm(x, residual) -> (x, residual) elsewhere
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        if residual is None:
            return x
        return x, residual


class _LinearReturnsTuple(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x, None


class _SiluAndMulStub(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _AttentionStub(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer_name = kwargs.get("prefix", "attn")

    def forward(self, positions, hidden_states, kv_cache=None, attn_metadata=None):
        return hidden_states


class _FusedMoEStub(nn.Module):
    def __init__(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        reduce_results,
        quant_config=None,
        custom_routing_function=None,
        prefix="",
        **kwargs,
    ):
        super().__init__()
        self.top_k = top_k
        self.global_num_experts = num_experts
        self.expert_map = None
        self.custom_routing_function = custom_routing_function
        self.apply_router_weight_on_input = False

        class _QM:
            def __init__(self):
                self.apply = Mock(side_effect=lambda **kw: kw["x"])

        self.quant_method = _QM()

    @staticmethod
    def make_expert_params_mapping(*args, **kwargs):
        return []

    def forward(self, hidden_states=None, router_logits=None, **kwargs):
        return hidden_states


class _ReplicatedLinearGateStub(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__()
        self.out_features = out_features

    def forward(self, x: torch.Tensor):
        return (
            torch.zeros((x.shape[0], self.out_features), dtype=x.dtype, device=x.device),
            None,
        )


class _IntermediateTensorsDict(dict):
    """Lightweight replacement to avoid entering real vllm.sequence.IntermediateTensors."""
    pass


def _make_hf_config(**overrides):
    base = dict(
        pad_token_id=0,
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        num_attention_heads=2,
        num_key_value_heads=2,
        rope_theta=10000,
        rope_scaling=None,
        max_position_embeddings=128,
        # MLP/MoE
        num_experts=0,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=0,
        intermediate_size=16,
        hidden_act="silu",
        tie_word_embeddings=True,
        mlp_only_layers=[],
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _safe_all_gather_into_tensor(output_tensor, input_tensor, group=None):
    reps = output_tensor.shape[0] // max(1, input_tensor.shape[0])
    output_tensor.copy_(input_tensor.repeat((reps, 1)))


def _safe_reduce_scatter_tensor(x, op, scatter_dim=0, group=None):
    world_size = getattr(group, "world_size", 1) if group is not None else 1
    rank = getattr(group, "rank_in_group", 0) if group is not None else 0
    chunks = torch.tensor_split(x, max(1, world_size), dim=scatter_dim)
    return chunks[min(rank, len(chunks) - 1)].contiguous()


def _install_torch_npu_stub(stack: contextlib.ExitStack) -> None:
    """Install torch_npu / torch.npu stubs ONLY within the stack lifetime."""
    try:
        import torch_npu  # noqa: F401
        return
    except Exception:
        stub = types.ModuleType("torch_npu")
        stub.npu_prefetch = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        stack.enter_context(patch.dict(sys.modules, {"torch_npu": stub}))

        # Some projects rely on torch.npu existing at import time.
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


class TestModelsPanguMOE(unittest.TestCase):
    _PANGU_MOE_MOD = "omni.models.pangu.pangu_pro_moe.pangu_moe"

    def setUp(self):
        super().setUp()
        self._stack = contextlib.ExitStack()

        try:
            # ---- RNG isolation (weak but real cross-UT coupling) ----
            rng_state = torch.random.get_rng_state()
            self._stack.callback(lambda: torch.random.set_rng_state(rng_state))

            # ---- torch_npu / torch.npu stubs are scoped to THIS test only ----
            _install_torch_npu_stub(self._stack)

            # ---- Import business module lazily (avoid collection-time side effects) ----
            self._pangu_moe_preexisted = self._PANGU_MOE_MOD in sys.modules
            self.M = importlib.import_module(self._PANGU_MOE_MOD)

            # If this test is the first one importing the module (esp. under stubs),
            # unload it after test to avoid leaking stub-bound references.
            def _maybe_unimport():
                if not self._pangu_moe_preexisted:
                    sys.modules.pop(self._PANGU_MOE_MOD, None)

            self._stack.callback(_maybe_unimport)

            # ---- Anti-hang / anti-heavy path patches (scoped) ----
            self._stack.enter_context(
                patch.object(self.M, "IntermediateTensors", _IntermediateTensorsDict, create=True)
            )
            self._stack.enter_context(
                patch.object(
                    self.M,
                    "get_forward_context",
                    return_value=types.SimpleNamespace(attn_metadata=None),
                    create=True,
                )
            )

            # ---- Prevent real dist collectives from blocking (scoped) ----
            if hasattr(self.M, "dist"):
                self._stack.enter_context(
                    patch.object(
                        self.M.dist,
                        "all_gather_into_tensor",
                        side_effect=_safe_all_gather_into_tensor,
                        create=True,
                    )
                )
                # Ensure _functional_collectives exists for compatibility, then patch reduce_scatter_tensor
                if not hasattr(self.M.dist, "_functional_collectives"):
                    fc = types.SimpleNamespace()
                    self._stack.enter_context(patch.object(self.M.dist, "_functional_collectives", fc, create=True))
                self._stack.enter_context(
                    patch.object(
                        self.M.dist._functional_collectives,  # type: ignore[attr-defined]
                        "reduce_scatter_tensor",
                        side_effect=_safe_reduce_scatter_tensor,
                        create=True,
                    )
                )
        except Exception:
            # CRITICAL: setUp failure must rollback everything to avoid polluting subsequent UTs.
            self._stack.close()
            raise

    def tearDown(self):
        with contextlib.suppress(Exception):
            self._stack.close()
        super().tearDown()

    def test_use_h2p_gate_by_dp_world_size(self):
        with patch.object(self.M, "get_dp_group", return_value=_DummyGroup(world_size=1), create=True):
            self.assertFalse(self.M.use_h2p())
        with patch.object(self.M, "get_dp_group", return_value=_DummyGroup(world_size=2), create=True):
            self.assertTrue(self.M.use_h2p())

    def test_mlp_linear_impl_selected_by_h2p_flag(self):
        class _MergedStub(_LinearReturnsTuple): pass
        class _RowStub(_LinearReturnsTuple): pass
        class _H2PMergedStub(_LinearReturnsTuple): pass
        class _H2PRowStub(_LinearReturnsTuple): pass

        common_patches = dict(
            SiluAndMul=_SiluAndMulStub,
            MergedColumnParallelLinear=_MergedStub,
            RowParallelLinear=_RowStub,
            PanguProMoEMergedColumnParallelLinear=_H2PMergedStub,
            PanguProMoERowParallelLinear=_H2PRowStub,
        )

        with patch.object(self.M, "get_dp_group", return_value=_DummyGroup(world_size=1), create=True), \
             patch.multiple(self.M, create=True, **common_patches):
            mlp = self.M.PanguProMoEMLP(
                hidden_size=8, intermediate_size=16, hidden_act="silu",
                quant_config=None, reduce_results=True, prefix="x",
            )
            self.assertIsInstance(mlp.gate_up_proj, _MergedStub)
            self.assertIsInstance(mlp.down_proj, _RowStub)

        with patch.object(self.M, "get_dp_group", return_value=_DummyGroup(world_size=2), create=True), \
             patch.object(self.M, "get_world_group", return_value=_DummyGroup(world_size=2), create=True), \
             patch.multiple(self.M, create=True, **common_patches):
            mlp = self.M.PanguProMoEMLP(
                hidden_size=8, intermediate_size=16, hidden_act="silu",
                quant_config=None, reduce_results=True, prefix="x",
            )
            self.assertIsInstance(mlp.gate_up_proj, _H2PMergedStub)
            self.assertIsInstance(mlp.down_proj, _H2PRowStub)

    def test_mlp_activation_contract_only_supports_silu(self):
        with patch.object(self.M, "get_dp_group", return_value=_DummyGroup(world_size=1), create=True), \
             patch.object(self.M, "MergedColumnParallelLinear", _LinearReturnsTuple, create=True), \
             patch.object(self.M, "RowParallelLinear", _LinearReturnsTuple, create=True), \
             patch.object(self.M, "SiluAndMul", _SiluAndMulStub, create=True):
            with self.assertRaises(ValueError):
                self.M.PanguProMoEMLP(
                    hidden_size=8, intermediate_size=16, hidden_act="gelu",
                    quant_config=None, reduce_results=True, prefix="x",
                )

    def test_topk_wrapper_routes_two_branches_and_returns_expected_contract(self):
        with patch.object(self.M, "get_ep_group", return_value=_DummyGroup(world_size=1, rank_in_group=0), create=True):
            num_tokens = 4
            global_num_experts = 8
            topk = 2
            hidden = torch.randn(num_tokens, 8)
            gating = torch.randn(num_tokens, global_num_experts)

            # CRITICAL: do NOT permanently mutate module globals
            with patch.object(self.M, "_ROUTER_SCALE", torch.ones((1, global_num_experts)), create=True):
                fn8 = self.M.topk_wrapper(8)
                w8, ids8 = fn8(hidden, gating, topk=topk, global_num_experts=global_num_experts)
                self.assertEqual(tuple(w8.shape), (num_tokens, topk))
                self.assertEqual(tuple(ids8.shape), (num_tokens, topk))
                self.assertEqual(ids8.dtype, torch.int32)

                fn5 = self.M.topk_wrapper(5)
                w5, ids5 = fn5(hidden, gating, topk=topk, global_num_experts=global_num_experts)
                self.assertEqual(tuple(w5.shape), (num_tokens, topk))
                self.assertEqual(tuple(ids5.shape), (num_tokens, topk))
                self.assertEqual(ids5.dtype, torch.int32)

    def test_sparse_moe_block_switches_execution_path_by_h2p_and_tp_reduce_rule(self):
        hf_config = _make_hf_config(
            num_experts=4, num_experts_per_tok=2,
            hidden_size=8, moe_intermediate_size=16,
            shared_expert_intermediate_size=0, hidden_act="silu",
        )
        x = torch.randn(3, hf_config.hidden_size)

        common = [
            patch.object(self.M, "FusedMoE", _FusedMoEStub, create=True),
            patch.object(self.M, "ReplicatedLinear", _ReplicatedLinearGateStub, create=True),
            patch.object(self.M, "get_tensor_model_parallel_world_size", return_value=1, create=True),
            patch.object(self.M, "get_ep_group", return_value=_DummyGroup(world_size=1, rank_in_group=0), create=True),
        ]

        # 非 H2P：应走 experts(...) 路径（quant_method.apply 不应被调用）
        with patch.object(self.M, "get_dp_group", return_value=_DummyGroup(world_size=1), create=True), \
             patch.object(self.M, "tensor_model_parallel_all_reduce", new=Mock(side_effect=lambda t: t), create=True), \
             contextlib.ExitStack() as s:
            for p in common:
                s.enter_context(p)
            blk = self.M.PanguProMoESparseMoeBlock(config=hf_config, quant_config=None, prefix="m.layers.0.mlp")
            out = blk(x, attn_metadata=None)
            self.assertEqual(tuple(out.shape), tuple(x.shape))
            self.assertFalse(blk.experts.quant_method.apply.called)

        # H2P：应走 experts.quant_method.apply(...) 路径
        with patch.object(self.M, "get_dp_group", return_value=_DummyGroup(world_size=2), create=True), \
             patch.object(self.M, "get_world_group", return_value=_DummyGroup(world_size=2), create=True), \
             patch.object(self.M, "tensor_model_parallel_all_reduce", new=Mock(side_effect=lambda t: t), create=True), \
             contextlib.ExitStack() as s:
            for p in common:
                s.enter_context(p)
            blk = self.M.PanguProMoESparseMoeBlock(config=hf_config, quant_config=None, prefix="m.layers.0.mlp")
            out = blk(x, attn_metadata=None)
            self.assertEqual(tuple(out.shape), tuple(x.shape))
            self.assertTrue(blk.experts.quant_method.apply.called)

    def test_decoder_layer_forward_runs_end_to_end_in_dense_or_moe_mode(self):
        class _MergedStub(_LinearReturnsTuple): pass
        class _RowStub(_LinearReturnsTuple): pass

        def _run_one(config_obj, prefix):
            with patch.object(self.M, "PanguProMoEAttention", _AttentionStub, create=True), \
                 patch.object(self.M, "RMSNorm", _DummyRMSNorm, create=True), \
                 patch.object(self.M, "SiluAndMul", _SiluAndMulStub, create=True), \
                 patch.object(self.M, "MergedColumnParallelLinear", _MergedStub, create=True), \
                 patch.object(self.M, "RowParallelLinear", _RowStub, create=True), \
                 patch.object(self.M, "get_tp_group", return_value=_DummyGroup(world_size=1, rank_in_group=0), create=True), \
                 patch.object(self.M, "get_dp_group", return_value=_DummyGroup(world_size=1), create=True), \
                 patch.object(self.M, "get_tensor_model_parallel_world_size", return_value=1, create=True), \
                 patch.object(self.M, "FusedMoE", _FusedMoEStub, create=True), \
                 patch.object(self.M, "ReplicatedLinear", _ReplicatedLinearGateStub, create=True), \
                 patch.object(self.M, "tensor_model_parallel_all_reduce", Mock(side_effect=lambda t: t), create=True), \
                 patch.object(self.M, "get_ep_group", return_value=_DummyGroup(world_size=1, rank_in_group=0), create=True):
                layer = self.M.PanguProMoEDecoderLayer(config=config_obj, cache_config=None, quant_config=None, prefix=prefix)
                positions = torch.arange(0, 3, dtype=torch.long)
                hidden = torch.randn(3, config_obj.hidden_size)
                out, residual = layer(
                    positions=positions,
                    hidden_states=hidden,
                    residual=None,
                    kv_cache=None,
                    attn_metadata=None,
                    h2p_unpad_idx=None,
                    h2p_pad_idx=None,
                    is_start_layer=True,
                )
                self.assertEqual(tuple(out.shape), tuple(hidden.shape))
                self.assertIsNotNone(residual)

        dense_cfg = _make_hf_config(num_experts=0, hidden_size=8, intermediate_size=16, hidden_act="silu", num_hidden_layers=1)
        _run_one(dense_cfg, prefix="model.layers.0")

        moe_cfg = _make_hf_config(
            num_experts=4, num_experts_per_tok=2, hidden_size=8, moe_intermediate_size=16,
            shared_expert_intermediate_size=0, hidden_act="silu", num_hidden_layers=1, mlp_only_layers=[]
        )
        _run_one(moe_cfg, prefix="model.layers.0")

    def test_decoder_layer_h2p_padding_and_tp_collectives_called_when_needed(self):
        tp_group = _DummyGroup(world_size=2, rank_in_group=0)
        world_group = _DummyGroup(world_size=2, rank_in_group=0)

        cfg = _make_hf_config(
            num_experts=4, num_experts_per_tok=2, hidden_size=8,
            moe_intermediate_size=16, shared_expert_intermediate_size=0,
            hidden_act="silu", num_hidden_layers=1,
        )

        with patch.object(self.M, "get_dp_group", return_value=_DummyGroup(world_size=2), create=True), \
             patch.object(self.M, "get_tp_group", return_value=tp_group, create=True), \
             patch.object(self.M, "get_world_group", return_value=world_group, create=True), \
             patch.object(self.M, "get_tensor_model_parallel_world_size", return_value=2, create=True), \
             patch.object(self.M, "PanguProMoEAttention", _AttentionStub, create=True), \
             patch.object(self.M, "RMSNorm", _DummyRMSNorm, create=True), \
             patch.object(self.M, "FusedMoE", _FusedMoEStub, create=True), \
             patch.object(self.M, "ReplicatedLinear", _ReplicatedLinearGateStub, create=True), \
             patch.object(self.M, "tensor_model_parallel_all_reduce", Mock(side_effect=lambda t: t), create=True), \
             patch.object(self.M, "get_ep_group", return_value=_DummyGroup(world_size=1, rank_in_group=0), create=True), \
             patch.object(self.M.dist, "all_gather_into_tensor", Mock(side_effect=_safe_all_gather_into_tensor), create=True), \
             patch.object(self.M.dist._functional_collectives, "reduce_scatter_tensor", Mock(side_effect=_safe_reduce_scatter_tensor), create=True):

            layer = self.M.PanguProMoEDecoderLayer(config=cfg, cache_config=None, quant_config=None, prefix="model.layers.0")

            h2p_unpad_idx = torch.arange(0, 2, dtype=torch.int32)
            h2p_pad_idx = torch.tensor([0, 1, 0, 0], dtype=torch.int32)  # padded to 4, divisible by tp=2

            positions = torch.arange(0, 2, dtype=torch.long)
            hidden = torch.randn(2, cfg.hidden_size)

            out, residual = layer(
                positions=positions,
                hidden_states=hidden,
                residual=None,
                kv_cache=None,
                attn_metadata=None,
                h2p_unpad_idx=h2p_unpad_idx,
                h2p_pad_idx=h2p_pad_idx,
                is_start_layer=True,
            )

            self.assertEqual(out.shape[-1], cfg.hidden_size)
            self.assertIsNotNone(residual)
            self.assertTrue(self.M.dist.all_gather_into_tensor.called)
            self.assertTrue(self.M.dist._functional_collectives.reduce_scatter_tensor.called)

    def test_model_forward_pipeline_contract_first_vs_not_first_and_intermediate_on_not_last(self):
        model = self.M.PanguProMoEModel.__new__(self.M.PanguProMoEModel)
        nn.Module.__init__(model)

        hidden_size = 8

        class _LayerStub(nn.Module):
            def forward(
                self, positions, hidden_states, residual,
                kv_cache=None, attn_metadata=None,
                h2p_unpad_idx=None, h2p_pad_idx=None, is_start_layer=False
            ):
                if residual is None:
                    residual = hidden_states
                return hidden_states, residual

        model.start_layer = 0
        model.end_layer = 1
        model.layers = nn.ModuleList([_LayerStub()])
        model.norm = _DummyRMSNorm()
        model.embed_tokens = nn.Embedding(16, hidden_size)

        pp = types.SimpleNamespace(is_first_rank=True, is_last_rank=False)
        with patch.object(self.M, "get_pp_group", return_value=pp, create=True), \
             patch.object(self.M, "get_dp_group", return_value=_DummyGroup(world_size=1), create=True):
            input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
            positions = torch.tensor([0, 1, 2], dtype=torch.long)
            out = self.M.PanguProMoEModel.forward(
                model, input_ids, positions, kv_caches=None, attn_metadata=None,
                intermediate_tensors=None, inputs_embeds=None
            )
            self.assertIsInstance(out, dict)
            self.assertIn("hidden_states", out)
            self.assertIn("residual", out)

        pp = types.SimpleNamespace(is_first_rank=False, is_last_rank=False)
        with patch.object(self.M, "get_pp_group", return_value=pp, create=True), \
             patch.object(self.M, "get_dp_group", return_value=_DummyGroup(world_size=1), create=True):
            input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
            positions = torch.tensor([0, 1, 2], dtype=torch.long)
            inter = {"hidden_states": torch.randn(3, hidden_size), "residual": torch.randn(3, hidden_size)}
            out = self.M.PanguProMoEModel.forward(
                model, input_ids, positions, kv_caches=None, attn_metadata=None,
                intermediate_tensors=inter, inputs_embeds=None
            )
            self.assertIsInstance(out, dict)
            self.assertIn("hidden_states", out)
            self.assertIn("residual", out)

    def test_model_forward_h2p_pad_unpad_contract_on_output_shape(self):
        model = self.M.PanguProMoEModel.__new__(self.M.PanguProMoEModel)
        nn.Module.__init__(model)

        hidden_size = 8

        class _LayerStub(nn.Module):
            def forward(
                self, positions, hidden_states, residual,
                kv_cache=None, attn_metadata=None,
                h2p_unpad_idx=None, h2p_pad_idx=None, is_start_layer=False
            ):
                if residual is None:
                    residual = hidden_states
                return hidden_states, residual

        model.start_layer = 0
        model.end_layer = 1
        model.layers = nn.ModuleList([_LayerStub()])
        model.norm = _DummyRMSNorm()
        model.embed_tokens = nn.Embedding(16, hidden_size)

        tp_group = _DummyGroup(world_size=2, rank_in_group=0)

        def _tp_all_gather(x, dim=0):
            target_len = 6
            if x.shape[0] >= target_len:
                return x
            pad = torch.zeros((target_len - x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=0)

        tp_group.all_gather = Mock(side_effect=_tp_all_gather)

        attn_metadata = types.SimpleNamespace(max_num_tokens_across_dp=5)
        pp = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)

        with patch.object(self.M, "get_pp_group", return_value=pp, create=True), \
             patch.object(self.M, "get_dp_group", return_value=_DummyGroup(world_size=2), create=True), \
             patch.object(self.M, "get_tp_group", return_value=tp_group, create=True):
            input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
            positions = torch.tensor([0, 1, 2], dtype=torch.long)
            out = self.M.PanguProMoEModel.forward(
                model, input_ids, positions, kv_caches=None,
                attn_metadata=attn_metadata, intermediate_tensors=None,
                inputs_embeds=None
            )
            self.assertEqual(out.shape[0], input_ids.shape[0])
            self.assertEqual(out.shape[1], hidden_size)
            self.assertTrue(tp_group.all_gather.called)

    def test_for_causal_lm_tie_embeddings_and_logits_sample_delegation(self):
        # Avoid importing compilation.wrapper via string patch (module import side effects).
        # Patch the symbol in PANGU_MOE module namespace instead.
        class _NoOpCompilerWrapper:
            def __init__(self, vllm_config, dynamic_arg_dims):
                self.compiled_model = None
                self.cached_compiled_models = {}
                self.vllm_config = vllm_config
                self.dynamic_arg_dims = dynamic_arg_dims
                self.do_not_compile = True

        class _ModelStub(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.embed_tokens = nn.Embedding(16, 8)
                self.start_layer = 0
                self.layers = [types.SimpleNamespace(layer_name="model.layers.0.self_attn.attn")]
                self.make_empty_intermediate_tensors = Mock()

            def get_input_embeddings(self, input_ids):
                return self.embed_tokens(input_ids)

            def forward(self, input_ids, positions, kv_caches=None, attn_metadata=None, intermediate_tensors=None, inputs_embeds=None):
                if inputs_embeds is not None:
                    return inputs_embeds
                return self.get_input_embeddings(input_ids)

        class _LMHeadStub(nn.Module):
            def __init__(self, vocab, hidden, *args, **kwargs):
                super().__init__()
                self.weight = Parameter(torch.empty(vocab, hidden))

        logits_processor = Mock(return_value=torch.randn(3, 16))
        sampler = Mock(return_value=types.SimpleNamespace(next_tokens=torch.tensor([1])))

        hf = _make_hf_config(tie_word_embeddings=True, vocab_size=16, hidden_size=8)
        vllm_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(hf_config=hf),
            cache_config=None,
            quant_config=None,
        )

        with patch(
            "omni.adaptors.vllm.compilation.decorators."
            "TorchNpuCompilerWrapperWithCustomDispatcher.__init__",
            new=_NoOpCompilerWrapper.__init__,
            create=True,
        ), \
             patch.object(self.M, "patch_fused_moe_ops", Mock(), create=True), \
             patch.object(self.M, "PanguProMoEModel", _ModelStub, create=True), \
             patch.object(self.M, "ParallelLMHead", _LMHeadStub, create=True), \
             patch.object(self.M, "LogitsProcessor", Mock(return_value=logits_processor), create=True), \
             patch.object(self.M, "get_sampler", Mock(return_value=sampler), create=True):
            m = self.M.PanguProMoEForCausalLM(vllm_config=vllm_config, prefix="")

            self.assertIs(m.lm_head.weight, m.model.embed_tokens.weight)

            hs = torch.randn(3, hf.hidden_size)
            sm = object()
            _ = m.compute_logits(hs, sm)
            logits_processor.assert_called_once()

            _ = m.sample(torch.randn(3, hf.vocab_size), sm)
            sampler.assert_called_once()

    def test_should_use_eager_mode_dispatch_by_attn_metadata_and_state(self):
        m = self.M.PanguProMoEForCausalLM.__new__(self.M.PanguProMoEForCausalLM)
        layer_name = "model.layers.0.self_attn.attn"
        m.model = types.SimpleNamespace(start_layer=0, layers=[types.SimpleNamespace(layer_name=layer_name)])

        self.assertTrue(self.M.PanguProMoEForCausalLM.should_use_eager_mode(m, attn_metadata=None))

        non_decode = types.SimpleNamespace(attn_state=object())
        decode = types.SimpleNamespace(attn_state=self.M.AscendAttentionState.DecodeOnly)
        self.assertTrue(self.M.PanguProMoEForCausalLM.should_use_eager_mode(m, attn_metadata={layer_name: non_decode}))
        self.assertFalse(self.M.PanguProMoEForCausalLM.should_use_eager_mode(m, attn_metadata={layer_name: decode}))

    def test_load_weights_minimal_skip_and_remap_contract(self):
        m = self.M.PanguProMoEForCausalLM.__new__(self.M.PanguProMoEForCausalLM)
        m.config = types.SimpleNamespace(num_experts=0)
        m.model = types.SimpleNamespace(end_layer=1)

        p = Parameter(torch.empty(1))
        p.weight_loader = Mock()
        params = {"model.layers.0.self_attn.attn.key_antiquant_scale": p}

        def _named_parameters():
            for k, v in params.items():
                yield k, v

        m.named_parameters = _named_parameters  # type: ignore[assignment]

        with patch.object(self.M, "get_tp_group", return_value=_DummyGroup(world_size=1, rank_in_group=0), create=True), \
             patch.object(self.M, "FusedMoE", _FusedMoEStub, create=True), \
             patch.object(self.M, "default_weight_loader", Mock(), create=True), \
             patch.object(self.M, "is_pp_missing_parameter", Mock(return_value=False), create=True), \
             patch.object(self.M, "logger", types.SimpleNamespace(warning_once=Mock()), create=True):
            weights = [
                ("model.rotary_emb.inv_freq", torch.tensor(1.0)),  # skipped
                ("model.layers.9.self_attn.q_proj.weight", torch.randn(1)),  # skipped by end_layer
                ("model.layers.0.self_attn.k_proj.kv_cache_scale", torch.tensor([2.0])),
            ]
            loaded = self.M.PanguProMoEForCausalLM.load_weights(m, weights)

            self.assertGreaterEqual(p.weight_loader.call_count, 1)
            self.assertIn("model.layers.0.self_attn.attn.key_antiquant_scale", loaded)
            self.assertNotIn("model.rotary_emb.inv_freq", loaded)


if __name__ == "__main__":
    unittest.main()
