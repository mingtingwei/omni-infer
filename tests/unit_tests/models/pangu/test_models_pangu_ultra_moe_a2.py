import os
import sys
import types
import contextlib
import pytest  # noqa: F401  (框架中保留，不强依赖)
import unittest
from unittest.mock import patch
import importlib

import torch
from torch import nn


# ---------------------------------------------------------------------
# Dummy / Spy helpers (只看护分支路由与 IO 形态)
# ---------------------------------------------------------------------
class DummyGroup:
    def __init__(self, world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank


class CallLog:
    def __init__(self):
        self.calls = []  # list[(name, args, kwargs)]

    def add(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def names(self):
        return [n for n, _, _ in self.calls]

    def count(self, name):
        return sum(1 for n, _, _ in self.calls if n == name)


class DummySiluAndMul:
    """SiluAndMul 的最小替身：保持 shape，不关心数值。"""

    def __call__(self, x, quant_symbol: bool):
        if isinstance(x, dict):
            return x["x_int8"]
        return x


class DummyMergedColumnLinear(nn.Module):
    """AscendMergedColumnParallelLinear 最小替身。"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight_scale = torch.tensor(1.0)  # 代码中会读取

    def forward(self, x):
        # gate_up_proj.forward 返回 (gate_up, None)
        if isinstance(x, dict):
            return x["x_int8"], None
        return x, None


class DummyRowLinear(nn.Module):
    """AscendRowParallelLinear 最小替身。"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x, None


class DummyRMSNorm(nn.Module):
    """兼容两种调用方式：
    - y = norm(x)
    - y, res = norm(x, res, quant_symbol=...)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.log = CallLog()

    def forward(self, x, residual=None, **kwargs):
        self.log.add("rmsnorm", x, residual, **kwargs)
        if residual is None:
            return x
        return x, residual


class DummyAttn(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.log = CallLog()

    def forward(self, *, positions, hidden_states, kv_cache, attn_metadata):
        self.log.add("attn", positions, hidden_states, kv_cache, attn_metadata)
        return hidden_states


class DummyDenseMLP(nn.Module):
    """DecoderLayer dense mlp 替身：记录调用次数，返回 (hidden, residual)。"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.log = CallLog()

    def forward(self, hidden, residual, attn_metadata, *args, **kwargs):
        self.log.add("mlp", hidden, residual, attn_metadata, *args, **kwargs)
        return hidden, residual


class DummyMoE(nn.Module):
    """MoE 替身：签名更宽松。"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, hidden, residual, attn_metadata, *args, **kwargs):
        return hidden, residual


class DummyParam:
    """用于 load_weights 的参数替身。"""

    def __init__(self, name, log: CallLog):
        self.name = name
        self.log = log

    def weight_loader(self, param, weight, shard_id=None, **kwargs):
        # 只记录 shard_id 是否正确，不关心 weight 内容
        self.log.add("weight_loader", self.name, shard_id, **kwargs)


def _make_minimal_config(**overrides):
    base = dict(
        hidden_size=4,
        intermediate_size=8,
        hidden_act="silu",
        num_attention_heads=2,
        attention_qk_dim=2,
        attention_qk_rope_dim=2,
        attention_v_dim=2,
        attention_kv_lora_dim=2,
        rms_norm_eps=1e-5,
        vocab_size=32,
        num_hidden_layers=4,
        num_dense_layers=1,
        moe_layer_freq=1,
        num_routed_experts=None,
        sandwich_norm=False,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _make_torch_npu_stub():
    m = types.ModuleType("torch_npu")
    m.npu_prefetch = lambda *args, **kwargs: None
    # 默认给一个可用的 dynamic_quant（各 test 里也会用 spy 覆盖）
    m.npu_dynamic_quant = lambda x: (x, torch.ones((x.shape[0],), dtype=torch.float32, device=x.device))
    return m


def _make_torch_npu_attr_stub():
    # torch.npu 的最小替身，避免在 UT 中依赖真实 NPU runtime
    return types.SimpleNamespace(
        config=types.SimpleNamespace(allow_internal_format=False),
        Stream=lambda *args, **kwargs: None,
        current_stream=lambda: None,
        stream=lambda s: contextlib.nullcontext(),
    )


class TestModelsPanguUltraMOEA2(unittest.TestCase):

    @unittest.skip("debug only; not part of CI gate")
    def test_print_env(self):
        import sys, transformers
        print("exe=", sys.executable)
        print("transformers=", transformers.__version__, transformers.__file__)
        print("has ProcessorMixin=", hasattr(transformers, "ProcessorMixin"))
        from transformers import ProcessorMixin  # noqa: F401
    # -----------------------------
    # patch helper（强隔离：所有 patch 都 addCleanup，避免 setUp 中途异常导致泄露）
    # -----------------------------
    def _start_patch(self, p):
        obj = p.start()
        self.addCleanup(lambda: self._safe_stop(p))
        return obj

    @staticmethod
    def _safe_stop(p):
        try:
            p.stop()
        except Exception:
            pass

    def _patch_attr(self, obj, attr, new_value, create=False):
        return self._start_patch(patch.object(obj, attr, new=new_value, create=create))

    def setUp(self):
        super().setUp()

        # 0) 先预热 transformers 的导出（避免后续 patch torch.npu 影响其 lazy import）
        from transformers import ProcessorMixin  # noqa: F401

        # 1) RNG 隔离
        rng_state = torch.get_rng_state()
        self.addCleanup(lambda: torch.set_rng_state(rng_state))

        # 2) 环境变量隔离
        self._start_patch(patch.dict(os.environ, {"ROLE": "prefill"}, clear=False))

        # 3) torch_npu：优先使用真实模块；只有 import 失败才 stub（避免误覆盖）
        try:
            import torch_npu as real_torch_npu  # noqa: F401
            self._torch_npu_stub = real_torch_npu
        except Exception:
            self._torch_npu_stub = _make_torch_npu_stub()
            self._start_patch(patch.dict(sys.modules, {"torch_npu": self._torch_npu_stub}, clear=False))

        # 4) 关键：先 import 被测模块（避免 torch.npu stub 干扰 vllm->transformers 导入链）
        self.mut = importlib.import_module("omni.models.pangu.pangu_ultra_moe_a2")

        # 5) 再做 torch.npu 的“最小补齐”，不要替换整对象（降低对外部库/其它 UT 的影响面）
        stub_npu = _make_torch_npu_attr_stub()
        if not hasattr(torch, "npu"):
            # 环境里没有 torch.npu 才创建
            self._start_patch(patch.object(torch, "npu", new=stub_npu, create=True))
        else:
            # 环境里已有 torch.npu：只 patch UT 需要用到的几个入口
            self._patch_attr(torch.npu, "Stream", stub_npu.Stream, create=True)
            self._patch_attr(torch.npu, "current_stream", stub_npu.current_stream, create=True)
            self._patch_attr(torch.npu, "stream", stub_npu.stream, create=True)
            # config：只保证 allow_internal_format 存在且可读
            if hasattr(torch.npu, "config"):
                self._patch_attr(torch.npu, "config", stub_npu.config, create=True)
            else:
                self._patch_attr(torch.npu, "config", stub_npu.config, create=True)

        # 6) 确保 mut.torch_npu 可控
        self._patch_attr(self.mut, "torch_npu", self._torch_npu_stub, create=True)

        # --- 通用 patch：把 NPU/分布式/外部依赖替成最小可跑版本 ---
        self._patch_attr(self.mut, "get_npu_device_count", lambda: 1, create=True)
        self._patch_attr(self.mut, "get_local_group_size", lambda: 1, create=True)
        self._patch_attr(self.mut, "get_local_group_rank", lambda: 0, create=True)

        # groups（每个 test 各自一份，不会跨用例共享）
        self._dp = DummyGroup(world_size=1, rank_in_group=0)
        self._ep = DummyGroup(world_size=1, rank_in_group=0)
        self._wg = DummyGroup(world_size=2, rank_in_group=0)
        self._pp = DummyGroup(world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True)

        self._patch_attr(self.mut, "get_dp_group", lambda: self._dp, create=True)
        self._patch_attr(self.mut, "get_ep_group", lambda: self._ep, create=True)
        self._patch_attr(self.mut, "get_world_group", lambda: self._wg, create=True)
        self._patch_attr(self.mut, "get_pp_group", lambda: self._pp, create=True)

        # platform device
        if hasattr(self.mut, "current_platform") and hasattr(self.mut.current_platform, "device_type"):
            self._patch_attr(self.mut.current_platform, "device_type", "cpu", create=True)

        # dist all_reduce
        if hasattr(self.mut, "dist"):
            self._patch_attr(self.mut.dist, "all_reduce", lambda *a, **k: None, create=True)

        # torch_npu dynamic quant（返回 pertoken_scale，满足 forward_no_tp 路径）
        self._patch_attr(
            self.mut.torch_npu,
            "npu_dynamic_quant",
            lambda x: (x, torch.ones((x.shape[0],), dtype=torch.float32, device=x.device)),
            create=True,
        )

        # comm ops（默认 identity）
        self._patch_attr(self.mut, "all_gather_local", lambda x, **k: x, create=True)
        self._patch_attr(self.mut, "reduce_scatter_local", lambda x, **k: x, create=True)
        self._patch_attr(self.mut, "all_gather_two_stage", lambda x, **k: x, create=True)
        self._patch_attr(self.mut, "reduce_scatter_two_stage", lambda x, **k: x, create=True)
        self._patch_attr(self.mut, "all_gather_pipeline", lambda x, **k: x, create=True)
        self._patch_attr(self.mut, "reduce_scatter_pipeline", lambda x, **k: x, create=True)
        self._patch_attr(self.mut, "all_gather_round_pipeline", lambda x, **k: x, create=True)
        self._patch_attr(self.mut, "reduce_scatter_round_pipeline", lambda x, **k: x, create=True)

        # model_extra_config：只保留用到的开关（注意：这个对象是 patch 注入的，测试结束自动回滚）
        operator_opt_config = types.SimpleNamespace(
            enable_round_pipeline_comm=False,
            enable_pipeline_comm=False,
            prefill_moe_all_to_all=False,
            use_mlaprolog=False,
            enable_dsa=False,
            use_prefetch=False,
            merge_qkv=False,
        )
        parall_config = types.SimpleNamespace(dense_mlp_tp_size=1)
        self._patch_attr(
            self.mut,
            "model_extra_config",
            types.SimpleNamespace(operator_opt_config=operator_opt_config, parall_config=parall_config),
            create=True,
        )

        # layers / ops（替换成 dummy，避免外部依赖）
        self._patch_attr(self.mut, "SiluAndMul", DummySiluAndMul, create=True)
        self._patch_attr(self.mut, "AscendMergedColumnParallelLinear", DummyMergedColumnLinear, create=True)
        self._patch_attr(self.mut, "AscendRowParallelLinear", DummyRowLinear, create=True)
        self._patch_attr(self.mut, "RMSNorm", DummyRMSNorm, create=True)
        self._patch_attr(self.mut, "DeepseekMLA", DummyAttn, create=True)
        self._patch_attr(self.mut, "DeepseekMoE", DummyMoE, create=True)

        # vllm objects：用于 PP / IntermediateTensors 行为
        self._patch_attr(self.mut, "IntermediateTensors", dict, create=True)
        self._patch_attr(self.mut, "tensor_model_parallel_all_gather", lambda x, dim=0: x, create=True)
        self._patch_attr(self.mut, "is_pp_missing_parameter", lambda name, module: False, create=True)

        # FusedMoE expert mapping 简化为空
        if hasattr(self.mut, "FusedMoE"):
            try:
                self._patch_attr(self.mut.FusedMoE, "make_expert_params_mapping", lambda **k: [], create=True)
            except Exception:
                pass

    # -----------------------------
    # 1) ParallelPanguUltraMoEMLP
    # -----------------------------
    def test_parallel_mlp_init_unsupported_activation_raises(self):
        with self.assertRaises(ValueError):
            self.mut.ParallelPanguUltraMoEMLP(
                hidden_size=8,
                intermediate_size=16,
                hidden_act="gelu",
                tp_parallel="no_tp",
                quant_config=None,
                prefix="x",
            )

    @unittest.expectedFailure
    def test_parallel_mlp_init_role_env_missing_should_not_crash(self):
        # 设计意图：不应强依赖环境变量 ROLE；当前实现会 KeyError
        old = os.environ.pop("ROLE", None)
        try:
            self.mut.ParallelPanguUltraMoEMLP(
                hidden_size=8,
                intermediate_size=16,
                hidden_act="silu",
                tp_parallel="no_tp",
                quant_config=None,
                prefix="x",
            )
        finally:
            if old is not None:
                os.environ["ROLE"] = old

    def test_parallel_mlp_forward_routes_no_tp_or_no_communication(self):
        qlog = CallLog()

        def dyn_quant_spy(x):
            qlog.add("dyn_quant", x)
            return x, torch.ones((x.shape[0],), dtype=torch.float32, device=x.device)

        self._patch_attr(self.mut.torch_npu, "npu_dynamic_quant", dyn_quant_spy, create=True)

        mlp = self.mut.ParallelPanguUltraMoEMLP(
            hidden_size=4,
            intermediate_size=8,
            hidden_act="silu",
            tp_parallel="no_tp",
            quant_config=None,
            prefix="mlp",
        )
        x = torch.randn(3, 4)
        residual = torch.randn(3, 4)

        out, res2 = mlp.forward(x, residual, attn_metadata=None, pertoken_scale=torch.ones(3), no_communication=False)
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertIs(res2, residual)
        self.assertEqual(qlog.count("dyn_quant"), 0)

        mlp2 = self.mut.ParallelPanguUltraMoEMLP(
            hidden_size=4,
            intermediate_size=8,
            hidden_act="silu",
            tp_parallel="global",
            quant_config=None,
            prefix="mlp2",
        )
        sentinel = object()
        mlp2.forward_no_tp = lambda *a, **k: (sentinel, residual)
        out2, _ = mlp2.forward(x, residual, attn_metadata=None, pertoken_scale=torch.ones(3), no_communication=True)
        self.assertIs(out2, sentinel)

    def test_parallel_mlp_forward_local_tp_prefill_dp_padding_and_unpad(self):
        self._dp.world_size = 2

        def all_reduce_set_max(t, *a, **k):
            t.fill_(5)
            return None

        if hasattr(self.mut, "dist"):
            self._patch_attr(self.mut.dist, "all_reduce", all_reduce_set_max, create=True)

        mlp = self.mut.ParallelPanguUltraMoEMLP(
            hidden_size=4,
            intermediate_size=8,
            hidden_act="silu",
            tp_parallel="local",
            quant_config=None,
            prefix="mlp",
        )
        x = torch.randn(3, 4)
        residual = torch.randn(3, 4)
        attn = types.SimpleNamespace(prefill=True)

        out, res2 = mlp.forward(x, residual, attn_metadata=attn)
        self.assertEqual(out.shape[0], 3)
        self.assertEqual(out.shape[1], 4)
        self.assertIs(res2, residual)

    def test_parallel_mlp_forward_global_tp_decode_communication_routing(self):
        cases = [
            (True, False, "all_gather_round_pipeline", "reduce_scatter_round_pipeline"),
            (False, True, "all_gather_pipeline", "reduce_scatter_pipeline"),
            (False, False, "all_gather_two_stage", "reduce_scatter_two_stage"),
        ]

        for round_comm, pipeline_comm, expect_gather, expect_scatter in cases:
            with self.subTest(round_comm=round_comm, pipeline_comm=pipeline_comm):
                call = CallLog()

                self._patch_attr(
                    self.mut,
                    "all_gather_round_pipeline",
                    lambda x, **k: (call.add("all_gather_round_pipeline"), x)[1],
                    create=True,
                )
                self._patch_attr(
                    self.mut,
                    "reduce_scatter_round_pipeline",
                    lambda x, **k: (call.add("reduce_scatter_round_pipeline"), x)[1],
                    create=True,
                )
                self._patch_attr(
                    self.mut,
                    "all_gather_pipeline",
                    lambda x, **k: (call.add("all_gather_pipeline"), x)[1],
                    create=True,
                )
                self._patch_attr(
                    self.mut,
                    "reduce_scatter_pipeline",
                    lambda x, **k: (call.add("reduce_scatter_pipeline"), x)[1],
                    create=True,
                )
                self._patch_attr(
                    self.mut,
                    "all_gather_two_stage",
                    lambda x, **k: (call.add("all_gather_two_stage"), x)[1],
                    create=True,
                )
                self._patch_attr(
                    self.mut,
                    "reduce_scatter_two_stage",
                    lambda x, **k: (call.add("reduce_scatter_two_stage"), x)[1],
                    create=True,
                )

                self.mut.model_extra_config.operator_opt_config.enable_round_pipeline_comm = round_comm
                self.mut.model_extra_config.operator_opt_config.enable_pipeline_comm = pipeline_comm

                if pipeline_comm:
                    self._patch_attr(
                        self.mut,
                        "all_gather_two_stage",
                        lambda t, **k: torch.cat([t, t], dim=0),
                        create=True,
                    )

                mlp = self.mut.ParallelPanguUltraMoEMLP(
                    hidden_size=4,
                    intermediate_size=8,
                    hidden_act="silu",
                    tp_parallel="global",
                    quant_config=None,
                    prefix="mlp",
                )
                x = torch.randn(4, 4)
                residual = torch.randn(4, 4)
                attn = types.SimpleNamespace(prefill=False)  # decode

                out, res2 = mlp.forward(x, residual, attn_metadata=attn)
                self.assertEqual(tuple(out.shape), tuple(x.shape))
                self.assertIs(res2, residual)

                self.assertIn(expect_gather, call.names())
                self.assertIn(expect_scatter, call.names())

    # -----------------------------
    # 2) PanguUltraMoEDecoderLayer
    # -----------------------------
    def test_decoder_layer_mlp_type_selection_moe_vs_dense(self):
        cfg_moe = _make_minimal_config(num_routed_experts=4, num_dense_layers=1, moe_layer_freq=1)
        layer_moe = self.mut.PanguUltraMoEDecoderLayer(cfg_moe, prefix="model.layers.1")
        self.assertTrue(layer_moe.is_moe)
        self.assertIsInstance(layer_moe.mlp, DummyMoE)

        cfg_dense = _make_minimal_config(num_routed_experts=None, num_dense_layers=1, moe_layer_freq=1)
        layer_dense = self.mut.PanguUltraMoEDecoderLayer(cfg_dense, prefix="model.layers.1")
        self.assertFalse(layer_dense.is_moe)
        self.assertTrue(hasattr(layer_dense.mlp, "forward"))

    def test_decoder_layer_forward_dict_metadata_and_residual_branches(self):
        cfg = _make_minimal_config(num_routed_experts=None)
        layer = self.mut.PanguUltraMoEDecoderLayer(cfg, prefix="model.layers.1")

        layer.input_layernorm = DummyRMSNorm()
        layer.post_attention_layernorm = DummyRMSNorm()
        layer.self_attn = DummyAttn()
        layer.mlp = DummyDenseMLP()
        layer.is_moe = False

        positions = torch.zeros((3,), dtype=torch.int64)
        hidden = torch.randn(3, 4)

        meta = types.SimpleNamespace(prefill=True)
        meta_dict = {layer.layer_name: meta}

        out, res = layer.forward(positions, hidden, kv_cache=None, attn_metadata=meta_dict, residual=None)
        self.assertEqual(tuple(out.shape), tuple(hidden.shape))
        self.assertEqual(tuple(res.shape), tuple(hidden.shape))
        self.assertIs(layer.self_attn.log.calls[-1][1][-1], meta)

        residual = torch.randn(3, 4)
        out2, res2 = layer.forward(positions, hidden, kv_cache=None, attn_metadata=meta, residual=residual)
        self.assertEqual(tuple(out2.shape), tuple(hidden.shape))
        self.assertEqual(tuple(res2.shape), tuple(residual.shape))

    def test_decoder_layer_prefill_split_path_calls_mlp_multiple_chunks_and_reconcat(self):
        self._dp.world_size = 2
        self.mut.model_extra_config.operator_opt_config.prefill_moe_all_to_all = False

        def all_reduce_set_max(t, *a, **k):
            t.fill_(700)
            return None

        if hasattr(self.mut, "dist"):
            self._patch_attr(self.mut.dist, "all_reduce", all_reduce_set_max, create=True)

        cfg = _make_minimal_config(num_routed_experts=None)
        layer = self.mut.PanguUltraMoEDecoderLayer(cfg, prefix="model.layers.1")
        layer.input_layernorm = DummyRMSNorm()
        layer.post_attention_layernorm = DummyRMSNorm()
        layer.self_attn = DummyAttn()
        dense_mlp = DummyDenseMLP()
        layer.mlp = dense_mlp
        layer.is_moe = False

        positions = torch.zeros((600,), dtype=torch.int64)
        hidden = torch.randn(600, 4)
        residual = torch.randn(600, 4)
        meta = types.SimpleNamespace(prefill=True)

        out, res = layer.forward(positions, hidden, kv_cache=None, attn_metadata=meta, residual=residual)
        self.assertEqual(out.shape[0], 600)
        self.assertEqual(res.shape[0], 600)
        self.assertGreater(dense_mlp.log.count("mlp"), 1)

    @unittest.expectedFailure
    def test_decoder_layer_is_prefill_logic_decode_should_not_take_prefill_path(self):
        self._dp.world_size = 2

        def all_reduce_should_not_be_called(*a, **k):
            raise AssertionError("decode(prefill=False) should NOT call dist.all_reduce for prefill split")

        if hasattr(self.mut, "dist"):
            self._patch_attr(self.mut.dist, "all_reduce", all_reduce_should_not_be_called, create=True)

        cfg = _make_minimal_config(num_routed_experts=None)
        layer = self.mut.PanguUltraMoEDecoderLayer(cfg, prefix="model.layers.1")
        layer.input_layernorm = DummyRMSNorm()
        layer.post_attention_layernorm = DummyRMSNorm()
        layer.self_attn = DummyAttn()
        layer.mlp = DummyDenseMLP()
        layer.is_moe = False

        positions = torch.zeros((10,), dtype=torch.int64)
        hidden = torch.randn(10, 4)
        residual = torch.randn(10, 4)
        meta = types.SimpleNamespace(prefill=False)

        layer.forward(positions, hidden, kv_cache=None, attn_metadata=meta, residual=residual)

    # -----------------------------
    # 3) PanguUltraMoEModel forward（PP 路由 + prefetch）
    # -----------------------------
    def test_model_forward_pp_rank_routing_and_prefetch_wiring(self):
        model = self.mut.PanguUltraMoEModel.__new__(self.mut.PanguUltraMoEModel)
        nn.Module.__init__(model)

        class DummyLayer(nn.Module):
            def __init__(self, name):
                super().__init__()
                self.self_attn = object()
                self.mlp = types.SimpleNamespace(attn_prefetch=None)
                self.name = name
                self.log = CallLog()

            def forward(self, positions, hidden_states, kv_cache, attn_metadata, residual, layer_id, kv_prefetch):
                self.log.add("layer_forward", kv_cache, kv_prefetch, layer_id)
                if residual is None:
                    residual = hidden_states
                return hidden_states, residual

            __call__ = forward

        model.layers = [DummyLayer(f"L{i}") for i in range(4)]
        model.start_layer = 0
        model.end_layer = 2
        model.num_dense_layers = 0
        model.num_hidden_layers = 4
        model.is_init = False

        model.embed_tokens = lambda input_ids, reduce=1: torch.randn(input_ids.shape[0], 4)
        model.norm = DummyRMSNorm()
        model.make_empty_intermediate_tensors = lambda *a, **k: {"hidden_states": None, "residual": None}

        self.mut.model_extra_config.operator_opt_config.use_prefetch = True

        input_ids = torch.zeros((5,), dtype=torch.int64)
        positions = torch.zeros((5,), dtype=torch.int64)
        kv_caches = [object(), object(), object()]
        meta = types.SimpleNamespace(prefill=True)

        self._patch_attr(self.mut, "get_pp_group", lambda: DummyGroup(is_first_rank=True, is_last_rank=False), create=True)
        out = self.mut.PanguUltraMoEModel.forward(model, input_ids, positions, kv_caches, meta, intermediate_tensors=None)
        self.assertIsInstance(out, dict)
        self.assertIn("hidden_states", out)
        self.assertIn("residual", out)

        self.assertIs(model.layers[0].mlp.attn_prefetch, model.layers[1].self_attn)
        self.assertIs(model.layers[1].mlp.attn_prefetch, model.layers[2].self_attn)
        self.assertTrue(model.is_init)

        call = CallLog()
        self._patch_attr(
            self.mut,
            "tensor_model_parallel_all_gather",
            lambda x, dim=0: (call.add("all_gather"), x)[1],
            create=True,
        )
        self._patch_attr(self.mut, "get_pp_group", lambda: DummyGroup(is_first_rank=False, is_last_rank=True), create=True)

        intermediate = {"hidden_states": torch.randn(5, 4), "residual": torch.randn(5, 4)}
        out2 = self.mut.PanguUltraMoEModel.forward(model, input_ids, positions, kv_caches, meta, intermediate_tensors=intermediate)
        self.assertIsInstance(out2, torch.Tensor)
        self.assertEqual(call.count("all_gather"), 1)

    # -----------------------------
    # 4) PanguUltraMoEForCausalLM（forward slicing + eager mode + load_weights）
    # -----------------------------
    def test_causallm_forward_slices_last_token_when_attn_metadata_none(self):
        clm = self.mut.PanguUltraMoEForCausalLM.__new__(self.mut.PanguUltraMoEForCausalLM)
        nn.Module.__init__(clm)

        full_hidden = torch.randn(6, 4)
        clm.model = lambda *a, **k: full_hidden
        clm.return_hidden_states = True
        clm.embedding_bias = None

        seen = {}

        def compute_lmhead_spy(hidden_states, selected_indices=None, embedding_bias=None):
            seen["hidden_shape"] = tuple(hidden_states.shape)
            seen["selected_indices"] = selected_indices
            return torch.zeros((hidden_states.shape[0], 10))

        clm.compute_lmhead = compute_lmhead_spy

        input_ids = torch.zeros((6,), dtype=torch.int64)
        positions = torch.zeros((6,), dtype=torch.int64)

        hs, logits = self.mut.PanguUltraMoEForCausalLM.forward(
            clm, input_ids, positions, kv_caches=None, attn_metadata=None, selected_indices=None, intermediate_tensors=None
        )
        self.assertTrue(torch.equal(hs, full_hidden))
        self.assertEqual(seen["hidden_shape"][0], 1)
        self.assertIsNone(seen["selected_indices"])
        self.assertEqual(logits.shape[0], 1)

        seen.clear()
        meta = types.SimpleNamespace(prefill=False)
        sel = torch.tensor([0, 2, 5], dtype=torch.int64)
        self.mut.PanguUltraMoEForCausalLM.forward(
            clm, input_ids, positions, kv_caches=None, attn_metadata=meta, selected_indices=sel, intermediate_tensors=None
        )
        self.assertEqual(seen["hidden_shape"][0], 6)
        self.assertIs(seen["selected_indices"], sel)

    def test_should_use_eager_mode_metadata_none_prefill_and_dict(self):
        clm = self.mut.PanguUltraMoEForCausalLM.__new__(self.mut.PanguUltraMoEForCausalLM)
        nn.Module.__init__(clm)

        dummy_layer = types.SimpleNamespace(layer_name="model.layers.0.self_attn.attn")
        clm.model = types.SimpleNamespace(layers=[dummy_layer], start_layer=0)

        self.assertTrue(self.mut.PanguUltraMoEForCausalLM.should_use_eager_mode(clm, attn_metadata=None))
        self.assertTrue(self.mut.PanguUltraMoEForCausalLM.should_use_eager_mode(clm, attn_metadata=types.SimpleNamespace(prefill=True)))
        self.assertFalse(self.mut.PanguUltraMoEForCausalLM.should_use_eager_mode(clm, attn_metadata=types.SimpleNamespace(prefill=False)))

        meta = types.SimpleNamespace(prefill=True)
        meta_dict = {dummy_layer.layer_name: meta}
        self.assertTrue(self.mut.PanguUltraMoEForCausalLM.should_use_eager_mode(clm, attn_metadata=meta_dict))

    def test_load_weights_skips_rotary_and_mtp_and_calls_weight_loader_mapping(self):
        clm = self.mut.PanguUltraMoEForCausalLM.__new__(self.mut.PanguUltraMoEForCausalLM)
        nn.Module.__init__(clm)

        clm.config = types.SimpleNamespace(
            architectures=["PanguUltraMoEForCausalLM"],
            num_mtp_layers=1,
            num_hidden_layers=2,
            num_routed_experts=None,
        )

        log = CallLog()
        p_name = "model.layers.0.mlp.gate_up_proj.weight"
        dummy_param = DummyParam(p_name, log)

        def named_parameters():
            yield p_name, dummy_param

        clm.named_parameters = named_parameters

        w = torch.randn(4, 4)
        weights = [
            ("model.layers.0.self_attn.rotary_emb.inv_freq", w),
            ("model.layers.2.mlp.gate_proj.weight", w),
            ("model.layers.0.mlp.gate_proj.weight", w),
            ("model.layers.0.mlp.up_proj.weight", w),
        ]

        loaded = self.mut.PanguUltraMoEForCausalLM.load_weights(clm, weights)

        self.assertEqual(log.count("weight_loader"), 2)
        shard_ids = {args[1] for _, args, _ in log.calls}
        self.assertEqual(shard_ids, {0, 1})
        self.assertTrue(any("gate_up_proj" in n for n in loaded))


if __name__ == "__main__":
    unittest.main()
