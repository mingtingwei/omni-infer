import os
import sys
import types
import contextlib
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

_TARGET_MODULE = "omni.models.pangu.pangu_ultra_moe_mtp"


class _ConfigProxy:
    """Proxy for torch.npu.config to avoid delattr teardown crash.

    Some torch_npu builds expose torch.npu.config as a C-extension object that does not
    support dynamic attribute add/del (e.g. allow_internal_format). Patching that
    attribute with create=True may crash during teardown (delattr). This proxy provides
    allow_internal_format while delegating all other attributes to the real config,
    and we patch torch.npu.config as a whole (setattr restore), avoiding delattr.
    """

    def __init__(self, real_cfg, allow_internal_format: bool = False):
        self._real_cfg = real_cfg
        self.allow_internal_format = allow_internal_format

    def __getattr__(self, name):
        return getattr(self._real_cfg, name)


class _FakeGroup:
    def __init__(self, world_size=1, rank_in_group=0):
        self.world_size = world_size
        self.rank_in_group = rank_in_group

    def all_gather(self, x, dim=0):
        return x

    def all_to_all(self, x):
        return x


def _make_dummy_hf_config(
    *,
    hidden_size=8,
    vocab_size=32,
    num_hidden_layers=10,
    num_mtp_layers=2,
    rms_norm_eps=1e-6,
    num_routed_experts=2,
):
    # 这里用 SimpleNamespace，避免依赖 transformers 的具体 config 类
    return types.SimpleNamespace(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_hidden_layers=num_hidden_layers,
        num_mtp_layers=num_mtp_layers,
        rms_norm_eps=rms_norm_eps,
        num_routed_experts=num_routed_experts,
    )


class TestModelsPanguUltraMOEMTP(unittest.TestCase):
    """
    目标：让该 UT 在任何环境下“自给自足”，并且在测试结束后不遗留全局状态改动，
    避免污染其它 UT（尤其是 sys.modules / torch.* 全局属性 / 全局 RNG）。
    """

    def setUp(self):
        super().setUp()

        self._stack = contextlib.ExitStack()

        # 1) RNG 隔离：该用例大量 torch.randn，若不隔离会推进全局 RNG，影响依赖 RNG 的其它 UT
        self._stack.enter_context(torch.random.fork_rng(devices=[]))
        self._gen = torch.Generator().manual_seed(1234)

        # 2) sys.modules 隔离：仅在 torch_npu 不可用时，临时注入 stub（测试结束自动恢复）
        try:
            import torch_npu  # noqa: F401
        except Exception:
            torch_npu_stub = types.ModuleType("torch_npu")
            torch_npu_stub.npu_prefetch = lambda *args, **kwargs: None  # type: ignore[attr-defined]
            self._stack.enter_context(
                patch.dict(sys.modules, {"torch_npu": torch_npu_stub}, clear=False)
            )

        # 3) torch.npu 隔离：仅补齐缺失属性，不覆写已有真实实现；并确保补齐行为可回滚
        if not hasattr(torch, "npu"):
            fake_npu = types.SimpleNamespace(
                config=types.SimpleNamespace(allow_internal_format=False)
            )
            self._stack.enter_context(patch.object(torch, "npu", fake_npu, create=True))
        else:
            if not hasattr(torch.npu, "config"):  # type: ignore[attr-defined]
                fake_cfg = types.SimpleNamespace(allow_internal_format=False)
                self._stack.enter_context(
                    patch.object(torch.npu, "config", fake_cfg, create=True)  # type: ignore[attr-defined]
                )
            elif not hasattr(torch.npu.config, "allow_internal_format"):  # type: ignore[attr-defined]
                # NOTE: Do NOT patch `torch.npu.config.allow_internal_format` with create=True.
                # Some torch_npu builds use a C-extension config object that cannot be delattr-ed,
                # which would crash in teardown and pollute other UTs.
                real_cfg = torch.npu.config  # type: ignore[attr-defined]
                proxy_cfg = _ConfigProxy(real_cfg, allow_internal_format=False)
                self._stack.enter_context(
                    patch.object(torch.npu, "config", proxy_cfg)  # type: ignore[attr-defined]
                )

        # 4) 目标模块导入隔离：避免“因为本 UT 先导入+缓存模块”而改变其它 UT 的导入/skip 行为
        self._had_target = _TARGET_MODULE in sys.modules
        self._orig_target = sys.modules.get(_TARGET_MODULE)

        def _restore_target_module():
            if self._had_target:
                sys.modules[_TARGET_MODULE] = self._orig_target  # type: ignore[assignment]
            else:
                sys.modules.pop(_TARGET_MODULE, None)

        self._stack.callback(_restore_target_module)

        # 若之前未导入过，强制在本 UT 内导入；结束后回滚 sys.modules
        if not self._had_target:
            sys.modules.pop(_TARGET_MODULE, None)

        import importlib

        self.m = importlib.import_module(_TARGET_MODULE)

    def tearDown(self):
        try:
            self._stack.close()
        finally:
            super().tearDown()


    def test_get_spec_layer_idx_from_weight_name_matches_only_mtp_layers(self):
        m = self.m
        cfg = _make_dummy_hf_config(num_hidden_layers=10, num_mtp_layers=3)

        # MTP 起始层 = num_hidden_layers
        self.assertEqual(
            m.get_spec_layer_idx_from_weight_name(cfg, "model.layers.10.mlp.gate_proj.weight"),
            10,
        )
        self.assertEqual(
            m.get_spec_layer_idx_from_weight_name(cfg, "model.layers.11.self_attn.q_proj.weight"),
            11,
        )
        self.assertEqual(
            m.get_spec_layer_idx_from_weight_name(cfg, "model.layers.12.input_layernorm.weight"),
            12,
        )

        # 非 MTP 层
        self.assertIsNone(
            m.get_spec_layer_idx_from_weight_name(cfg, "model.layers.9.mlp.gate_proj.weight")
        )
        self.assertIsNone(
            m.get_spec_layer_idx_from_weight_name(
                cfg, "some.other.prefix.model.layers.10.mlp.gate_proj.weight"
            )
        )

        # config 没有 num_mtp_layers 或 num_mtp_layers <= 0
        cfg2 = types.SimpleNamespace(num_hidden_layers=10)  # no num_mtp_layers
        self.assertIsNone(
            m.get_spec_layer_idx_from_weight_name(cfg2, "model.layers.10.mlp.gate_proj.weight")
        )
        cfg3 = _make_dummy_hf_config(num_hidden_layers=10, num_mtp_layers=0)
        self.assertIsNone(
            m.get_spec_layer_idx_from_weight_name(cfg3, "model.layers.10.mlp.gate_proj.weight")
        )

    def test_predictor_layer_should_use_eager_mode_branching(self):
        m = self.m
        layer = m.PanguUltraMoEMultiTokenPredictorLayer.__new__(
            m.PanguUltraMoEMultiTokenPredictorLayer
        )
        layer.layer_name = "layer_0"

        # kwargs 为空 => True
        self.assertTrue(layer.should_use_eager_mode())

        # attn_metadata 缺失/为 None/为 falsy => True
        self.assertTrue(layer.should_use_eager_mode(attn_metadata=None))
        self.assertTrue(layer.should_use_eager_mode(attn_metadata={}))  # falsy dict

        # attn_metadata 为对象：prefill True => True; prefill False => False
        self.assertTrue(
            layer.should_use_eager_mode(attn_metadata=types.SimpleNamespace(prefill=True))
        )
        self.assertFalse(
            layer.should_use_eager_mode(attn_metadata=types.SimpleNamespace(prefill=False))
        )

        # attn_metadata 为 dict：按 layer_name 索引
        attn_dict = {
            "layer_0": types.SimpleNamespace(prefill=False),
            "layer_1": types.SimpleNamespace(prefill=True),
        }
        self.assertFalse(layer.should_use_eager_mode(attn_metadata=attn_dict))

    def test_set_share_weight_injects_embed_and_head_for_ignore_share_weight_mode(self):
        m = self.m
        predictor = m.PanguUltraMoEMultiTokenPredictor.__new__(
            m.PanguUltraMoEMultiTokenPredictor
        )
        nn.Module.__init__(predictor)
        predictor.ignore_share_weight = True

        layer0 = types.SimpleNamespace(
            embed_tokens=None, shared_head=types.SimpleNamespace(head=None)
        )
        layer1 = types.SimpleNamespace(
            embed_tokens=None, shared_head=types.SimpleNamespace(head=None)
        )
        predictor.layers = {"10": layer0, "11": layer1}

        target_embed = object()
        target_head = object()
        target_model = types.SimpleNamespace(
            model=types.SimpleNamespace(embed_tokens=target_embed),
            lm_head=target_head,
        )

        predictor.set_share_weight(target_model)

        self.assertIs(layer0.embed_tokens, target_embed)
        self.assertIs(layer1.embed_tokens, target_embed)
        self.assertIs(layer0.shared_head.head, target_head)
        self.assertIs(layer1.shared_head.head, target_head)

    def test_predictor_layer_forward_routes_by_attn_metadata_and_selected_indices(self):
        m = self.m

        layer = m.PanguUltraMoEMultiTokenPredictorLayer.__new__(
            m.PanguUltraMoEMultiTokenPredictorLayer
        )
        nn.Module.__init__(layer)

        layer.config = _make_dummy_hf_config(
            hidden_size=8, vocab_size=32, num_hidden_layers=10, num_mtp_layers=2
        )
        layer.enorm = nn.Identity()
        layer.hnorm = nn.Identity()
        layer.eh_tp_size = 1
        layer.layer_idx = 0

        gen = self._gen

        def _embed_tokens(input_ids, reduce=1):
            token_num = int(input_ids.numel())
            return torch.randn(token_num, layer.config.hidden_size, generator=gen)

        layer.embed_tokens = _embed_tokens

        token_num = 3
        eh_out = torch.randn(token_num, layer.config.hidden_size, generator=gen)
        layer.eh_proj = types.SimpleNamespace(forward=Mock(return_value=(eh_out, None)))

        head_call = Mock(
            side_effect=lambda hs, bias=None: torch.empty(hs.shape[0], layer.config.vocab_size)
        )
        layer.shared_head = types.SimpleNamespace(
            norm=Mock(side_effect=lambda encoded, residual: (encoded, None)),
            head=head_call,
        )

        input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        positions = torch.arange(token_num, dtype=torch.long)
        kv_caches = [torch.empty(1)]
        prev_h = torch.randn(token_num, layer.config.hidden_size, generator=gen)

        with patch.object(m, "get_tensor_model_parallel_world_size", return_value=1), \
             patch.object(m, "get_tensor_model_parallel_rank", return_value=0), \
             patch.object(m, "tensor_model_parallel_all_gather", side_effect=lambda x, dim=0: x), \
             patch.object(m, "get_dp_group", return_value=_FakeGroup(world_size=1)), \
             patch.object(m.PanguUltraMoEDecoderLayer, "forward", return_value=(eh_out, None)):

            head_call.reset_mock()
            logits, hidden = layer.forward(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=None,
                previous_hidden_states=prev_h,
                selected_indices=None,
            )
            self.assertTrue(isinstance(logits, torch.Tensor))
            self.assertTrue(isinstance(hidden, torch.Tensor))
            self.assertEqual(head_call.call_count, 1)
            hs_arg = head_call.call_args[0][0]
            self.assertEqual(hs_arg.shape[0], 1)

            head_call.reset_mock()
            selected = torch.tensor([0, 2], dtype=torch.long)
            logits2, hidden2 = layer.forward(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=types.SimpleNamespace(prefill=False),
                previous_hidden_states=prev_h,
                selected_indices=selected,
            )
            self.assertTrue(isinstance(logits2, torch.Tensor))
            self.assertTrue(isinstance(hidden2, torch.Tensor))
            self.assertEqual(head_call.call_count, 1)
            hs_arg2 = head_call.call_args[0][0]
            self.assertEqual(hs_arg2.shape[0], 2)

    def test_mtp_forward_caps_mtp_layer_idx_to_last_predictor(self):
        m = self.m
        mtp = m.PanguUltraMoEMTP.__new__(m.PanguUltraMoEMTP)
        nn.Module.__init__(mtp)
        mtp.n_predictor = 2
        mtp.model = Mock(return_value="ok")

        gen = self._gen
        input_ids = torch.tensor([1, 2], dtype=torch.long)
        positions = torch.tensor([0, 1], dtype=torch.long)
        kv_caches = [torch.empty(1)]
        attn_metadata = types.SimpleNamespace(prefill=False)
        prev_h = torch.randn(2, 8, generator=gen)

        out = mtp.forward(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            previous_hidden_states=prev_h,
            selected_indices=None,
            mtp_layer_idx=999,
        )
        self.assertEqual(out, "ok")
        called_kwargs = mtp.model.call_args.kwargs
        self.assertEqual(called_kwargs["mtp_layer_idx"], 1)

    def test_load_weights_filters_out_irrelevant_weights_and_non_spec_layers(self):
        m = self.m
        mtp = m.PanguUltraMoEMTP.__new__(m.PanguUltraMoEMTP)
        nn.Module.__init__(mtp)
        mtp.config = _make_dummy_hf_config(
            num_hidden_layers=10, num_mtp_layers=1, num_routed_experts=2
        )
        mtp.model = types.SimpleNamespace(ignore_share_weight=True)
        mtp.named_parameters = Mock(return_value=[])

        gen = self._gen
        weights = [
            ("model.layers.10.rotary_emb.inv_freq", torch.randn(1, generator=gen)),
            ("model.layers.10.embed_tokens.weight", torch.randn(1, generator=gen)),
            ("model.layers.10.shared_head.head.weight", torch.randn(1, generator=gen)),
            ("model.layers.9.mlp.gate_proj.weight", torch.randn(1, generator=gen)),
        ]

        with patch.object(m, "is_pp_missing_parameter", return_value=False), \
             patch.object(m.FusedMoE, "make_expert_params_mapping", return_value=[]):
            loaded = mtp.load_weights(weights)

        self.assertEqual(loaded, set())

    def test_load_weights_routes_to_stacked_expert_or_default_loader_paths(self):
        m = self.m
        mtp = m.PanguUltraMoEMTP.__new__(m.PanguUltraMoEMTP)
        nn.Module.__init__(mtp)
        mtp.config = _make_dummy_hf_config(
            num_hidden_layers=10, num_mtp_layers=1, num_routed_experts=2
        )
        mtp.model = types.SimpleNamespace(ignore_share_weight=False)

        stacked_param = Mock()
        stacked_param.weight_loader = Mock()

        expert_param = Mock()
        expert_param.weight_loader = Mock()

        class _NoLoaderParam:
            pass

        default_param = _NoLoaderParam()

        params = [
            ("model.layers.10.mlp.gate_up_proj.weight", stacked_param),
            ("model.layers.10.mlp.experts.0.w13.weight", expert_param),
            ("model.layers.10.mlp.down_proj.weight", default_param),
        ]
        mtp.named_parameters = Mock(return_value=params)

        gen = self._gen
        weights = [
            ("model.layers.10.mlp.gate_proj.weight", torch.randn(1, generator=gen)),
            ("model.layers.10.mlp.up_proj.weight", torch.randn(1, generator=gen)),
            ("model.layers.10.mlp.experts.0.gate_proj.weight", torch.randn(1, generator=gen)),
            ("model.layers.10.mlp.down_proj.weight", torch.randn(1, generator=gen)),
        ]

        expert_mapping = [("w13", "gate_proj", 0, 0)]
        default_loader_mock = Mock()

        with patch.object(m, "is_pp_missing_parameter", return_value=False), \
             patch.object(m.FusedMoE, "make_expert_params_mapping", return_value=expert_mapping), \
             patch.object(m, "default_weight_loader", default_loader_mock):
            loaded = mtp.load_weights(weights)

        self.assertGreaterEqual(stacked_param.weight_loader.call_count, 2)
        call0 = stacked_param.weight_loader.call_args_list[0]
        call1 = stacked_param.weight_loader.call_args_list[1]
        self.assertEqual(call0.args[2], 0)
        self.assertEqual(call1.args[2], 1)

        self.assertEqual(expert_param.weight_loader.call_count, 1)
        _, _, name_arg = expert_param.weight_loader.call_args.args[:3]
        self.assertIn("model.layers.10.mlp.experts.0.w13.weight", name_arg)
        self.assertEqual(expert_param.weight_loader.call_args.kwargs["expert_id"], 0)
        self.assertEqual(expert_param.weight_loader.call_args.kwargs["shard_id"], 0)

        self.assertEqual(default_loader_mock.call_count, 1)
        self.assertIs(default_loader_mock.call_args.args[0], default_param)

        self.assertIn("model.layers.10.mlp.gate_up_proj.weight", loaded)
        self.assertIn("model.layers.10.mlp.experts.0.w13.weight", loaded)
        self.assertIn("model.layers.10.mlp.down_proj.weight", loaded)

    def test_compute_logits_uses_shared_head_head_attribute(self):
        m = self.m
        layer = m.PanguUltraMoEMultiTokenPredictorLayer.__new__(
            m.PanguUltraMoEMultiTokenPredictorLayer
        )
        nn.Module.__init__(layer)

        dummy_head = object()
        layer.shared_head = types.SimpleNamespace(head=dummy_head)
        layer.logits_processor = Mock(return_value="logits")

        hidden_states = torch.randn(2, 8, generator=self._gen)
        sampling_metadata = Mock()

        out = layer.compute_logits(hidden_states, sampling_metadata)

        self.assertEqual(out, "logits")
        first_arg = layer.logits_processor.call_args.args[0]
        self.assertIs(first_arg, dummy_head)


if __name__ == "__main__":
    unittest.main()
