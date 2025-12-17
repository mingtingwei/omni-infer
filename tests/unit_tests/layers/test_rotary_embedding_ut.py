import pytest
import unittest
from unittest import mock
from unittest.mock import Mock, patch, MagicMock
import importlib

import torch
from torch import nn
from torch.nn import Parameter

from omni.layers.rotary_embedding import *
import omni.layers.rotary_embedding as rope_mod
import omni.layers.linear as omni_linear_mod


class TestRotaryEmbedding(unittest.TestCase):
    def setUp(self):
        super().setUp()
        if hasattr(rope_mod, "_ROPE_DICT"):
            rope_mod._ROPE_DICT.clear()

    def tearDown(self):
        if hasattr(rope_mod, "_ROPE_DICT"):
            rope_mod._ROPE_DICT.clear()
        super().tearDown()

# ==================== get_rope  ====================

    def test_default_no_scaling_returns_npu_impl_and_cached(self):
        sentinel = object()
        ctor = MagicMock(return_value=sentinel)

        with patch.object(rope_mod, "RotaryEmbeddingTorchNpu", ctor):
            rope_mod._ROPE_DICT.clear()
            obj1 = rope_mod.get_rope(
                head_size=128,
                rotary_dim=64,
                max_position=2048,
                base=10000,
                is_neox_style=True,
                rope_scaling=None,
                dtype=torch.float16,
            )
            obj2 = rope_mod.get_rope(
                head_size=128,
                rotary_dim=64,
                max_position=2048,
                base=10000,
                is_neox_style=True,
                rope_scaling=None,
                dtype=torch.float16,
            )

        self.assertIs(obj1, sentinel)
        self.assertIs(obj2, sentinel)
        self.assertEqual(ctor.call_count, 1)
        self.assertEqual(len(rope_mod._ROPE_DICT), 1)

    def test_rope_scaling_cache_key_normalizes_list_to_tuple_for_cache_hit(self):
        sentinel = object()
        ctor = MagicMock(return_value=sentinel)

        with patch.object(rope_mod, "LinearScalingRotaryEmbedding", ctor):
            rope_mod._ROPE_DICT.clear()

            rope_scaling_1 = {"rope_type": "linear", "factor": 2.0, "mrope_section": [2, 3]}
            rope_scaling_2 = {"rope_type": "linear", "factor": 2.0, "mrope_section": [2, 3]}  # 新 dict

            obj1 = rope_mod.get_rope(
                head_size=128,
                rotary_dim=64,
                max_position=2048,
                base=10000,
                is_neox_style=True,
                rope_scaling=rope_scaling_1,
                dtype=torch.float16,
            )
            obj2 = rope_mod.get_rope(
                head_size=128,
                rotary_dim=64,
                max_position=2048,
                base=10000,
                is_neox_style=True,
                rope_scaling=rope_scaling_2,
                dtype=torch.float16,
            )

        self.assertIs(obj1, sentinel)
        self.assertIs(obj2, sentinel)
        self.assertEqual(ctor.call_count, 1)
        self.assertEqual(len(rope_mod._ROPE_DICT), 1)

    def test_partial_rotary_factor_reduces_effective_rotary_dim(self):
        sentinel = object()
        ctor = MagicMock(return_value=sentinel)

        with patch.object(rope_mod, "RotaryEmbeddingTorchNpu", ctor):
            rope_mod._ROPE_DICT.clear()
            _ = rope_mod.get_rope(
                head_size=128,
                rotary_dim=64,
                max_position=2048,
                base=10000,
                is_neox_style=True,
                rope_scaling=None,
                dtype=torch.float16,
                partial_rotary_factor=0.5,
            )

        # ctor(head_size, rotary_dim, max_position, base, is_neox_style, dtype, ...)
        called_rotary_dim = ctor.call_args[0][1]
        self.assertLessEqual(called_rotary_dim, int(64 * 0.5))  # “至少缩小到不超过一次缩放后的值”
        self.assertGreater(called_rotary_dim, 0)

    def test_dispatch_scaling_types_and_unknown_raises(self):
        # 为每个构造器准备返回对象（用不同哨兵，便于断言走到哪个分支）
        sent = {name: object() for name in [
            "LinearScalingRotaryEmbedding",
            "YaRNScalingRotaryEmbedding",
            "DeepseekScalingRotaryEmbedding",
            "DynamicNTKScalingRotaryEmbedding",
            "ExtendedRotaryEmbedding",
            "QwenRotaryEmbedding",
            "QwenMRotaryEmbedding",
            "RotaryEmbeddingTorchNpu",
            "GPUMRotaryEmbedding",
            "PanguProMoERotaryEmbedding",
        ]}

        ctors = {k: MagicMock(return_value=v) for k, v in sent.items()}

        class _FakePPGroup:
            world_size = 2

        patches = [
            patch.object(rope_mod, "LinearScalingRotaryEmbedding", ctors["LinearScalingRotaryEmbedding"]),
            patch.object(rope_mod, "YaRNScalingRotaryEmbedding", ctors["YaRNScalingRotaryEmbedding"]),
            patch.object(rope_mod, "DeepseekScalingRotaryEmbedding", ctors["DeepseekScalingRotaryEmbedding"]),
            patch.object(rope_mod, "DynamicNTKScalingRotaryEmbedding", ctors["DynamicNTKScalingRotaryEmbedding"]),
            patch.object(rope_mod, "ExtendedRotaryEmbedding", ctors["ExtendedRotaryEmbedding"]),
            patch.object(rope_mod, "QwenRotaryEmbedding", ctors["QwenRotaryEmbedding"]),
            patch.object(rope_mod, "QwenMRotaryEmbedding", ctors["QwenMRotaryEmbedding"]),
            patch.object(rope_mod, "RotaryEmbeddingTorchNpu", ctors["RotaryEmbeddingTorchNpu"]),
            patch.object(rope_mod, "GPUMRotaryEmbedding", ctors["GPUMRotaryEmbedding"], create=True),
            patch.object(rope_mod, "PanguProMoERotaryEmbedding", ctors["PanguProMoERotaryEmbedding"]),
            patch.object(rope_mod, "get_pp_group", lambda: _FakePPGroup(), create=True),
        ]

        for p in patches:
            p.start()
        try:
            cases = [
                ({"rope_type": "linear", "factor": 2.0}, "LinearScalingRotaryEmbedding"),
                ({"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 2048}, "YaRNScalingRotaryEmbedding"),
                ({"rope_type": "deepseek_yarn", "factor": 2.0, "original_max_position_embeddings": 2048}, "DeepseekScalingRotaryEmbedding"),
                ({"rope_type": "dynamic", "factor": 2.0}, "DynamicNTKScalingRotaryEmbedding"),
                ({"rope_type": "llama3"}, "ExtendedRotaryEmbedding"),

                ({"rope_type": "qwen"}, "QwenRotaryEmbedding"),
                ({"rope_type": "qwen", "mrope_section": [2, 3]}, "QwenMRotaryEmbedding"),

                ({"rope_type": "gemma_default"}, "RotaryEmbeddingTorchNpu"),
                ({"rope_type": "gemma_default", "mrope_section": [2, 3]}, "GPUMRotaryEmbedding"),

                ({"rope_type": "pangu_pro_moe"}, "PanguProMoERotaryEmbedding"),
            ]

            for rope_scaling, expect in cases:
                rope_mod._ROPE_DICT.clear()
                obj = rope_mod.get_rope(
                    head_size=128,
                    rotary_dim=64,
                    max_position=2048,
                    base=10000,
                    is_neox_style=True,
                    rope_scaling=rope_scaling,
                    dtype=torch.float16,
                )
                self.assertIs(obj, sent[expect], msg=f"rope_scaling={rope_scaling} should dispatch to {expect}")

            rope_mod._ROPE_DICT.clear()
            with self.assertRaises(ValueError):
                rope_mod.get_rope(
                    head_size=128,
                    rotary_dim=64,
                    max_position=2048,
                    base=10000,
                    is_neox_style=True,
                    rope_scaling={"rope_type": "unknown_xxx", "type": "unknown_xxx"},
                    dtype=torch.float16,
                )
        finally:
            for p in reversed(patches):
                p.stop()

# ==================== RotaryEmbeddingTorchNpu  ====================

    @staticmethod
    def _fake_compute_cos_sin_cache(layer_self):
        """Avoid .npu() and any device-specific ops. Return CPU cos/sin with expected indexability.
        """
        max_len = int(layer_self.max_position_embeddings)
        head_size = int(layer_self.head_size)

        # 统一造 2D [max_len, head_size]，满足 get_cos_sin / fused path 的 index_select 需求
        cos = torch.arange(max_len * head_size, dtype=torch.float32).reshape(max_len, head_size)
        sin = torch.arange(max_len * head_size, dtype=torch.float32).reshape(max_len, head_size) + 0.5
        return cos, sin

    @staticmethod
    def _fake_compute_cos_sin_cache_alt(layer_self):
        """Return CPU cos_sin_cache with even last dim (so chunk(2, -1) would work if ever used)."""
        max_pos = int(layer_self.max_position_embeddings)
        dim = int(layer_self.rotary_dim)
        if dim % 2 != 0:
            dim += 1
        cache = torch.zeros((max_pos, dim), dtype=torch.float32)
        return cache

    def _make_layer(self, *, head_size, rotary_dim, max_pos=16, is_neox_style=False, dtype=torch.float16):
        with patch.object(
            rope_mod.RotaryEmbeddingTorchNpu,
            "_compute_cos_sin_cache",
            new=TestRotaryEmbedding._fake_compute_cos_sin_cache,
        ), patch.object(
            rope_mod.RotaryEmbeddingTorchNpu,
            "_compute_cos_sin_cache_alt",
            new=TestRotaryEmbedding._fake_compute_cos_sin_cache_alt,
        ):
            return rope_mod.RotaryEmbeddingTorchNpu(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_pos,
                base=10000,
                is_neox_style=is_neox_style,
                dtype=dtype,
            )

    # 1) init：buffer/属性契约（不跑真实 npu cache）
    def test_rotary_embedding_torch_npu_init_registers_cos_sin_cache_and_basic_attrs(self):
        layer = self._make_layer(head_size=16, rotary_dim=8, max_pos=32, is_neox_style=False, dtype=torch.float16)

        self.assertEqual(layer.head_size, 16)
        self.assertEqual(layer.rotary_dim, 8)
        self.assertEqual(layer.max_position_embeddings, 32)
        self.assertEqual(layer.base, 10000)
        self.assertFalse(layer.is_neox_style)

        # cos/sin exist and indexable
        self.assertTrue(hasattr(layer, "cos"))
        self.assertTrue(hasattr(layer, "sin"))
        self.assertEqual(layer.cos.shape[0], 32)
        self.assertEqual(layer.sin.shape[0], 32)

        # cos_sin_cache registered as buffer (persistent=False means not in state_dict, but must be in named_buffers)
        buffers = dict(layer.named_buffers())
        self.assertIn("cos_sin_cache", buffers)
        self.assertEqual(buffers["cos_sin_cache"].dtype, torch.float16)

    # 2) get_cos_sin：offsets + shape 契约
    def test_get_cos_sin_applies_offsets_and_returns_b11d_shape(self):
        layer = self._make_layer(head_size=8, rotary_dim=4, max_pos=16, dtype=torch.float16)

        positions = torch.tensor([0, 2, 3], dtype=torch.long)
        offsets = torch.tensor([1, 0, 2], dtype=torch.long)

        cos, sin = layer.get_cos_sin(positions, offsets=offsets)

        # shape: (-1, 1, 1, D)
        self.assertEqual(cos.shape, (3, 1, 1, layer.cos.shape[-1]))
        self.assertEqual(sin.shape, (3, 1, 1, layer.sin.shape[-1]))

        # sanity: the indexing is positions + offsets
        idx = positions + offsets
        expected_cos = layer.cos[idx].view(3, 1, 1, -1)
        expected_sin = layer.sin[idx].view(3, 1, 1, -1)
        self.assertTrue(torch.equal(cos, expected_cos))
        self.assertTrue(torch.equal(sin, expected_sin))

        # offsets=None should also work
        cos2, sin2 = layer.get_cos_sin(positions, offsets=None)
        self.assertEqual(cos2.shape, (3, 1, 1, layer.cos.shape[-1]))
        self.assertEqual(sin2.shape, (3, 1, 1, layer.sin.shape[-1]))

    # 3) forward：三分支调度（核心门禁）
    def test_forward_dispatches_to_native_smallops_or_fused_by_rotary_dim(self):
        T = 4
        position_ids = torch.arange(T, dtype=torch.long)

        # (a) rotary_dim < head_size -> _forward_native
        layer_a = self._make_layer(head_size=16, rotary_dim=8, max_pos=16, dtype=torch.float16)
        q_a = torch.zeros((T, 16), dtype=torch.float16)
        k_a = torch.zeros((T, 16), dtype=torch.float16)
        out_q_a = torch.ones_like(q_a)
        out_k_a = torch.ones_like(k_a)

        layer_a._forward_native = MagicMock(return_value=(out_q_a, out_k_a))
        layer_a._forward_ascend_ops_and_small_ops = MagicMock()
        layer_a._forward_fused_ops = MagicMock()

        oq, ok = layer_a.forward(position_ids, q_a, k_a)
        self.assertTrue(torch.equal(oq, out_q_a))
        self.assertTrue(torch.equal(ok, out_k_a))
        layer_a._forward_native.assert_called_once()
        layer_a._forward_ascend_ops_and_small_ops.assert_not_called()
        layer_a._forward_fused_ops.assert_not_called()

        # (b) rotary_dim >= head_size and rotary_dim != 128 -> _forward_ascend_ops_and_small_ops
        layer_b = self._make_layer(head_size=16, rotary_dim=16, max_pos=16, dtype=torch.float16)
        q_b = torch.zeros((T, 16), dtype=torch.float16)
        k_b = torch.zeros((T, 16), dtype=torch.float16)
        out_q_b = torch.full_like(q_b, 2)
        out_k_b = torch.full_like(k_b, 3)

        layer_b._forward_native = MagicMock()
        layer_b._forward_ascend_ops_and_small_ops = MagicMock(return_value=(out_q_b, out_k_b))
        layer_b._forward_fused_ops = MagicMock()

        oq, ok = layer_b.forward(position_ids, q_b, k_b)
        self.assertTrue(torch.equal(oq, out_q_b))
        self.assertTrue(torch.equal(ok, out_k_b))
        layer_b._forward_native.assert_not_called()
        layer_b._forward_ascend_ops_and_small_ops.assert_called_once()
        layer_b._forward_fused_ops.assert_not_called()

        # (c) rotary_dim == 128 -> _forward_fused_ops
        layer_c = self._make_layer(head_size=128, rotary_dim=128, max_pos=16, dtype=torch.float16)
        q_c = torch.zeros((T, 128), dtype=torch.float16)
        k_c = torch.zeros((T, 128), dtype=torch.float16)
        out_q_c = torch.full_like(q_c, 7)
        out_k_c = torch.full_like(k_c, 8)

        layer_c._forward_native = MagicMock()
        layer_c._forward_ascend_ops_and_small_ops = MagicMock()
        layer_c._forward_fused_ops = MagicMock(return_value=(out_q_c, out_k_c))

        oq, ok = layer_c.forward(position_ids, q_c, k_c, cos=None, sin=None)
        self.assertTrue(torch.equal(oq, out_q_c))
        self.assertTrue(torch.equal(ok, out_k_c))
        layer_c._forward_native.assert_not_called()
        layer_c._forward_ascend_ops_and_small_ops.assert_not_called()
        layer_c._forward_fused_ops.assert_called_once()

    # 4) fused：调用 torch_npu 的契约 + flatten 逻辑
    def test_forward_fused_ops_calls_torch_npu_apply_rotary_pos_emb_and_flattens_to_2d(self):
        # If the runtime image doesn't include torch_npu, skip (gate should be stable across CPU-only CI).
        if not hasattr(rope_mod, "torch_npu"):
            self.skipTest("torch_npu not available in this environment")

        layer = self._make_layer(head_size=128, rotary_dim=128, max_pos=16, dtype=torch.float16)

        T = 5
        position_ids = torch.arange(T, dtype=torch.long)
        query = torch.randn((T, 128), dtype=torch.float16)
        key = torch.randn((T, 128), dtype=torch.float16)

        calls = {"count": 0}

        def _fake_apply(q, k, cos, sin, fmt):
            calls["count"] += 1
            # contract checks (shape only)
            self.assertEqual(fmt, "TND")
            self.assertEqual(q.dim(), 3)   # [T, N, D]
            self.assertEqual(k.dim(), 3)
            self.assertEqual(cos.dim(), 3)  # [T, 1, D]
            self.assertEqual(sin.dim(), 3)
            self.assertEqual(q.shape[0], T)
            self.assertEqual(q.shape[-1], 128)
            return q, k

        with patch.object(rope_mod.torch_npu, "npu_apply_rotary_pos_emb", side_effect=_fake_apply):
            # (a) cos/sin None path: module selects from self.cos/self.sin
            q_out, k_out = layer._forward_fused_ops(position_ids, query, key, cos=None, sin=None)
            self.assertEqual(q_out.shape, (T, 128))
            self.assertEqual(k_out.shape, (T, 128))

            # (b) cos/sin provided path: squeeze(2) branch
            cos_idx = torch.index_select(layer.cos, dim=0, index=position_ids).unsqueeze(1).unsqueeze(2)
            sin_idx = torch.index_select(layer.sin, dim=0, index=position_ids).unsqueeze(1).unsqueeze(2)
            q_out2, k_out2 = layer._forward_fused_ops(position_ids, query, key, cos=cos_idx, sin=sin_idx)
            self.assertEqual(q_out2.shape, (T, 128))
            self.assertEqual(k_out2.shape, (T, 128))

        self.assertEqual(calls["count"], 2)

# ==================== YaRNScalingRotaryEmbedding  ====================

    @staticmethod
    def _fake_cos_sin_cache_alt_for_yarn(layer_self):
        max_pos = int(layer_self.max_position_embeddings)
        last_dim = max(2, int(layer_self.rotary_dim))  # 确保可 chunk(2, -1)
        if last_dim % 2 == 1:
            last_dim += 1
        return torch.zeros((max_pos, last_dim), dtype=torch.float32)

    @staticmethod
    def _fake_inv_freq(self, _any_factor):
        n = max(1, int(self.rotary_dim) // 2)
        return torch.ones((n,), dtype=torch.float32)

    def _make_yarn_layer(
        self,
        *,
        head_size=16,
        rotary_dim=8,
        max_pos=8,
        base=10000,
        is_neox_style=True,
        scaling_factor=2,
        dtype=torch.float16,
        extrapolation_factor=1.0,
        attn_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        yarn_get_mscale_ret=3.0,
    ):
        yarn_get_mscale_mock = MagicMock(return_value=yarn_get_mscale_ret)

        with patch.object(torch.Tensor, "npu", lambda t: t, create=True), \
             patch.object(rope_mod.RotaryEmbeddingTorchNpu, "_compute_cos_sin_cache_alt", type(self)._fake_cos_sin_cache_alt_for_yarn), \
             patch.object(rope_mod, "_yarn_get_mscale", yarn_get_mscale_mock), \
             patch.object(rope_mod.YaRNScalingRotaryEmbedding, "_compute_inv_freq", autospec=True, side_effect=type(self)._fake_inv_freq) as inv_freq_mock:
            layer = rope_mod.YaRNScalingRotaryEmbedding(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_pos,
                base=base,
                is_neox_style=is_neox_style,
                scaling_factor=scaling_factor,
                dtype=dtype,
                extrapolation_factor=extrapolation_factor,
                attn_factor=attn_factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            )
        return layer, inv_freq_mock, yarn_get_mscale_mock

    def test_yarn_init_sets_scaling_fields_and_computes_mscale_from_yarn_get_mscale_and_attn_factor(self):
        layer, inv_freq_mock, yarn_get_mscale_mock = self._make_yarn_layer(
            scaling_factor=2,
            attn_factor=2.5,
            yarn_get_mscale_ret=3.0,
        )

        self.assertEqual(layer.scaling_factor, 2)
        self.assertEqual(layer.attn_factor, 2.5)
        self.assertEqual(layer.beta_fast, 32)
        self.assertEqual(layer.beta_slow, 1)
        self.assertEqual(layer.extrapolation_factor, 1.0)

        yarn_get_mscale_mock.assert_called_once_with(2)
        self.assertAlmostEqual(layer.mscale, float(3.0 * 2.5), places=6)

    def test_yarn_compute_cos_sin_cache_calls_compute_inv_freq_with_scaling_factor_and_updates_max_len(self):
        layer, inv_freq_mock, _ = self._make_yarn_layer(max_pos=8, scaling_factor=2, is_neox_style=True)

        self.assertGreaterEqual(inv_freq_mock.call_count, 1)

        last_call = inv_freq_mock.call_args_list[-1]
        self.assertEqual(last_call[0][1], layer.scaling_factor)

        expected_len = int(layer.max_position_embeddings * layer.scaling_factor)
        self.assertEqual(int(layer.max_len), expected_len)
        self.assertEqual(layer.cos.shape, (expected_len, layer.rotary_dim))
        self.assertEqual(layer.sin.shape, (expected_len, layer.rotary_dim))

    def test_yarn_compute_cos_sin_cache_shape_contract_covers_neox_and_non_neox_branches(self):
        for is_neox_style in (True, False):
            layer, _, _ = self._make_yarn_layer(
                head_size=16,
                rotary_dim=8,
                max_pos=8,
                scaling_factor=2,
                is_neox_style=is_neox_style,
            )
            expected_len = int(layer.max_position_embeddings * layer.scaling_factor)
            self.assertEqual(layer.cos.shape, (expected_len, layer.rotary_dim))
            self.assertEqual(layer.sin.shape, (expected_len, layer.rotary_dim))

    def test_yarn_compute_cos_sin_cache_applies_mscale_multiplier_without_checking_numerical_accuracy(self):
        def _ones_like(x):
            return torch.ones_like(x)

        with patch.object(rope_mod.torch, "cos", side_effect=_ones_like), \
             patch.object(rope_mod.torch, "sin", side_effect=_ones_like):
            layer, _, _ = self._make_yarn_layer(
                rotary_dim=8,
                max_pos=8,
                scaling_factor=2,
                attn_factor=2.0,
                yarn_get_mscale_ret=3.0,  # mscale=6.0
                is_neox_style=True,
            )

        expected = torch.tensor(layer.mscale, dtype=layer.cos.dtype, device=layer.cos.device)
        self.assertTrue(torch.allclose(layer.cos, expected))
        self.assertTrue(torch.allclose(layer.sin, expected))

# ==================== LinearScalingRotaryEmbedding  ====================

    def _make_inv_freq_cpu_with_npu_stub(self, rotary_dim: int, dtype=torch.float32):
        """Return a CPU tensor but with a .npu() stub so codepath won't create real NPU tensors."""
        inv = torch.arange(rotary_dim // 2, dtype=dtype) + 1  # values don't matter
        inv.npu = lambda: inv  # override instance attribute, keep CPU
        return inv

    def _make_linear_layer_without_real_cache(self, *, head_size, rotary_dim, max_pos, scaling_factor,
                                              is_neox_style, dtype=torch.float16):
        """
        Construct LinearScalingRotaryEmbedding but avoid running its real _compute_cos_sin_cache
        during __init__ (so init is stable and doesn't touch device-specific ops).
        We'll call the real _compute_cos_sin_cache explicitly in tests.
        """
        dummy_cos = torch.zeros((max_pos, rotary_dim), dtype=torch.float32)
        dummy_sin = torch.zeros((max_pos, rotary_dim), dtype=torch.float32)
        dummy_alt = torch.zeros((max_pos, 2), dtype=torch.float32)  # last dim even

        with patch.object(rope_mod.LinearScalingRotaryEmbedding,
                          "_compute_cos_sin_cache",
                          return_value=(dummy_cos, dummy_sin)), \
             patch.object(rope_mod.RotaryEmbeddingTorchNpu,
                          "_compute_cos_sin_cache_alt",
                          return_value=dummy_alt):
            layer = rope_mod.LinearScalingRotaryEmbedding(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_pos,
                base=10000,
                is_neox_style=is_neox_style,
                scaling_factor=scaling_factor,
                dtype=dtype,
            )
        return layer

    def test_linear_scaling_rope_init_records_scaling_factor_and_has_buffer(self):
        layer = self._make_linear_layer_without_real_cache(
            head_size=16, rotary_dim=8, max_pos=32, scaling_factor=2,
            is_neox_style=True, dtype=torch.float16
        )
        self.assertEqual(layer.scaling_factor, 2)
        self.assertEqual(layer.max_position_embeddings, 32)

        # buffer contract from base __init__
        buffers = dict(layer.named_buffers())
        self.assertIn("cos_sin_cache", buffers)
        self.assertEqual(buffers["cos_sin_cache"].dtype, torch.float16)

    def test_linear_scaling_compute_cache_updates_max_len_and_calls_inv_freq_with_base(self):
        layer = self._make_linear_layer_without_real_cache(
            head_size=16, rotary_dim=8, max_pos=8, scaling_factor=2,
            is_neox_style=True, dtype=torch.float16
        )

        inv = self._make_inv_freq_cpu_with_npu_stub(rotary_dim=layer.rotary_dim, dtype=torch.float32)
        inv_mock = MagicMock(return_value=inv)
        layer._compute_inv_freq = inv_mock  # override at instance level (most robust)

        # call real implementation (unbound)
        cos, sin = rope_mod.LinearScalingRotaryEmbedding._compute_cos_sin_cache(layer)

        inv_mock.assert_called_once()
        self.assertEqual(inv_mock.call_args[0][0], layer.base)  # called with base
        self.assertEqual(layer.max_len, layer.max_position_embeddings * layer.scaling_factor)
        self.assertEqual(cos.shape[0], layer.max_len)
        self.assertEqual(sin.shape[0], layer.max_len)
        self.assertEqual(cos.shape[-1], layer.rotary_dim)
        self.assertEqual(sin.shape[-1], layer.rotary_dim)
        self.assertEqual(cos.dtype, torch.get_default_dtype())
        self.assertEqual(sin.dtype, torch.get_default_dtype())

    def test_linear_scaling_compute_cache_neox_style_uses_einsum_and_divides_t_by_scaling_factor(self):
        layer = self._make_linear_layer_without_real_cache(
            head_size=16, rotary_dim=8, max_pos=8, scaling_factor=2,
            is_neox_style=True, dtype=torch.float16
        )

        inv = self._make_inv_freq_cpu_with_npu_stub(rotary_dim=layer.rotary_dim, dtype=torch.float32)
        layer._compute_inv_freq = MagicMock(return_value=inv)

        # spy: ensure "t = t / scaling_factor" happened, without relying on real numeric correctness
        orig_arange = rope_mod.torch.arange
        div_calls = []

        def arange_spy(*args, **kwargs):
            t0 = orig_arange(*args, **kwargs)
            t_mock = MagicMock()
            def _truediv(div):
                div_calls.append(div)
                return t0 / div
            t_mock.__truediv__.side_effect = _truediv
            return t_mock

        # ensure einsum branch used (and outer NOT used)
        einsum_called = {"n": 0}
        outer_called = {"n": 0}

        def fake_einsum(eq, a, b):
            einsum_called["n"] += 1
            # return minimal correct shape
            return torch.zeros((layer.max_position_embeddings * layer.scaling_factor, inv.numel()), dtype=torch.float32)

        def fake_outer(a, b):
            outer_called["n"] += 1
            return torch.zeros((layer.max_position_embeddings * layer.scaling_factor, inv.numel()), dtype=torch.float32)

        with patch.object(rope_mod.torch, "arange", side_effect=arange_spy), \
             patch.object(rope_mod.torch, "einsum", side_effect=fake_einsum), \
             patch.object(rope_mod.torch, "outer", side_effect=fake_outer):
            cos, sin = rope_mod.LinearScalingRotaryEmbedding._compute_cos_sin_cache(layer)

        self.assertGreaterEqual(einsum_called["n"], 1)
        self.assertEqual(outer_called["n"], 0)
        self.assertIn(layer.scaling_factor, div_calls)  # division happened with scaling_factor
        self.assertEqual(cos.shape, (layer.max_len, layer.rotary_dim))
        self.assertEqual(sin.shape, (layer.max_len, layer.rotary_dim))

    def test_linear_scaling_compute_cache_non_neox_style_uses_outer_and_reshapes_to_2d(self):
        layer = self._make_linear_layer_without_real_cache(
            head_size=16, rotary_dim=8, max_pos=8, scaling_factor=2,
            is_neox_style=False, dtype=torch.float16
        )

        inv = self._make_inv_freq_cpu_with_npu_stub(rotary_dim=layer.rotary_dim, dtype=torch.float32)
        layer._compute_inv_freq = MagicMock(return_value=inv)

        einsum_called = {"n": 0}
        outer_called = {"n": 0}

        def fake_einsum(*args, **kwargs):
            einsum_called["n"] += 1
            return torch.zeros((1, 1), dtype=torch.float32)

        def fake_outer(a, b):
            outer_called["n"] += 1
            return torch.zeros((layer.max_position_embeddings * layer.scaling_factor, inv.numel()), dtype=torch.float32)

        with patch.object(rope_mod.torch, "einsum", side_effect=fake_einsum), \
             patch.object(rope_mod.torch, "outer", side_effect=fake_outer):
            cos, sin = rope_mod.LinearScalingRotaryEmbedding._compute_cos_sin_cache(layer)

        self.assertEqual(einsum_called["n"], 0)
        self.assertGreaterEqual(outer_called["n"], 1)
        self.assertEqual(cos.shape, (layer.max_len, layer.rotary_dim))
        self.assertEqual(sin.shape, (layer.max_len, layer.rotary_dim))

    # ==================== ExtendedRotaryEmbedding ====================

    def test_extended_rope_compute_inv_freq_calls_super_and_then_apply_scaling(self):
        inv = torch.tensor([1.0, 2.0], dtype=torch.float32)
        scaled = torch.tensor([10.0, 20.0], dtype=torch.float32)

        with patch.object(
            rope_mod.RotaryEmbeddingTorchNpu,
            "_compute_inv_freq",
            autospec=True,
            return_value=inv,
        ) as super_mock, patch.object(
            rope_mod.ExtendedRotaryEmbedding,
            "apply_scaling",
            autospec=True,
            return_value=scaled,
        ) as scale_mock:
            layer = rope_mod.ExtendedRotaryEmbedding.__new__(rope_mod.ExtendedRotaryEmbedding)
            out = rope_mod.ExtendedRotaryEmbedding._compute_inv_freq(layer, base=10000)

        self.assertTrue(torch.equal(out, scaled))

        super_mock.assert_called_once()
        # autospec=True => call_args[0] = (self_layer, base)
        self.assertIs(super_mock.call_args[0][0], layer)
        self.assertEqual(super_mock.call_args[0][1], 10000)

        scale_mock.assert_called_once()
        self.assertIs(scale_mock.call_args[0][0], layer)
        self.assertTrue(torch.equal(scale_mock.call_args[0][1], inv))

    def test_extended_rope_apply_scaling_three_regions_high_mid_low(self):
        layer = rope_mod.ExtendedRotaryEmbedding.__new__(rope_mod.ExtendedRotaryEmbedding)

        wavelen_high = 1024.0
        wavelen_mid = 4096.0
        wavelen_low = 16384.0

        f_high = 2.0 * math.pi / wavelen_high
        f_mid = 2.0 * math.pi / wavelen_mid
        f_low = 2.0 * math.pi / wavelen_low

        freqs = torch.tensor([f_high, f_mid, f_low], dtype=torch.float32)
        out = layer.apply_scaling(freqs)

        self.assertEqual(out.shape, freqs.shape)
        self.assertEqual(out.dtype, freqs.dtype)
        self.assertEqual(out.device, freqs.device)

        self.assertTrue(torch.isclose(out[0], freqs[0], rtol=1e-6, atol=1e-8).item())

        self.assertTrue(torch.isclose(out[2], freqs[2] / rope_mod.SCALE_FACTOR, rtol=1e-6, atol=1e-8).item())

        lo = freqs[1] / rope_mod.SCALE_FACTOR
        hi = freqs[1]
        self.assertTrue((out[1] >= lo - 1e-12).item())
        self.assertTrue((out[1] <= hi + 1e-12).item())

    def test_extended_rope_apply_scaling_threshold_boundaries_match_expected_endpoints(self):
        layer = rope_mod.ExtendedRotaryEmbedding.__new__(rope_mod.ExtendedRotaryEmbedding)

        high_w = rope_mod.OLD_CONTEXT_LEN / rope_mod.HIGH_FREQ_FACTOR  # 2048
        low_w = rope_mod.OLD_CONTEXT_LEN / rope_mod.LOW_FREQ_FACTOR    # 8192

        f_at_high = 2.0 * math.pi / float(high_w)
        f_at_low = 2.0 * math.pi / float(low_w)

        freqs = torch.tensor([f_at_high, f_at_low], dtype=torch.float32)
        out = layer.apply_scaling(freqs)

        self.assertTrue(torch.isclose(out[0], freqs[0], rtol=1e-6, atol=1e-8).item())
        self.assertTrue(torch.isclose(out[1], freqs[1] / rope_mod.SCALE_FACTOR, rtol=1e-6, atol=1e-8).item())

    def test_extended_rope_apply_scaling_preserves_length_dtype_device_and_handles_empty(self):
        layer = rope_mod.ExtendedRotaryEmbedding.__new__(rope_mod.ExtendedRotaryEmbedding)

        freqs = torch.tensor([0.1, 1.0, 10.0], dtype=torch.float16)  # CPU
        out = layer.apply_scaling(freqs)
        self.assertEqual(out.shape, freqs.shape)
        self.assertEqual(out.dtype, freqs.dtype)
        self.assertEqual(out.device, freqs.device)

        empty = torch.empty((0,), dtype=torch.float16, device=freqs.device)
        out_empty = layer.apply_scaling(empty)
        self.assertEqual(out_empty.numel(), 0)
        self.assertEqual(out_empty.dtype, empty.dtype)
        self.assertEqual(out_empty.device, empty.device)

# ==================== DeepseekScalingRotaryEmbedding  ====================

    def _new_deepseek_layer_minimal(
        self,
        *,
        rotary_dim=8,
        max_pos=8,
        scaling_factor=2,
        base=10000,
        is_neox_style=True,
        mscale=1.0,
        beta_fast=32,
        beta_slow=1,
        extrapolation_factor=1.0,
    ):
        """
        Create a minimal DeepseekScalingRotaryEmbedding instance WITHOUT running parent __init__.
        CPU-only safe: used for calling _set_cos_sin_cache/_compute_cos_sin_cache/get_cos_sin/forward.
        """
        layer = rope_mod.DeepseekScalingRotaryEmbedding.__new__(rope_mod.DeepseekScalingRotaryEmbedding)
        nn.Module.__init__(layer)

        # minimal attributes used by methods
        layer.rotary_dim = int(rotary_dim)
        layer.max_position_embeddings = int(max_pos)
        layer.base = base
        layer.scaling_factor = scaling_factor  # keep it int-like to avoid arange corner cases
        layer.is_neox_style = bool(is_neox_style)

        layer.beta_fast = beta_fast
        layer.beta_slow = beta_slow
        layer.extrapolation_factor = extrapolation_factor
        layer.mscale = float(mscale)
        return layer

    def test_deepseek_scaling_rope_init_calls_set_cos_sin_cache_with_expected_args(self):
        if not hasattr(rope_mod, "DeepseekScalingRotaryEmbedding") or not hasattr(rope_mod, "DeepseekScalingRotaryEmbeddingGPU"):
            self.skipTest("DeepseekScalingRotaryEmbedding not available in this runtime")

        max_pos = 16

        with patch.object(rope_mod.DeepseekScalingRotaryEmbeddingGPU, "__init__", return_value=None) as super_init, \
             patch.object(rope_mod.DeepseekScalingRotaryEmbedding, "_set_cos_sin_cache", autospec=True) as set_cache:
            _ = rope_mod.DeepseekScalingRotaryEmbedding(
                head_size=128,
                rotary_dim=64,
                max_position_embeddings=max_pos,
                base=10000,
                is_neox_style=True,
                scaling_factor=2,
                dtype=torch.float16,
                extrapolation_factor=1,
                attn_factor=1,
                beta_fast=32,
                beta_slow=1,
                mscale=1,
                mscale_all_dim=0,
            )

        self.assertTrue(super_init.called)
        self.assertTrue(set_cache.called)
        # called as _set_cos_sin_cache(self, seq_len=..., device=..., dtype=...)
        _, kwargs = set_cache.call_args
        self.assertEqual(kwargs.get("seq_len"), max_pos)
        self.assertEqual(kwargs.get("device"), rope_mod.current_platform.device_type)
        self.assertEqual(kwargs.get("dtype"), torch.get_default_dtype())

    def test_deepseek_set_cos_sin_cache_registers_inv_freq_cos_cached_sin_cached_with_expected_shapes(self):
        layer = self._new_deepseek_layer_minimal(rotary_dim=8, max_pos=8, scaling_factor=2, is_neox_style=True, mscale=1.0)

        # CPU-only
        layer._set_cos_sin_cache(seq_len=layer.max_position_embeddings, device="cpu", dtype=torch.float32)

        bufs = dict(layer.named_buffers())
        self.assertIn("inv_freq", bufs)
        self.assertIn("cos_cached", bufs)
        self.assertIn("sin_cached", bufs)

        self.assertEqual(bufs["inv_freq"].shape, (layer.rotary_dim // 2,))
        self.assertEqual(bufs["cos_cached"].shape, (layer.max_position_embeddings * layer.scaling_factor, layer.rotary_dim))
        self.assertEqual(bufs["sin_cached"].shape, (layer.max_position_embeddings * layer.scaling_factor, layer.rotary_dim))

        # persistent=False => should not appear in state_dict
        sd = layer.state_dict()
        self.assertNotIn("inv_freq", sd)
        self.assertNotIn("cos_cached", sd)
        self.assertNotIn("sin_cached", sd)

    def test_deepseek_compute_cos_sin_cache_outputs_concat_cache_and_supports_neox_and_non_neox(self):
        for is_neox_style in (True, False):
            with self.subTest(is_neox_style=is_neox_style):
                layer = self._new_deepseek_layer_minimal(
                    rotary_dim=8,
                    max_pos=8,
                    scaling_factor=2,
                    is_neox_style=is_neox_style,
                    mscale=1.0,
                )

                # Avoid calling the real _compute_inv_freq (it uses .npu()) by patching instance method
                inv_freq = torch.ones((layer.rotary_dim // 2,), dtype=torch.float32)
                layer._compute_inv_freq = MagicMock(return_value=inv_freq)

                cache = layer._compute_cos_sin_cache()
                self.assertEqual(cache.shape, (layer.max_position_embeddings * layer.scaling_factor, 2 * layer.rotary_dim))

                # must be chunkable
                cos, sin = cache.chunk(2, dim=-1)
                self.assertEqual(cos.shape, (layer.max_position_embeddings * layer.scaling_factor, layer.rotary_dim))
                self.assertEqual(sin.shape, (layer.max_position_embeddings * layer.scaling_factor, layer.rotary_dim))

                layer._compute_inv_freq.assert_called_once_with(layer.scaling_factor)

    def test_deepseek_get_cos_sin_applies_offsets_and_returns_b11d_shape(self):
        layer = self._new_deepseek_layer_minimal(rotary_dim=8, max_pos=8, scaling_factor=2, is_neox_style=True)

        L = layer.max_position_embeddings * layer.scaling_factor
        D = layer.rotary_dim
        cos_cached = torch.arange(L * D, dtype=torch.float32).reshape(L, D)
        sin_cached = cos_cached + 0.25

        layer.register_buffer("cos_cached", cos_cached, persistent=False)
        layer.register_buffer("sin_cached", sin_cached, persistent=False)

        positions = torch.tensor([0, 2, 3], dtype=torch.long)
        offsets = torch.tensor([1, 0, 2], dtype=torch.long)
        idx = positions + offsets

        cos, sin = layer.get_cos_sin(positions, offsets=offsets)
        self.assertEqual(cos.shape, (3, 1, 1, D))
        self.assertEqual(sin.shape, (3, 1, 1, D))
        self.assertTrue(torch.equal(cos, cos_cached[idx].view(3, 1, 1, D)))
        self.assertTrue(torch.equal(sin, sin_cached[idx].view(3, 1, 1, D)))

        cos2, sin2 = layer.get_cos_sin(positions, offsets=None)
        self.assertEqual(cos2.shape, (3, 1, 1, D))
        self.assertEqual(sin2.shape, (3, 1, 1, D))

    def test_deepseek_forward_runs_with_offsets_and_preserves_shapes_for_neox_and_non_neox(self):
        for is_neox_style in (True, False):
            with self.subTest(is_neox_style=is_neox_style):
                layer = self._new_deepseek_layer_minimal(
                    rotary_dim=8,
                    max_pos=8,
                    scaling_factor=2,
                    is_neox_style=is_neox_style,
                    mscale=1.0,
                )

                # Provide cos_sin_cache on CPU: shape (L, 2*D)
                L = layer.max_position_embeddings * layer.scaling_factor
                D = layer.rotary_dim
                cos_sin_cache = torch.zeros((L, 2 * D), dtype=torch.float32)
                layer.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

                T = 4
                positions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
                offsets = torch.tensor([1, 0, 1, 0], dtype=torch.long)

                # query/key shape must be broadcastable with cos/sin after view -> (T,1,D)
                query = torch.randn((T, 1, D), dtype=torch.float32)
                key = torch.randn((T, 1, D), dtype=torch.float32)

                rotate_neox = MagicMock(side_effect=lambda x: x)
                rotate_gptj = MagicMock(side_effect=lambda x: x)

                with patch.object(rope_mod, "_rotate_neox", rotate_neox), \
                     patch.object(rope_mod, "_rotate_gptj", rotate_gptj):
                    q_out, k_out = layer.forward(positions, query, key, offsets=offsets)

                self.assertEqual(q_out.shape, query.shape)
                self.assertEqual(k_out.shape, key.shape)

                if is_neox_style:
                    self.assertTrue(rotate_neox.called)
                    self.assertFalse(rotate_gptj.called)
                else:
                    self.assertTrue(rotate_gptj.called)
                    self.assertFalse(rotate_neox.called)

# ==================== QwenRotaryEmbedding ====================

    def _make_qwen_layer(
        self,
        *,
        head_size: int,
        rotary_dim: int,
        max_pos: int = 16,
        base: int = 10000,
        is_neox_style: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        return rope_mod.QwenRotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_pos,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

    def test_qwen_init_registers_cos_sin_buffers_and_shapes(self):
        max_pos = 32
        rotary_dim = 8
        layer = self._make_qwen_layer(head_size=8, rotary_dim=rotary_dim, max_pos=max_pos)

        # basic attrs
        self.assertEqual(layer.max_position_embeddings, max_pos)
        self.assertEqual(layer.rotary_dim, rotary_dim)
        self.assertEqual(layer.head_size, 8)

        # buffers exist
        buffers = dict(layer.named_buffers())
        self.assertIn("cos", buffers)
        self.assertIn("sin", buffers)

        # shape contract: [max_len, rotary_dim]
        self.assertEqual(buffers["cos"].shape, (max_pos, rotary_dim))
        self.assertEqual(buffers["sin"].shape, (max_pos, rotary_dim))

        # persistent=False: should not appear in state_dict
        sd = layer.state_dict()
        self.assertNotIn("cos", sd)
        self.assertNotIn("sin", sd)

    def test_qwen_get_cos_sin_offsets_and_shape_contract(self):
        layer = self._make_qwen_layer(head_size=8, rotary_dim=8, max_pos=16)

        positions = torch.tensor([0, 2, 3], dtype=torch.long)
        offsets = torch.tensor([1, 0, 2], dtype=torch.long)

        cos, sin = layer.get_cos_sin(positions, offsets=offsets)
        self.assertEqual(cos.shape, (3, layer.rotary_dim))
        self.assertEqual(sin.shape, (3, layer.rotary_dim))

        idx = positions + offsets
        expected_cos = layer.cos[idx].view(3, -1)
        expected_sin = layer.sin[idx].view(3, -1)
        self.assertTrue(torch.equal(cos, expected_cos))
        self.assertTrue(torch.equal(sin, expected_sin))

        # offsets=None branch
        cos2, sin2 = layer.get_cos_sin(positions, offsets=None)
        self.assertEqual(cos2.shape, (3, layer.rotary_dim))
        self.assertEqual(sin2.shape, (3, layer.rotary_dim))

    def test_qwen_forward_smallops_path_preserves_2d_shape_when_rotary_dim_not_128(self):
        # rotary_dim != 128 => small-ops path
        # 注意：这里让 rotary_dim == head_size，保证 cos/sin 的最后一维能广播到 head_size
        head_size = 8
        rotary_dim = 8
        nheads = 2
        T = 5

        layer = self._make_qwen_layer(head_size=head_size, rotary_dim=rotary_dim, max_pos=32)

        position_ids = torch.arange(T, dtype=torch.long)
        query = torch.randn((T, nheads * head_size), dtype=torch.float32)
        key = torch.randn((T, nheads * head_size), dtype=torch.float32)

        cos, sin = layer.get_cos_sin(position_ids, offsets=None)

        q_out, k_out = layer.forward(position_ids, query, key, cos, sin)
        self.assertEqual(q_out.shape, (T, nheads * head_size))
        self.assertEqual(k_out.shape, (T, nheads * head_size))

    def test_qwen_forward_fused_path_calls_torch_npu_and_flattens_when_rotary_dim_128(self):
        # rotary_dim == 128 => fused path (torch_npu)
        if (not hasattr(rope_mod, "torch_npu")) or (not hasattr(rope_mod.torch_npu, "npu_apply_rotary_pos_emb")):
            self.skipTest("torch_npu.npu_apply_rotary_pos_emb not available in this environment")

        head_size = 128
        rotary_dim = 128
        nheads = 2
        T = 4

        layer = self._make_qwen_layer(head_size=head_size, rotary_dim=rotary_dim, max_pos=32)

        position_ids = torch.arange(T, dtype=torch.long)
        query = torch.randn((T, nheads * head_size), dtype=torch.float32)
        key = torch.randn((T, nheads * head_size), dtype=torch.float32)

        cos, sin = layer.get_cos_sin(position_ids, offsets=None)

        calls = {"n": 0}

        def _fake_apply(q, k, cos4d, sin4d):
            calls["n"] += 1
            # shape contract checks only
            self.assertEqual(q.dim(), 4)      # [T, 1, nheads, head_size]
            self.assertEqual(k.dim(), 4)
            self.assertEqual(cos4d.dim(), 4)  # [T, 1, 1, rotary_dim]
            self.assertEqual(sin4d.dim(), 4)

            self.assertEqual(q.shape[0], T)
            self.assertEqual(q.shape[1], 1)
            self.assertEqual(q.shape[2], nheads)
            self.assertEqual(q.shape[3], head_size)

            self.assertEqual(cos4d.shape[0], T)
            self.assertEqual(cos4d.shape[1], 1)
            self.assertEqual(cos4d.shape[2], 1)
            self.assertEqual(cos4d.shape[3], rotary_dim)

            # return same tensors to let caller flatten back
            return q, k

        with patch.object(rope_mod.torch_npu, "npu_apply_rotary_pos_emb", side_effect=_fake_apply):
            q_out, k_out = layer.forward(position_ids, query, key, cos, sin)

        self.assertEqual(calls["n"], 1)
        self.assertEqual(q_out.shape, (T, nheads * head_size))
        self.assertEqual(k_out.shape, (T, nheads * head_size))

    # ==================== QwenMRotaryEmbedding  ====================

    def _make_qwen_mrope_stub(
        self,
        *,
        head_size=8,
        rotary_dim=8,
        max_pos=32,
        mrope_section=None,
        is_neox_style=True,
    ):
        # 避免走 GPUMRotaryEmbedding 的复杂 __init__，只搭 forward 需要的最小契约
        m = rope_mod.QwenMRotaryEmbedding.__new__(rope_mod.QwenMRotaryEmbedding)
        nn.Module.__init__(m)
        m.head_size = head_size
        m.rotary_dim = rotary_dim
        m.is_neox_style = is_neox_style
        m.mrope_section = mrope_section
        # cos_sin_cache: [max_pos, 2 * rotary_dim]，供 cos/sin chunk(2, -1)
        m.cos_sin_cache = torch.zeros((max_pos, rotary_dim * 2), dtype=torch.float32)
        return m

    def test_qwen_mrope_forward_positions_1d_runs_and_keeps_shape(self):
        T = 4
        head_size = 8
        rotary_dim = 8
        num_heads = 2
        hidden = num_heads * head_size

        m = self._make_qwen_mrope_stub(head_size=head_size, rotary_dim=rotary_dim, mrope_section=None)

        positions = torch.arange(T, dtype=torch.long)  # ndim==1
        query = torch.randn((T, hidden), dtype=torch.float32)
        key = torch.randn((T, hidden), dtype=torch.float32)

        calls = {"n": 0}

        def _fake_apply(x, cos, sin, is_neox_style):
            # 1D 分支：cos/sin 应该是 [T, rotary_dim]
            self.assertEqual(cos.ndim, 2)
            self.assertEqual(sin.ndim, 2)
            self.assertEqual(cos.shape[0], T)
            self.assertEqual(cos.shape[-1], rotary_dim)
            calls["n"] += 1
            return x  # 不测数值，直接透传

        with patch.object(rope_mod, "_apply_rotary_emb_torch", side_effect=_fake_apply):
            out_q, out_k = m.forward(positions=positions, query=query, key=key)

        self.assertEqual(out_q.shape, query.shape)
        self.assertEqual(out_k.shape, key.shape)
        self.assertEqual(calls["n"], 2)  # query/key 各调用一次

    def test_qwen_mrope_forward_positions_2d_runs_and_uses_mrope_section(self):
        T = 4
        head_size = 8
        rotary_dim = 8
        num_heads = 2
        hidden = num_heads * head_size

        # 关键：positions.ndim==2 时必须走 mrope_section 重组分支，否则 cos/sin 维度会是 [3, T, D] 不匹配
        mrope_section = [2, 3, 3]  # sum == rotary_dim
        m = self._make_qwen_mrope_stub(head_size=head_size, rotary_dim=rotary_dim, mrope_section=mrope_section)

        positions = torch.stack([
            torch.arange(T, dtype=torch.long),
            torch.arange(T, dtype=torch.long),
            torch.arange(T, dtype=torch.long),
        ], dim=0)  # shape [3, T]

        query = torch.randn((T, hidden), dtype=torch.float32)
        key = torch.randn((T, hidden), dtype=torch.float32)

        def _fake_apply(x, cos, sin, is_neox_style):
            # 2D 分支：mrope_section 重组后，cos/sin 必须被压回 [T, rotary_dim]
            self.assertEqual(cos.ndim, 2)
            self.assertEqual(sin.ndim, 2)
            self.assertEqual(cos.shape, (T, rotary_dim))
            self.assertEqual(sin.shape, (T, rotary_dim))
            return x

        with patch.object(rope_mod, "_apply_rotary_emb_torch", side_effect=_fake_apply):
            out_q, out_k = m.forward(positions=positions, query=query, key=key)

        self.assertEqual(out_q.shape, query.shape)
        self.assertEqual(out_k.shape, key.shape)


if __name__ == "__main__":
    unittest.main()
