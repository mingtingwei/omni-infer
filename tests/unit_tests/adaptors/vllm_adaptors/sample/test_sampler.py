import os
import sys
import types
import contextlib
import pytest  # noqa: F401
import unittest
from unittest import mock  # noqa: F401
from unittest.mock import Mock, patch, MagicMock  # noqa: F401
import importlib

import torch
from torch import nn  # noqa: F401
from torch.nn import Parameter  # noqa: F401


class _DummyDefaultStream:
    def wait_stream(self, _s):
        return None


class _DummyTorchNPUAttr:
    """Patch到 torch.npu 上的对象，提供 Stream/default_stream。"""

    class Stream:
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs
        def wait_stream(self, _s):
            return None

    def default_stream(self):
        return _DummyDefaultStream()


@contextlib.contextmanager
def _nullctx():
    yield


class TestSampler(unittest.TestCase):
    def setUp(self):
        # --- isolate env ---
        self._stack = contextlib.ExitStack()

        # Patch torch.npu (create=True to work even if torch.npu doesn't exist on CPU env)
        self._stack.enter_context(
            patch.object(torch, "npu", new=_DummyTorchNPUAttr(), create=True)
        )

        # --- build stub modules (torch_npu / vllm / omni.* minimal APIs) ---
        torch_npu_mod = types.ModuleType("torch_npu")

        class _TorchNPUNPU:
            @contextlib.contextmanager
            def stream(self, _stream):
                yield

        torch_npu_mod.npu = _TorchNPUNPU()
        # default stub; tests can override per-case
        torch_npu_mod.npu_top_k_top_p_sample = MagicMock(
            side_effect=lambda logits, k, p, q, **kwargs: (
                logits.argmax(dim=-1),
                logits.to(torch.float32),
            )
        )

        # vllm.v1.outputs.SamplerOutput
        vllm_outputs_mod = types.ModuleType("vllm.v1.outputs")

        class SamplerOutput:
            def __init__(self, sampled_token_ids):
                self.sampled_token_ids = sampled_token_ids

        vllm_outputs_mod.SamplerOutput = SamplerOutput

        # vllm.v1.sample.metadata.SamplingMetadata
        vllm_meta_mod = types.ModuleType("vllm.v1.sample.metadata")

        class SamplingMetadata:

            def __init__(self, **kwargs):
                # core
                self.temperature = kwargs.get("temperature", None)
                self.all_greedy = kwargs.get("all_greedy", False)
                # ✅ 修复：expand_sampling_metadata 会读取 all_random
                self.all_random = kwargs.get("all_random", False)

                self.top_p = kwargs.get("top_p", None)
                self.top_k = kwargs.get("top_k", None)
                self.min_p = kwargs.get("min_p", None)

                # others used by sampler.py
                self.max_num_logprobs = kwargs.get("max_num_logprobs", None)
                self.prompt_token_ids = kwargs.get("prompt_token_ids", None)
                self.frequency_penalties = kwargs.get("frequency_penalties", None)
                self.presence_penalties = kwargs.get("presence_penalties", None)
                self.repetition_penalties = kwargs.get("repetition_penalties", None)
                self.output_token_ids = kwargs.get("output_token_ids", None)
                self.min_tokens = kwargs.get("min_tokens", None)
                self.no_penalties = kwargs.get("no_penalties", False)
                self.logit_bias = kwargs.get("logit_bias", None)
                self.allowed_token_ids_mask = kwargs.get("allowed_token_ids_mask", None)
                self.bad_words_token_ids = kwargs.get("bad_words_token_ids", None)
                self.generators = kwargs.get("generators", {}) or {}
                self.seq_data = kwargs.get("seq_data", None)

                # store any additional fields
                for k, v in kwargs.items():
                    setattr(self, k, v)

            # 保守兜底：如果业务未来读取了新字段，这里返回 None，避免 UT 因 stub 缺字段而误报
            def __getattr__(self, name):
                return None

        vllm_meta_mod.SamplingMetadata = SamplingMetadata

        # vllm.v1.sample.ops.penalties.apply_min_token_penalties
        vllm_penalties_mod = types.ModuleType("vllm.v1.sample.ops.penalties")

        def apply_min_token_penalties(_logits, _output_token_ids, _min_tokens):
            return None

        vllm_penalties_mod.apply_min_token_penalties = apply_min_token_penalties

        # vllm.v1.sample.sampler.Sampler (base class)
        vllm_sampler_mod = types.ModuleType("vllm.v1.sample.sampler")

        class Sampler:
            def __init__(self):
                pass

            # identity helpers (will be patched per test when we need call assertions)
            def apply_allowed_token_ids(self, logits, _sampling_metadata):
                return logits

            def apply_bad_words(self, logits, _sampling_metadata):
                return logits

            def apply_logits_bias(self, logits, _sampling_metadata):
                return logits

            def apply_temperature(self, logits, temperature):
                if temperature is None:
                    return logits
                t = temperature
                if isinstance(t, torch.Tensor):
                    t = torch.clamp(t, min=1e-8).unsqueeze(-1)
                    return logits / t
                return logits / max(float(t), 1e-8)

            def forward(self, logits, _sampling_metadata):
                return SamplerOutput(sampled_token_ids=logits.argmax(dim=-1))

        vllm_sampler_mod.Sampler = Sampler

        # vllm.v1.spec_decode.metadata.SpecDecodeMetadata
        vllm_spec_mod = types.ModuleType("vllm.v1.spec_decode.metadata")

        class SpecDecodeMetadata:
            def __init__(
                self,
                num_draft_tokens,
                cu_num_draft_tokens,
                max_spec_len=0,
                bonus_logits_indices=None,
            ):
                self.num_draft_tokens = list(num_draft_tokens)
                self.cu_num_draft_tokens = cu_num_draft_tokens
                self.max_spec_len = int(max_spec_len)
                if bonus_logits_indices is None:
                    bonus_logits_indices = torch.zeros(
                        (len(self.num_draft_tokens),),
                        dtype=torch.int64,
                        device=cu_num_draft_tokens.device,
                    )
                self.bonus_logits_indices = bonus_logits_indices

        vllm_spec_mod.SpecDecodeMetadata = SpecDecodeMetadata

        # omni.models.config_loader.loader.model_extra_config
        omni_cfg_loader_mod = types.ModuleType("omni.models.config_loader.loader")
        omni_cfg_loader_mod.model_extra_config = {}

        # omni.layers.npu_sampler_cache PenaltyCache/ProbCache stubs
        omni_cache_mod = types.ModuleType("omni.layers.npu_sampler_cache")

        class PenaltyCache:
            def __init__(self, *args, **kwargs):
                self.do_penalties = True
                self.do_repetition_penalties = True
                self.prompt_mask = None
                self.output_mask = None
                self.output_bin_counts = None
                self.do_presence_penalties = True
                self.do_frequency_penalties = True
                self.do_repetition_penalties = True

            def save_token_ids(self, _token_ids):
                return None

            def revert_rejected_tokens(self, _accepted_mask, _token_ids):
                return None

            def prepare_cache(self, *args, **kwargs):
                return None

        class ProbCache:
            def __init__(self, *args, **kwargs):
                pass

            def prepare_cache(self, *args, **kwargs):
                return None

        omni_cache_mod.PenaltyCache = PenaltyCache
        omni_cache_mod.ProbCache = ProbCache

        # omni.layers.sampler.AscendTopKTopPSamplerV1 stub
        omni_sampler_layer_mod = types.ModuleType("omni.layers.sampler")

        class AscendTopKTopPSamplerV1:
            def __init__(self, *args, **kwargs):
                pass

        omni_sampler_layer_mod.AscendTopKTopPSamplerV1 = AscendTopKTopPSamplerV1

        # Apply sys.modules patches (restored in tearDown via ExitStack)
        self._stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "torch_npu": torch_npu_mod,
                    "vllm.v1.outputs": vllm_outputs_mod,
                    "vllm.v1.sample.metadata": vllm_meta_mod,
                    "vllm.v1.sample.ops.penalties": vllm_penalties_mod,
                    "vllm.v1.sample.sampler": vllm_sampler_mod,
                    "vllm.v1.spec_decode.metadata": vllm_spec_mod,
                    "omni.models.config_loader.loader": omni_cfg_loader_mod,
                    "omni.layers.npu_sampler_cache": omni_cache_mod,
                    "omni.layers.sampler": omni_sampler_layer_mod,
                },
                clear=False,
            )
        )

        # Make sure env is restored for each test
        self._stack.enter_context(patch.dict(os.environ, {}, clear=False))

        # Import module under test freshly each time
        self._target_mod_name = "omni.adaptors.vllm.sample.sampler"
        self._prev_target_mod = sys.modules.get(self._target_mod_name, None)
        if self._target_mod_name in sys.modules:
            del sys.modules[self._target_mod_name]

        self.SAMPLER = importlib.import_module(self._target_mod_name)
        self.SAMPLER = importlib.reload(self.SAMPLER)

        # dummy runner for AscendSamplerV1 __init__
        class _Runner:
            device = torch.device("cpu")
            max_num_reqs = 8
            use_penalty = False
            use_rejection_sampler = False

            class batch:
                vocab_size = 64

        self.runner = _Runner()
        self.sampler = self.SAMPLER.AscendSamplerV1(self.runner)

    def tearDown(self):
        stack = getattr(self, "_stack", None)
        try:
            if stack is not None:
                stack.close()
        finally:
            target = getattr(self, "_target_mod_name", None)
            prev = getattr(self, "_prev_target_mod", None)
            if not target:
                return

            def _unlink_module(mod_name: str):
                """从 sys.modules 移除模块，并把父包上指向它的同名属性也解除，避免悬挂引用。"""
                mod = sys.modules.get(mod_name)
                if mod is None:
                    return

                # 先解除父包属性引用：pkg.child -> module
                if "." in mod_name:
                    parent_name, attr = mod_name.rsplit(".", 1)
                    parent = sys.modules.get(parent_name)
                    if parent is not None and getattr(parent, attr, None) is mod:
                        try:
                            delattr(parent, attr)
                        except Exception:
                            pass

                # 再移除 sys.modules
                try:
                    del sys.modules[mod_name]
                except KeyError:
                    pass

            # 目标模块的父包：omni.adaptors.vllm.sample
            parent_pkg_name, child_attr = target.rsplit(".", 1)
            parent_pkg = sys.modules.get(parent_pkg_name)

            if prev is None:
                # 本用例首次 import 了 target，最容易出现“半初始化包对象被缓存”的污染
                # 如果父包已经具备 validator，说明它可能早就被别的用例正常初始化过，就别做重清理。
                parent_looks_healthy = bool(parent_pkg is not None and hasattr(parent_pkg, "validator"))

                if parent_looks_healthy:
                    # 仅清理 leaf 模块 + 解除父包上的 sampler 引用即可
                    _unlink_module(target)
                else:
                    # 清理整个 omni.adaptors.vllm.sample.* 子树，强制后续用例重新 import 正常环境下的包
                    prefix = parent_pkg_name  # "omni.adaptors.vllm.sample"
                    to_del = [
                        name for name in list(sys.modules.keys())
                        if name == prefix or name.startswith(prefix + ".")
                    ]
                    # 先删子模块再删包本身
                    for name in sorted(to_del, key=len, reverse=True):
                        _unlink_module(name)

                    # 额外：如果上一级包还挂着 sample 属性且指向“已删除的旧对象”，也解除掉
                    if "." in prefix:
                        gp_name, gp_attr = prefix.rsplit(".", 1)  # omni.adaptors.vllm , sample
                        gp = sys.modules.get(gp_name)
                        if gp is not None:
                            try:
                                m = getattr(gp, gp_attr, None)
                                if getattr(m, "__name__", None) == prefix:
                                    delattr(gp, gp_attr)
                            except Exception:
                                pass
            else:
                # 目标模块原本就存在：恢复到原对象，避免破坏其它用例的 import 预期
                sys.modules[target] = prev
                # 同步父包属性（避免 parent_pkg.sampler 悬挂到我们 reload 后的对象）
                parent = sys.modules.get(parent_pkg_name)
                if parent is not None:
                    try:
                        setattr(parent, child_attr, prev)
                    except Exception:
                        pass


    # --------------------------
    # tests
    # --------------------------

    def test_apply_top_k_only_masks_non_topk_for_logits_and_prob(self):
        logits = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [5.0, 4.0, 3.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        k = torch.tensor([2, logits.shape[1]], dtype=torch.int64)
        out = self.SAMPLER.apply_top_k_only(logits.clone(), k.clone(), is_logits=True)
        self.assertTrue(torch.isinf(out[0, 0]) and out[0, 0] < 0)
        self.assertTrue(torch.isinf(out[0, 1]) and out[0, 1] < 0)
        self.assertTrue(torch.isinf(out[0, 2]) and out[0, 2] < 0)
        self.assertFalse(torch.isinf(out[0, 3]))
        self.assertFalse(torch.isinf(out[0, 4]))
        self.assertTrue(torch.isfinite(out[1]).all().item())

        prob = torch.tensor([[0.01, 0.02, 0.03, 0.04, 0.90]], dtype=torch.float32)
        k2 = torch.tensor([3], dtype=torch.int64)
        out2 = self.SAMPLER.apply_top_k_only(prob.clone(), k2.clone(), is_logits=False)
        self.assertEqual(float(out2[0, 0].item()), 0.0)
        self.assertEqual(float(out2[0, 1].item()), 0.0)
        self.assertGreater(out2[0, 2].item(), 0.0)

    def test_apply_top_k_top_p_when_p_is_none_returns_normalized_probs_and_idx_none(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        probs, idx = self.SAMPLER.apply_top_k_top_p(logits.clone(), None, None, is_logits=True)
        self.assertIsNone(idx)
        self.assertEqual(probs.shape, logits.shape)
        s = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(s, torch.ones_like(s), atol=1e-5, rtol=0))

        k = torch.tensor([2], dtype=torch.int64)
        probs2, idx2 = self.SAMPLER.apply_top_k_top_p(logits.clone(), k, None, is_logits=True)
        self.assertIsNone(idx2)
        self.assertEqual(probs2.shape, logits.shape)
        s2 = probs2.sum(dim=-1)
        self.assertTrue(torch.allclose(s2, torch.ones_like(s2), atol=1e-5, rtol=0))
        self.assertTrue((probs2[0] == 0).any().item())

    def test_apply_top_k_top_p_with_p_applies_threshold_and_returns_sort_index(self):
        logits = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        p = torch.tensor([0.5, 0.7], dtype=torch.float32)
        probs, idx = self.SAMPLER.apply_top_k_top_p(logits.clone(), None, p, is_logits=True)
        self.assertIsNotNone(idx)
        self.assertEqual(probs.shape, logits.shape)
        self.assertEqual(idx.shape, logits.shape)
        s = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(s, torch.ones_like(s), atol=1e-5, rtol=0))
        for r in range(idx.shape[0]):
            uniq = torch.unique(idx[r])
            self.assertEqual(int(uniq.numel()), idx.shape[1])

    def test_expand_sampling_metadata_repeats_params_and_temperature_eps_floor(self):
        SamplingMetadata = self.SAMPLER.SamplingMetadata
        SpecDecodeMetadata = self.SAMPLER.SpecDecodeMetadata

        sm = SamplingMetadata(
            temperature=torch.tensor([0.0, 1e-9], dtype=torch.float32),
            all_greedy=False,
            all_random=False,  # ✅ 明确给出，和真实 vLLM 行为一致
            top_k=torch.tensor([2, 3], dtype=torch.int64),
            top_p=torch.tensor([0.9, 0.8], dtype=torch.float32),
            min_p=torch.tensor([0.1, 0.2], dtype=torch.float32),
            allowed_token_ids_mask=torch.ones((2, 5), dtype=torch.bool),
            generators={},
            no_penalties=True,
        )
        spec = SpecDecodeMetadata(
            num_draft_tokens=[1, 2],
            cu_num_draft_tokens=torch.tensor([1, 3], dtype=torch.int64),
        )

        out = self.sampler.expand_sampling_metadata(sm, spec)
        self.assertEqual(int(out.temperature.numel()), 5)
        self.assertTrue((out.temperature >= 1e-5).all().item())
        self.assertEqual(int(out.top_k.numel()), 5)
        self.assertEqual(int(out.top_p.numel()), 5)
        self.assertEqual(int(out.min_p.numel()), 5)
        self.assertEqual(tuple(out.allowed_token_ids_mask.shape), (5, 5))
        self.assertFalse(bool(out.all_random))

        sm2 = SamplingMetadata(
            temperature=None,
            all_greedy=False,
            all_random=False,
            top_k=torch.tensor([2, 3], dtype=torch.int64),
            top_p=None,
            min_p=None,
            generators={},
            no_penalties=True,
        )
        out2 = self.sampler.expand_sampling_metadata(sm2, spec)
        self.assertEqual(int(out2.temperature.numel()), 5)
        self.assertTrue((out2.temperature >= 1e-5).all().item())

    def test_apply_min_p_masks_logits_when_return_logits_true_and_masks_probs_otherwise(self):
        logits = torch.tensor([[0.0, 1.0, 2.0, 10.0]], dtype=torch.float32)
        min_p = torch.tensor([0.9], dtype=torch.float32)
        masked_logits = self.sampler.apply_min_p(logits.clone(), min_p, return_logits=True)
        self.assertEqual(masked_logits.shape, logits.shape)
        self.assertTrue(torch.isinf(masked_logits).any().item())

        masked_probs = self.sampler.apply_min_p(logits.clone(), min_p, return_logits=False)
        self.assertEqual(masked_probs.shape, logits.shape)
        self.assertTrue((masked_probs == 0).any().item())
        self.assertGreater(masked_probs[0, 3].item(), 0.0)

    def test_apply_sampling_params_greedy_branch_returns_argmax_or_logits(self):
        SamplingMetadata = self.SAMPLER.SamplingMetadata
        logits = torch.tensor([[1.0, 3.0, 2.0], [0.0, -1.0, 5.0]], dtype=torch.float32)

        sm = SamplingMetadata(
            all_greedy=True,
            all_random=False,
            temperature=None,
            top_k=None,
            top_p=None,
            min_p=None,
            generators={},
            no_penalties=True,
        )

        self.sampler.apply_allowed_token_ids = MagicMock(side_effect=lambda x, _: x)
        self.sampler.apply_penalties = MagicMock(side_effect=lambda x, *_args, **_kw: x)
        self.sampler.apply_bad_words = MagicMock(side_effect=lambda x, _: x)
        self.sampler.apply_logits_bias = MagicMock(side_effect=lambda x, _: x)

        out_ids = self.sampler.apply_sampling_params(logits.clone(), sm, do_sample=True)
        self.assertEqual(tuple(out_ids.shape), (2,))
        self.assertTrue(torch.equal(out_ids, logits.argmax(dim=-1)))

        out_logits = self.sampler.apply_sampling_params(logits.clone(), sm, do_sample=False)
        self.assertEqual(tuple(out_logits.shape), tuple(logits.shape))

    def test_apply_sampling_params_fallback_path_env_disable_calls_apply_top_k_top_p_and_optional_do_sample(self):
        SamplingMetadata = self.SAMPLER.SamplingMetadata
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        sm = SamplingMetadata(
            all_greedy=False,
            all_random=False,
            temperature=torch.tensor([1.0], dtype=torch.float32),
            top_k=torch.tensor([2], dtype=torch.int64),
            top_p=None,
            min_p=None,
            generators={},
            no_penalties=True,
        )

        self.sampler.apply_allowed_token_ids = MagicMock(side_effect=lambda x, _: x)
        self.sampler.apply_penalties = MagicMock(side_effect=lambda x, *_a, **_k: x)
        self.sampler.apply_bad_words = MagicMock(side_effect=lambda x, _: x)
        self.sampler.apply_logits_bias = MagicMock(side_effect=lambda x, _: x)
        self.sampler.apply_temperature = MagicMock(side_effect=lambda x, _t: x)

        fake_probs = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
        fake_idx = None
        with patch.dict(os.environ, {"OMNI_DISABLE_NPU_TOP_K_TOP_P_SAMPLE": "1"}, clear=False), patch.object(
            self.SAMPLER, "apply_top_k_top_p", autospec=True, return_value=(fake_probs, fake_idx)
        ) as m_top, patch.object(
            self.sampler, "do_sample", autospec=True, return_value=torch.tensor([2], dtype=torch.int64)
        ) as m_do:
            out = self.sampler.apply_sampling_params(logits.clone(), sm, do_sample=True)
            self.assertTrue(torch.equal(out, torch.tensor([2], dtype=torch.int64)))
            m_top.assert_called()
            m_do.assert_called()

            out2 = self.sampler.apply_sampling_params(logits.clone(), sm, do_sample=False)
            self.assertIsInstance(out2, tuple)
            self.assertTrue(torch.equal(out2[0], fake_probs))
            self.assertEqual(out2[1], fake_idx)

    def test_apply_sampling_params_npu_path_calls_torch_npu_top_k_top_p_sample_and_returns_expected_shapes(self):
        SamplingMetadata = self.SAMPLER.SamplingMetadata
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]], dtype=torch.float32)
        sm = SamplingMetadata(
            all_greedy=False,
            all_random=False,
            temperature=torch.tensor([1.0, 1.0], dtype=torch.float32),
            top_k=torch.tensor([2, 2], dtype=torch.int64),
            top_p=None,
            min_p=None,
            generators={},
            no_penalties=True,
        )

        self.sampler.apply_allowed_token_ids = MagicMock(side_effect=lambda x, _: x)
        self.sampler.apply_penalties = MagicMock(side_effect=lambda x, *_a, **_k: x)
        self.sampler.apply_bad_words = MagicMock(side_effect=lambda x, _: x)
        self.sampler.apply_logits_bias = MagicMock(side_effect=lambda x, _: x)
        self.sampler.apply_temperature = MagicMock(side_effect=lambda x, _t: x)

        self.sampler.generate_random_sequence = MagicMock(
            side_effect=lambda probs, *_a, **_k: torch.ones_like(probs, dtype=torch.float32)
        )

        npu_fn = sys.modules["torch_npu"].npu_top_k_top_p_sample
        npu_fn.reset_mock()
        npu_fn.side_effect = lambda _logits, _k, _p, _q, **kwargs: (
            torch.tensor([1, 0], dtype=torch.int64),
            _logits.to(torch.float32),
        )

        with patch.dict(os.environ, {"OMNI_DISABLE_NPU_TOP_K_TOP_P_SAMPLE": "0"}, clear=False):
            out_ids = self.sampler.apply_sampling_params(logits.clone(), sm, do_sample=True)
            self.assertEqual(tuple(out_ids.shape), (2,))
            self.assertTrue(npu_fn.called)

            out_probs, out_idx = self.sampler.apply_sampling_params(logits.clone(), sm, do_sample=False)
            self.assertIsNone(out_idx)
            self.assertEqual(tuple(out_probs.shape), tuple(logits.shape))

    def test_spec_metadata_skips_bad_words_and_logits_bias_but_still_applies_allowed_ids_and_penalties(self):
        SamplingMetadata = self.SAMPLER.SamplingMetadata
        SpecDecodeMetadata = self.SAMPLER.SpecDecodeMetadata

        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        sm = SamplingMetadata(
            all_greedy=True,
            all_random=False,
            temperature=None,
            top_k=None,
            top_p=None,
            min_p=None,
            generators={},
            no_penalties=True,
        )
        spec = SpecDecodeMetadata(
            num_draft_tokens=[0],
            cu_num_draft_tokens=torch.tensor([0], dtype=torch.int64),
            max_spec_len=0,
            bonus_logits_indices=torch.tensor([0], dtype=torch.int64),
        )

        self.sampler.expand_sampling_metadata = MagicMock(side_effect=lambda a, b: a)
        self.sampler.apply_allowed_token_ids = MagicMock(side_effect=lambda x, _: x)
        self.sampler.apply_penalties = MagicMock(side_effect=lambda x, *_a, **_k: x)

        def _should_not_call(*_a, **_k):
            raise AssertionError("should not be called when spec_metadata is provided")

        self.sampler.apply_bad_words = MagicMock(side_effect=_should_not_call)
        self.sampler.apply_logits_bias = MagicMock(side_effect=_should_not_call)

        out = self.sampler.apply_sampling_params(logits.clone(), sm, spec_metadata=spec, do_sample=False)
        self.assertEqual(tuple(out.shape), tuple(logits.shape))
        self.sampler.apply_allowed_token_ids.assert_called()
        self.sampler.apply_penalties.assert_called()

    def test_forward_updates_penalty_cache_only_when_update_penalty_true(self):
        SamplingMetadata = self.SAMPLER.SamplingMetadata
        sm = SamplingMetadata(all_greedy=True, all_random=False, no_penalties=True)
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

        self.sampler.penalty_cache = MagicMock()
        self.sampler.penalty_cache.save_token_ids = MagicMock()

        fake_out = self.SAMPLER.SamplerOutput(sampled_token_ids=torch.tensor([2], dtype=torch.int64))
        base_sampler_mod = sys.modules["vllm.v1.sample.sampler"]
        with patch.object(base_sampler_mod.Sampler, "forward", autospec=True, return_value=fake_out):
            _ = self.sampler.forward(logits, sm, update_penalty=True)
            self.sampler.penalty_cache.save_token_ids.assert_called_once()

            self.sampler.penalty_cache.save_token_ids.reset_mock()
            _ = self.sampler.forward(logits, sm, update_penalty=False)
            self.sampler.penalty_cache.save_token_ids.assert_not_called()

    def test_revert_rejected_tokens_calls_penalty_cache_with_expected_masks_and_token_slices(self):
        SpecDecodeMetadata = self.SAMPLER.SpecDecodeMetadata
        pc = MagicMock()
        pc.do_penalties = True
        pc.revert_rejected_tokens = MagicMock()
        self.sampler.penalty_cache = pc

        batch = 2
        max_spec_len = 3
        input_ids = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64)
        accepted_num = torch.tensor([3, 1], dtype=torch.int64)

        spec = SpecDecodeMetadata(
            num_draft_tokens=[2, 0],
            cu_num_draft_tokens=torch.tensor([0, 2], dtype=torch.int64),
            max_spec_len=max_spec_len,
            bonus_logits_indices=torch.tensor([0, 2], dtype=torch.int64),
        )

        self.sampler.revert_rejected_tokens(accepted_num, input_ids, spec)

        self.assertEqual(pc.revert_rejected_tokens.call_count, max_spec_len)
        for call in pc.revert_rejected_tokens.call_args_list:
            args, _kwargs = call
            self.assertEqual(len(args), 2)
            mask, toks = args
            self.assertEqual(tuple(mask.shape), (batch,))
            self.assertEqual(tuple(toks.shape), (batch,))
            self.assertEqual(toks.dtype, input_ids.dtype)

    def test_prepare_cache_delegates_to_penalty_and_prob_cache(self):
        self.sampler.penalty_cache = MagicMock()
        self.sampler.prob_cache = MagicMock()
        self.sampler.penalty_cache.prepare_cache = MagicMock()
        self.sampler.prob_cache.prepare_cache = MagicMock()

        self.sampler.prepare_cache(1, 2, foo="bar")
        self.sampler.penalty_cache.prepare_cache.assert_called_once_with(1, 2, foo="bar")
        self.sampler.prob_cache.prepare_cache.assert_called_once_with(1, 2, foo="bar")


if __name__ == "__main__":
    unittest.main()
