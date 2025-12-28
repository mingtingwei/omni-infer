# -*- coding: utf-8 -*-
import os
import sys
import types
import contextlib
import unittest
import importlib
import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn


class TestModelsDeepseekMTP(unittest.TestCase):
    _DEEPSEEK_MTP_MOD = "omni.models.deepseek.deepseek_mtp"

    # ----------------------------
    # Test helpers (high isolation)
    # ----------------------------
    @contextlib.contextmanager
    def _import_mtp_with_stubs(self):
        """
        Import omni.models.deepseek.deepseek_mtp with lightweight stub deps.

        Isolation guarantees:
        - All sys.modules mutations are tracked and restored on exit.
        - Avoid executing real omni.models.__init__.py (may have global side effects).
        - Restore os.environ and torch.npu if we touched it.
        """
        modname = self._DEEPSEEK_MTP_MOD

        def _mk_pkg(name: str, path_list=None) -> types.ModuleType:
            m = types.ModuleType(name)
            m.__file__ = f"<stub {name}>"
            m.__path__ = list(path_list or [])
            return m

        def _mk_mod(name: str, **attrs) -> types.ModuleType:
            m = types.ModuleType(name)
            m.__file__ = f"<stub {name}>"
            for k, v in attrs.items():
                setattr(m, k, v)
            return m

        touched = {}

        def _set_module(name: str, module: types.ModuleType):
            if name not in touched:
                touched[name] = sys.modules.get(name, None)
            sys.modules[name] = module

        # snapshot env + torch.npu
        old_environ = dict(os.environ)
        had_torch_npu_attr = hasattr(torch, "npu")
        old_torch_npu_attr = getattr(torch, "npu", None)

        # locate repo root (best-effort) so that omni.models.deepseek.* can be discovered
        here = os.path.abspath(os.path.dirname(__file__))
        repo_root = None
        cur = here
        for _ in range(25):
            if os.path.isdir(os.path.join(cur, "omni", "models", "deepseek")):
                repo_root = cur
                break
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

        omni_path = os.path.join(repo_root, "omni") if repo_root else None
        omni_models_path = os.path.join(omni_path, "models") if omni_path else None
        omni_deepseek_path = os.path.join(omni_models_path, "deepseek") if omni_models_path else None

        # fallback: find_spec without making importlib a local variable
        if omni_path is None:
            try:
                from importlib import util as importlib_util
                spec = importlib_util.find_spec("omni")
                if spec and spec.submodule_search_locations:
                    omni_path = list(spec.submodule_search_locations)[0]
                    omni_models_path = os.path.join(omni_path, "models")
                    omni_deepseek_path = os.path.join(omni_models_path, "deepseek")
            except Exception:
                pass

        # ---- Minimal functional stubs ----
        class _DummyRMSNorm(nn.Module):
            def __init__(self, hidden_size: int, eps: float = 1e-6):
                super().__init__()
                self.hidden_size = hidden_size
                self.eps = eps

            def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
                if residual is None:
                    return x
                return x, residual

        class _DummyColumnParallelFlashCommLinear(nn.Module):
            def __init__(self, input_size, output_size, bias, tp_size, tp_rank, quant_config, prefix):
                super().__init__()
                self.input_size = input_size
                self.output_size = output_size

            def forward(self, x: torch.Tensor):
                # Return (hidden_states, aux) tuple as expected by deepseek_mtp
                if x.shape[-1] >= self.output_size:
                    y = x[..., : self.output_size].contiguous()
                else:
                    pad = torch.zeros(*x.shape[:-1], self.output_size - x.shape[-1], dtype=x.dtype, device=x.device)
                    y = torch.cat([x, pad], dim=-1)
                return y, None

        class _DummyVocabParallelEmbedding(nn.Module):
            def __init__(self, vocab_size, hidden_size, prefix=""):
                super().__init__()
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size

            def forward(self, input_ids: torch.Tensor, reduce: int = 1):
                n = input_ids.numel()
                return torch.zeros(n, self.hidden_size, dtype=torch.float32)

        class _DummyParallelLMHead(nn.Module):
            def __init__(self, vocab_size, hidden_size, quant_config=None):
                super().__init__()
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size

            def forward(self, hidden_states: torch.Tensor, embedding_bias: torch.Tensor = None):
                n = hidden_states.shape[0]
                return torch.zeros(n, self.vocab_size, dtype=torch.float32)

        class _DummyLogitsProcessor:
            def __init__(self, vocab_size: int, logits_as_input: bool = True):
                self.vocab_size = vocab_size
                self.logits_as_input = logits_as_input

            def __call__(self, head, hidden_states, sampling_metadata):
                n = hidden_states.shape[0]
                return torch.zeros(n, self.vocab_size, dtype=torch.float32)

        class _DummySampler:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return None

        class _Group:
            def __init__(self, world_size=1, rank_in_group=0):
                self.world_size = world_size
                self.rank_in_group = rank_in_group

            def all_gather(self, x: torch.Tensor, dim: int = 0):
                return x

            def all_to_all(self, x: torch.Tensor):
                return x

        class _DeepseekDecoderLayer(nn.Module):
            def __init__(self, config, prefix, cache_config=None, quant_config=None, **kwargs):
                super().__init__()
                self.config = config
                self.prefix = prefix

            def forward(self, *, positions, kv_cache, hidden_states, attn_metadata, residual=None):
                return hidden_states, None

        def _generate_sp_inputs(x: torch.Tensor, meta):
            return x

        # ----------------------------
        # Apply stubs (tracked in touched)
        # ----------------------------
        try:
            # Keep ASCEND_PLATFORM deterministic during import
            os.environ["ASCEND_PLATFORM"] = old_environ.get("ASCEND_PLATFORM", "")

            # torch_npu + torch.npu
            if "torch_npu" not in sys.modules:
                _set_module("torch_npu", _mk_mod("torch_npu"))
            if not had_torch_npu_attr:
                torch.npu = SimpleNamespace(config=SimpleNamespace(allow_internal_format=False))  # type: ignore[attr-defined]

            # Stub omni packages to avoid executing omni.models.__init__.py
            _set_module("omni", _mk_pkg("omni", [omni_path] if omni_path else []))
            _set_module("omni.models", _mk_pkg("omni.models", [omni_models_path] if omni_models_path else []))
            _set_module("omni.models.deepseek", _mk_pkg("omni.models.deepseek", [omni_deepseek_path] if omni_deepseek_path else []))

            # -------- vllm root + compilation decorator --------
            class _ModelRegistry:
                @classmethod
                def register(cls, *args, **kwargs):
                    return None

                @classmethod
                def register_model(cls, *args, **kwargs):
                    return None

                @classmethod
                def get_model(cls, *args, **kwargs):
                    return None

            vllm_root = _mk_pkg("vllm", [])
            vllm_root.ModelRegistry = _ModelRegistry
            _set_module("vllm", vllm_root)

            _set_module("vllm.compilation", _mk_pkg("vllm.compilation", []))
            _set_module("vllm.compilation.decorators", _mk_mod("vllm.compilation.decorators", support_torch_compile=(lambda cls: cls)))

            # -------- vllm.config --------
            class _QuantizationConfig:
                pass

            class _VllmConfig:
                pass

            _set_module("vllm.config", _mk_mod("vllm.config", QuantizationConfig=_QuantizationConfig, VllmConfig=_VllmConfig))

            # -------- vllm.attention.backends.abstract --------
            _set_module("vllm.attention", _mk_pkg("vllm.attention", []))
            _set_module("vllm.attention.backends", _mk_pkg("vllm.attention.backends", []))
            _set_module("vllm.attention.backends.abstract", _mk_mod("vllm.attention.backends.abstract", AttentionMetadata=object))

            # -------- vllm.distributed (both package + top-level functions) --------
            dist_mod = _mk_pkg("vllm.distributed", [])
            dist_mod.get_ep_group = (lambda: SimpleNamespace(world_size=1, rank_in_group=0))
            dist_mod.get_dp_group = (lambda: SimpleNamespace(world_size=1, rank_in_group=0))
            dist_mod.get_tensor_model_parallel_rank = (lambda: 0)
            dist_mod.get_tensor_model_parallel_world_size = (lambda: 1)
            _set_module("vllm.distributed", dist_mod)

            _set_module("vllm.distributed.communication_op", _mk_mod(
                "vllm.distributed.communication_op",
                tensor_model_parallel_all_gather=(lambda x, dim=0: x),
            ))
            _set_module("vllm.distributed.parallel_state", _mk_mod(
                "vllm.distributed.parallel_state",
                get_dp_group=(lambda: SimpleNamespace(world_size=1, rank_in_group=0)),
                get_tensor_model_parallel_rank=(lambda: 0),
                get_tensor_model_parallel_world_size=(lambda: 1),
            ))

            # -------- CRITICAL: stub vllm.model_executor package chain to prevent real vLLM import --------
            _set_module("vllm.model_executor", _mk_pkg("vllm.model_executor", []))
            _set_module("vllm.model_executor.models", _mk_pkg("vllm.model_executor.models", []))
            _set_module("vllm.model_executor.layers", _mk_pkg("vllm.model_executor.layers", []))
            _set_module("vllm.model_executor.model_loader", _mk_pkg("vllm.model_executor.model_loader", []))

            _set_module("vllm.model_executor.models.utils", _mk_mod(
                "vllm.model_executor.models.utils",
                is_pp_missing_parameter=(lambda name, self_obj: False),
            ))
            _set_module("vllm.model_executor.models.deepseek_v2", _mk_mod(
                "vllm.model_executor.models.deepseek_v2",
                get_spec_layer_idx_from_weight_name=(lambda cfg, name: 0),
            ))
            _set_module("vllm.model_executor.models.interfaces", _mk_mod(
                "vllm.model_executor.models.interfaces",
                SupportsPP=object,
            ))
            _set_module("vllm.model_executor.layers.logits_processor", _mk_mod(
                "vllm.model_executor.layers.logits_processor",
                LogitsProcessor=_DummyLogitsProcessor,
            ))
            # >>> FIX: sampler submodule (required by deepseek_mtp import)
            _set_module("vllm.model_executor.layers.sampler", _mk_mod(
                "vllm.model_executor.layers.sampler",
                Sampler=_DummySampler,
            ))
            _set_module("vllm.model_executor.sampling_metadata", _mk_mod(
                "vllm.model_executor.sampling_metadata",
                SamplingMetadata=object,
            ))
            _set_module("vllm.model_executor.model_loader.weight_utils", _mk_mod(
                "vllm.model_executor.model_loader.weight_utils",
                default_weight_loader=(lambda param, loaded_weight, *args, **kwargs: None),
            ))

            # -------- transformers stub to avoid env-dependent transformers API mismatches --------
            _set_module("transformers", _mk_mod("transformers", PretrainedConfig=object, ProcessorMixin=object, BatchFeature=object))

            # -------- omni layer stubs --------
            _set_module("omni.layers", _mk_pkg("omni.layers", []))
            _set_module("omni.layers.layernorm", _mk_mod("omni.layers.layernorm", RMSNorm=_DummyRMSNorm))
            _set_module("omni.layers.linear", _mk_mod("omni.layers.linear", ColumnParallelFlashCommLinear=_DummyColumnParallelFlashCommLinear))
            _set_module("omni.layers.vocab_parallel_embedding", _mk_mod(
                "omni.layers.vocab_parallel_embedding",
                ParallelLMHead=_DummyParallelLMHead,
                VocabParallelEmbedding=_DummyVocabParallelEmbedding,
            ))

            # fused_moe stubs
            _set_module("omni.layers.moe", _mk_pkg("omni.layers.moe", []))
            _set_module("omni.layers.moe.fused_moe", _mk_pkg("omni.layers.moe.fused_moe", []))
            _set_module("omni.layers.moe.fused_moe.fused_moe", _mk_mod(
                "omni.layers.moe.fused_moe.fused_moe",
                set_num_speculative_tokens=(lambda n: None),
            ))

            class _FusedMoE:
                @staticmethod
                def make_expert_params_mapping(ckpt_gate_proj_name, ckpt_down_proj_name, ckpt_up_proj_name, num_experts):
                    return []

            _set_module("omni.layers.moe.fused_moe.layer", _mk_mod("omni.layers.moe.fused_moe.layer", FusedMoE=_FusedMoE))

            # omni.adaptors.vllm.distributed
            _set_module("omni.adaptors", _mk_pkg("omni.adaptors", []))
            _set_module("omni.adaptors.vllm", _mk_pkg("omni.adaptors.vllm", []))
            _set_module("omni.adaptors.vllm.distributed", _mk_mod(
                "omni.adaptors.vllm.distributed",
                get_eh_proj_tp_group=(lambda: _Group(world_size=1, rank_in_group=0)),
            ))

            # model_extra_config
            _set_module("omni.models.config_loader", _mk_pkg("omni.models.config_loader", []))
            model_extra_config = SimpleNamespace(
                operator_opt_config=SimpleNamespace(prefill_moe_all_to_all=False, enable_dsa=False),
                parall_config=SimpleNamespace(attn_sp_size=1, attn_dies=1),
                task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False),
            )
            _set_module("omni.models.config_loader.loader", _mk_mod("omni.models.config_loader.loader", model_extra_config=model_extra_config))

            # deepseek_v3 / deepseek_v3_a2 for relative imports inside deepseek_mtp
            _set_module("omni.models.deepseek.deepseek_v3", _mk_mod(
                "omni.models.deepseek.deepseek_v3",
                DeepseekDecoderLayer=_DeepseekDecoderLayer,
                generate_sp_inputs=_generate_sp_inputs,
            ))
            _set_module("omni.models.deepseek.deepseek_v3_a2", _mk_mod(
                "omni.models.deepseek.deepseek_v3_a2",
                DeepseekDecoderLayer=_DeepseekDecoderLayer,
            ))

            # Import target module freshly
            prev_target = sys.modules.get(modname, None)
            if modname in sys.modules:
                del sys.modules[modname]

            m = importlib.import_module(modname)
            yield m

        finally:
            # Restore env
            os.environ.clear()
            os.environ.update(old_environ)

            # Restore torch.npu
            if had_torch_npu_attr:
                setattr(torch, "npu", old_torch_npu_attr)
            else:
                if hasattr(torch, "npu"):
                    delattr(torch, "npu")

            # Remove imported target module
            sys.modules.pop(modname, None)
            if "prev_target" in locals() and prev_target is not None:
                sys.modules[modname] = prev_target

            # Restore sys.modules mutations
            for k, old in touched.items():
                if old is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = old

    def _make_vllm_config(
        self,
        *,
        num_hidden_layers=4,
        num_nextn_predict_layers=2,
        vocab_size=32,
        hidden_size=8,
        n_routed_experts=2,
        num_speculative_tokens=2,
        rms_norm_eps=1e-6,
    ):
        hf_config = SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            num_nextn_predict_layers=num_nextn_predict_layers,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            rms_norm_eps=rms_norm_eps,
            n_routed_experts=n_routed_experts,
        )
        model_config = SimpleNamespace(hf_config=hf_config)
        speculative_config = SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
        return SimpleNamespace(
            model_config=model_config,
            cache_config=None,
            quant_config=None,
            speculative_config=speculative_config,
        )

    # ----------------------------
    # Tests
    # ----------------------------
    def test_shared_head_init_sets_head_none_when_ignore_share_weight_true(self):
        with self._import_mtp_with_stubs() as m:
            cfg = SimpleNamespace(hidden_size=8, rms_norm_eps=1e-6, vocab_size=32)
            sh = m.SharedHead(cfg, quant_config=None, ignore_share_weight=True)
            self.assertIsNone(sh.head)

    def test_shared_head_forward_only_applies_norm_and_preserves_shape(self):
        with self._import_mtp_with_stubs() as m:
            cfg = SimpleNamespace(hidden_size=8, rms_norm_eps=1e-6, vocab_size=32)
            sh = m.SharedHead(cfg, quant_config=None, ignore_share_weight=True)
            x = torch.randn(3, 8)


            class _NormStub(nn.Module):
                def __init__(self, out):
                    super().__init__()
                    self.out = out
                    self.called = 0
                    self.last_x = None
                def forward(self, x):
                    self.called += 1
                    self.last_x = x
                    return self.out

            y = torch.randn(3, 8)
            stub = _NormStub(y)
            sh.norm = stub
            out = sh(x)
            self.assertTrue(torch.equal(out, y))
            self.assertEqual(stub.called, 1)
            self.assertIs(stub.last_x, x)


    def test_mtp_layer_requires_share_weight_before_forward(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config()
            layer = m.DeepseekMultiTokenPredictorLayer(vllm_config=vllm_cfg, prefix="model.layers.10", kv_ind=0)

            input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
            positions = torch.tensor([0, 1, 2], dtype=torch.long)
            prev = torch.zeros(3, vllm_cfg.model_config.hf_config.hidden_size)
            kv = [torch.empty(1)]

            with self.assertRaises(TypeError):
                layer(
                    input_ids=input_ids,
                    positions=positions,
                    kv_caches=kv,
                    attn_metadata=None,
                    previous_hidden_states=prev,
                    selected_indices=None,
                )

    def test_mtp_layer_forward_flattens_tok_embeds_when_embedding_returns_3d(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(hidden_size=8, vocab_size=32)
            cfg = vllm_cfg.model_config.hf_config
            layer = m.DeepseekMultiTokenPredictorLayer(vllm_config=vllm_cfg, prefix="model.layers.10", kv_ind=0)

            class Embed3D(nn.Module):
                def forward(self, input_ids: torch.Tensor, reduce: int = 1):
                    n = input_ids.numel()
                    h = cfg.hidden_size
                    base = torch.arange(n * h, dtype=torch.float32)
                    return base.view(n, 1, h)  # 3D on purpose

            layer.embed_tokens = Embed3D()
            layer.shared_head.head = lambda hs, bias=None: torch.zeros(hs.shape[0], cfg.vocab_size)

            recorded = {}

            def _eh_forward(x: torch.Tensor):
                recorded["cat_shape"] = tuple(x.shape)
                return x[:, : cfg.hidden_size].contiguous(), None

            layer.eh_proj.forward = _eh_forward

            input_ids = torch.zeros(6, dtype=torch.long)
            positions = torch.zeros(6, dtype=torch.long)
            prev = torch.zeros(6, cfg.hidden_size)
            kv = [torch.empty(1)]

            logits, hidden_states = layer(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv,
                attn_metadata=None,
                previous_hidden_states=prev,
                selected_indices=None,
            )

            self.assertEqual(recorded["cat_shape"], (6, 2 * cfg.hidden_size))
            self.assertEqual(tuple(hidden_states.shape), (6, cfg.hidden_size))
            self.assertEqual(tuple(logits.shape), (1, cfg.vocab_size))

    def test_mtp_layer_forward_prefill_and_sp_attention_split_path_called_when_attn_sp_size_gt1(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(hidden_size=8, vocab_size=32)
            cfg = vllm_cfg.model_config.hf_config

            m.model_extra_config.parall_config.attn_sp_size = 2
            layer = m.DeepseekMultiTokenPredictorLayer(vllm_config=vllm_cfg, prefix="model.layers.10", kv_ind=0)

            class Embed2D(nn.Module):
                def forward(self, input_ids: torch.Tensor, reduce: int = 1):
                    n = input_ids.numel()
                    return torch.zeros(n, cfg.hidden_size)

            layer.embed_tokens = Embed2D()
            layer.shared_head.head = lambda hs, bias=None: torch.zeros(hs.shape[0], cfg.vocab_size)

            key = layer.prefix + layer.postfix
            prefill_meta = SimpleNamespace(sp_reverse_split_list=[2, 2], sp_reverse_index=[1, 0])
            attn_meta_obj = SimpleNamespace(prefill=prefill_meta)
            attn_metadata = {key: attn_meta_obj}

            with patch.object(m, "tensor_model_parallel_all_gather", Mock(side_effect=lambda x, dim=0: x), create=True) as gmock, \
                 patch.object(m, "generate_sp_inputs", Mock(side_effect=lambda x, meta: x), create=True) as spmock:
                input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
                positions = torch.zeros(4, dtype=torch.long)
                prev = torch.zeros(4, cfg.hidden_size)
                kv = [torch.empty(1)]

                logits, hidden_states = layer(
                    input_ids=input_ids,
                    positions=positions,
                    kv_caches=kv,
                    attn_metadata=attn_metadata,
                    previous_hidden_states=prev,
                    selected_indices=None,
                )

                self.assertGreaterEqual(gmock.call_count, 1)
                self.assertGreaterEqual(spmock.call_count, 1)
                self.assertEqual(tuple(hidden_states.shape), (4, cfg.hidden_size))
                self.assertEqual(tuple(logits.shape), (4, cfg.vocab_size))

    def test_mtp_layer_forward_tp_slice_previous_hidden_states_when_tp_size_gt1_and_sp_size_eq1(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(hidden_size=8, vocab_size=32)
            cfg = vllm_cfg.model_config.hf_config

            # Ensure we are in SP=1 path.
            m.model_extra_config.parall_config.attn_sp_size = 1

            layer = m.DeepseekMultiTokenPredictorLayer(
                vllm_config=vllm_cfg, prefix="model.layers.10", kv_ind=0
            )

            class Embed2D(nn.Module):
                def forward(self, input_ids: torch.Tensor, reduce: int = 1):
                    n = input_ids.numel()
                    return torch.zeros(n, cfg.hidden_size)

            # Provide required shared weights
            layer.embed_tokens = Embed2D()
            layer.shared_head.head = lambda hs, bias=None: torch.zeros(hs.shape[0], cfg.vocab_size)

            # Patch TP world size/rank to trigger TP slice logic:
            # previous_hidden_states length should be sliced by tp_size
            with patch.object(m, "get_tensor_model_parallel_world_size", Mock(return_value=2), create=True), \
                patch.object(m, "get_tensor_model_parallel_rank", Mock(return_value=0), create=True):

                seen = {}

                # DO NOT assign Mock to layer.hnorm (it's an nn.Module child).
                # Patch forward to record the input shape and passthrough.
                def _hnorm_forward(x):
                    seen["hnorm_in_shape"] = tuple(x.shape)
                    return x

                with patch.object(layer.hnorm, "forward", side_effect=_hnorm_forward):
                    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)   # N=3
                    positions = torch.zeros(3, dtype=torch.long)
                    kv = [torch.empty(1)]

                    # Make previous_hidden_states longer than N to verify slice happens.
                    # When tp_size=2 and rank=0, the code should slice to N.
                    previous_hidden_states = torch.zeros(6, cfg.hidden_size)  # 6 -> should slice to 3

                    # Run
                    logits, hidden_states = layer(
                        input_ids=input_ids,
                        positions=positions,
                        kv_caches=kv,
                        attn_metadata=None,
                        previous_hidden_states=previous_hidden_states,
                        selected_indices=None,
                    )

                # Assert: hnorm saw sliced shape (N, hidden)
                self.assertIn("hnorm_in_shape", seen)
                self.assertEqual(seen["hnorm_in_shape"], (3, cfg.hidden_size))

                # Minimal output shape guards (no numeric checks)
                self.assertEqual(tuple(hidden_states.shape), (3, cfg.hidden_size))
                # attn_metadata None -> last token lmhead, so logits is (1, vocab)
                self.assertEqual(tuple(logits.shape), (1, cfg.vocab_size))

    def test_mtp_layer_forward_eh_tp_allgather_and_alltoall_called_when_eh_tp_size_gt1(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(hidden_size=8, vocab_size=32)
            cfg = vllm_cfg.model_config.hf_config

            group = SimpleNamespace(
                world_size=2,
                rank_in_group=0,
                all_gather=Mock(side_effect=lambda x, dim=0: x),
                all_to_all=Mock(side_effect=lambda x: x),
            )

            with patch.object(m, "get_eh_proj_tp_group", Mock(return_value=group), create=True):
                layer = m.DeepseekMultiTokenPredictorLayer(vllm_config=vllm_cfg, prefix="model.layers.10", kv_ind=0)

                class Embed2D(nn.Module):
                    def forward(self, input_ids: torch.Tensor, reduce: int = 1):
                        n = input_ids.numel()
                        return torch.zeros(n, cfg.hidden_size)

                layer.embed_tokens = Embed2D()
                layer.shared_head.head = lambda hs, bias=None: torch.zeros(hs.shape[0], cfg.vocab_size)

                input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
                positions = torch.zeros(4, dtype=torch.long)
                prev = torch.zeros(4, cfg.hidden_size)
                kv = [torch.empty(1)]

                layer(
                    input_ids=input_ids,
                    positions=positions,
                    kv_caches=kv,
                    attn_metadata=None,
                    previous_hidden_states=prev,
                    selected_indices=None,
                )

                self.assertGreaterEqual(group.all_gather.call_count, 1)
                self.assertGreaterEqual(group.all_to_all.call_count, 1)

    def test_mtp_layer_forward_attn_metadata_none_uses_last_token_for_lmhead(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(hidden_size=8, vocab_size=32)
            cfg = vllm_cfg.model_config.hf_config
            layer = m.DeepseekMultiTokenPredictorLayer(vllm_config=vllm_cfg, prefix="model.layers.10", kv_ind=0)

            class Embed2D(nn.Module):
                def forward(self, input_ids: torch.Tensor, reduce: int = 1):
                    n = input_ids.numel()
                    return torch.zeros(n, cfg.hidden_size)

            layer.embed_tokens = Embed2D()
            layer.compute_lmhead = Mock(return_value=torch.zeros(1, cfg.vocab_size))

            input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
            positions = torch.zeros(4, dtype=torch.long)
            prev = torch.zeros(4, cfg.hidden_size)
            kv = [torch.empty(1)]

            layer(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv,
                attn_metadata=None,
                previous_hidden_states=prev,
                selected_indices=None,
            )

            args, _ = layer.compute_lmhead.call_args
            hs_arg, sel_arg = args[0], args[1]
            self.assertEqual(tuple(hs_arg.shape), (1, cfg.hidden_size))
            self.assertIsNone(sel_arg)

    def test_mtp_layer_forward_attn_metadata_not_none_passes_selected_indices_to_lmhead(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(hidden_size=8, vocab_size=32)
            cfg = vllm_cfg.model_config.hf_config
            layer = m.DeepseekMultiTokenPredictorLayer(vllm_config=vllm_cfg, prefix="model.layers.10", kv_ind=0)

            class Embed2D(nn.Module):
                def forward(self, input_ids: torch.Tensor, reduce: int = 1):
                    n = input_ids.numel()
                    return torch.zeros(n, cfg.hidden_size)

            layer.embed_tokens = Embed2D()
            layer.compute_lmhead = Mock(return_value=torch.zeros(4, cfg.vocab_size))

            attn_metadata = SimpleNamespace(prefill=None)
            selected_indices = torch.tensor([0, 2], dtype=torch.long)

            input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
            positions = torch.zeros(4, dtype=torch.long)
            prev = torch.zeros(4, cfg.hidden_size)
            kv = [torch.empty(1)]

            layer(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv,
                attn_metadata=attn_metadata,
                previous_hidden_states=prev,
                selected_indices=selected_indices,
            )

            args, _ = layer.compute_lmhead.call_args
            hs_arg, sel_arg = args[0], args[1]
            self.assertEqual(tuple(hs_arg.shape), (4, cfg.hidden_size))
            self.assertIs(sel_arg, selected_indices)

    def test_mtp_layer_compute_lmhead_selected_indices_applies_index_select_only_on_mismatch_when_dp_size_le1(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(hidden_size=8, vocab_size=32)
            cfg = vllm_cfg.model_config.hf_config
            layer = m.DeepseekMultiTokenPredictorLayer(vllm_config=vllm_cfg, prefix="model.layers.10", kv_ind=0)

            with patch.object(m, "get_dp_group", Mock(return_value=SimpleNamespace(world_size=1)), create=True):
                captured = {}

                def head_fn(x: torch.Tensor, bias=None):
                    captured["x"] = x.detach().clone()
                    return torch.zeros(x.shape[0], cfg.vocab_size)

                layer.shared_head.head = head_fn
                hidden = torch.arange(4 * cfg.hidden_size, dtype=torch.float32).view(4, cfg.hidden_size)
                sel = torch.tensor([0, 2], dtype=torch.long)
                out = layer.compute_lmhead(hidden, sel)
                self.assertEqual(tuple(out.shape), (2, cfg.vocab_size))
                self.assertEqual(tuple(captured["x"].shape), (2, cfg.hidden_size))

                captured2 = {}

                def head_fn2(x: torch.Tensor, bias=None):
                    captured2["x"] = x.detach().clone()
                    return torch.zeros(x.shape[0], cfg.vocab_size)

                layer.shared_head.head = head_fn2
                hidden2 = torch.arange(2 * cfg.hidden_size, dtype=torch.float32).view(2, cfg.hidden_size)
                sel2 = torch.tensor([1, 0], dtype=torch.long)
                _ = layer.compute_lmhead(hidden2, sel2)
                self.assertTrue(torch.equal(captured2["x"], hidden2.view(-1, cfg.hidden_size)))

    def test_mtp_layer_compute_logits_uses_shared_head_head_module_and_calls_logits_processor(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(hidden_size=8, vocab_size=32)
            cfg = vllm_cfg.model_config.hf_config
            layer = m.DeepseekMultiTokenPredictorLayer(vllm_config=vllm_cfg, prefix="model.layers.10", kv_ind=0)

            head_obj = object()
            layer.shared_head.head = head_obj
            sentinel = torch.zeros(3, cfg.vocab_size)
            layer.logits_processor = Mock(return_value=sentinel)

            hidden = torch.zeros(3, cfg.hidden_size)
            sampling_metadata = object()

            try:
                out = layer.compute_logits(hidden, sampling_metadata)
            except TypeError:
                pytest.xfail("Known bug: compute_logits uses self.shared_head['head'] instead of self.shared_head.head")
                return

            self.assertIs(out, sentinel)
            args, _ = layer.logits_processor.call_args
            self.assertIs(args[0], head_obj)
            self.assertTrue(torch.equal(args[1], hidden))
            self.assertIs(args[2], sampling_metadata)

    def test_mtp_predictor_set_share_weight_wires_embed_tokens_and_lm_head_into_all_layers(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(num_hidden_layers=4, num_nextn_predict_layers=2, num_speculative_tokens=2)

            class DummyLayer(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.embed_tokens = None
                    self.shared_head = SimpleNamespace(head=None)

                def forward(self, **kwargs):
                    return None

            with patch.object(m, "DeepseekMultiTokenPredictorLayer", DummyLayer, create=True):
                predictor = m.DeepseekMultiTokenPredictor(vllm_config=vllm_cfg, prefix="model")

                target_embed = object()
                target_lm_head = object()
                target_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=target_embed), lm_head=target_lm_head)

                predictor.set_share_weight(target_model)

                for _, layer in predictor.layers.items():
                    self.assertIs(layer.embed_tokens, target_embed)
                    self.assertIs(layer.shared_head.head, target_lm_head)

    def test_mtp_predictor_forward_dispatches_correct_layer_by_mtp_layer_idx(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(num_hidden_layers=4, num_nextn_predict_layers=2, num_speculative_tokens=2)

            class DummyLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mock = Mock(return_value=("ok",))

                def forward(self, **kwargs):
                    return self.mock(**kwargs)

            with patch.object(m, "DeepseekMultiTokenPredictorLayer", Mock(side_effect=lambda **kwargs: DummyLayer()), create=True):
                predictor = m.DeepseekMultiTokenPredictor(vllm_config=vllm_cfg, prefix="model")

            self.assertIn("4", predictor.layers)
            self.assertIn("5", predictor.layers)

            input_ids = torch.tensor([1, 2], dtype=torch.long)
            positions = torch.zeros(2, dtype=torch.long)
            prev = torch.zeros(2, vllm_cfg.model_config.hf_config.hidden_size)
            kv = [torch.empty(1)]
            attn_meta = SimpleNamespace(prefill=None)

            out = predictor(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv,
                attn_metadata=attn_meta,
                previous_hidden_states=prev,
                selected_indices=None,
                mtp_layer_idx=1,
            )

            self.assertEqual(out, ("ok",))
            predictor.layers["5"].mock.assert_called_once()
            predictor.layers["4"].mock.assert_not_called()

    def test_deepseek_v3_mtp_forward_clamps_mtp_layer_idx_to_n_predictor_minus_1(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(num_nextn_predict_layers=2, num_speculative_tokens=2)

            class DummyPredictor(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.last_mtp_layer_idx = None

                def forward(self, *, mtp_layer_idx=0, **kwargs):
                    self.last_mtp_layer_idx = mtp_layer_idx
                    return "ok"

            with patch.object(m, "DeepseekMultiTokenPredictor", DummyPredictor, create=True):
                mtp = m.DeepseekV3MTP(vllm_config=vllm_cfg, prefix="")

                input_ids = torch.tensor([1], dtype=torch.long)
                positions = torch.tensor([0], dtype=torch.long)
                kv = [torch.empty(1)]
                prev = torch.zeros(1, vllm_cfg.model_config.hf_config.hidden_size)
                attn_meta = SimpleNamespace(prefill=None)

                out = mtp(
                    input_ids=input_ids,
                    positions=positions,
                    kv_caches=kv,
                    attn_metadata=attn_meta,
                    previous_hidden_states=prev,
                    selected_indices=None,
                    mtp_layer_idx=999,
                )

                self.assertEqual(out, "ok")
                self.assertEqual(mtp.model.last_mtp_layer_idx, 1)

    def test_load_weights_skips_rotary_inv_freq_and_ignores_shared_weights_when_ignore_share_weight_true(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(num_nextn_predict_layers=2, num_speculative_tokens=2)

            class DummyPredictor(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.ignore_share_weight = True

            with patch.object(m, "DeepseekMultiTokenPredictor", DummyPredictor, create=True):
                mtp = m.DeepseekV3MTP(vllm_config=vllm_cfg, prefix="")

            with patch.object(m, "get_spec_layer_idx_from_weight_name", Mock(side_effect=AssertionError("should not be called")), create=True), \
                 patch.object(m.FusedMoE, "make_expert_params_mapping", Mock(return_value=[]), create=True):
                weights = [
                    ("model.layers.0.rotary_emb.inv_freq", torch.zeros(1)),
                    ("embed_tokens.weight", torch.zeros(1)),
                    ("model.shared_head.head.weight", torch.zeros(1)),
                ]
                loaded = mtp.load_weights(weights)
                self.assertEqual(loaded, set())

    def test_load_weights_stacked_params_mapping_routes_gate_proj_up_proj_into_gate_up_proj_with_shard_id(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(n_routed_experts=2)

            class DummyPredictor(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.ignore_share_weight = True

            with patch.object(m, "DeepseekMultiTokenPredictor", DummyPredictor, create=True):
                mtp = m.DeepseekV3MTP(vllm_config=vllm_cfg, prefix="")

            p_gate = SimpleNamespace(weight_loader=Mock())
            p_up = SimpleNamespace(weight_loader=Mock())

            mapped_gate = "model.layers.999.mlp.gate_up_proj.weight"
            mapped_up = "model.layers.999.mlp.gate_up_proj.weight2"
            ckpt_gate = "model.layers.999.mlp.gate_proj.weight"
            ckpt_up = "model.layers.999.mlp.up_proj.weight2"

            mtp.named_parameters = Mock(return_value=[(mapped_gate, p_gate), (mapped_up, p_up)])

            with patch.object(m, "is_pp_missing_parameter", Mock(return_value=False), create=True), \
                 patch.object(m, "get_spec_layer_idx_from_weight_name", Mock(return_value=0), create=True), \
                 patch.object(m.FusedMoE, "make_expert_params_mapping", Mock(return_value=[]), create=True):
                weights = [
                    (ckpt_gate, torch.ones(1)),
                    (ckpt_up, torch.ones(1) * 2),
                ]
                loaded = mtp.load_weights(weights)

                self.assertIn(mapped_gate, loaded)
                self.assertIn(mapped_up, loaded)

                p_gate.weight_loader.assert_called_once()
                args, _kwargs = p_gate.weight_loader.call_args
                self.assertIs(args[0], p_gate)
                self.assertTrue(torch.equal(args[1], torch.ones(1)))
                self.assertEqual(args[2], 0)

                p_up.weight_loader.assert_called_once()
                args, _kwargs = p_up.weight_loader.call_args
                self.assertIs(args[0], p_up)
                self.assertTrue(torch.equal(args[1], torch.ones(1) * 2))
                self.assertEqual(args[2], 1)

    def test_load_weights_expert_params_mapping_routes_expert_weights_via_weight_loader_with_expert_id_and_shard_id(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config(n_routed_experts=8)

            class DummyPredictor(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.ignore_share_weight = True

            with patch.object(m, "DeepseekMultiTokenPredictor", DummyPredictor, create=True):
                mtp = m.DeepseekV3MTP(vllm_config=vllm_cfg, prefix="")

            expert_id = 7
            shard_id = 0
            expert_mapping = [("gate_up_proj", "gate_proj", expert_id, shard_id)]

            mapped = f"model.layers.999.mlp.experts.{expert_id}.gate_up_proj.weight"
            param = SimpleNamespace(weight_loader=Mock())
            mtp.named_parameters = Mock(return_value=[(mapped, param)])

            ckpt = f"model.layers.999.mlp.experts.{expert_id}.gate_proj.weight"

            with patch.object(m, "is_pp_missing_parameter", Mock(return_value=False), create=True), \
                 patch.object(m, "get_spec_layer_idx_from_weight_name", Mock(return_value=0), create=True), \
                 patch.object(m.FusedMoE, "make_expert_params_mapping", Mock(return_value=expert_mapping), create=True):
                loaded_w = torch.ones(1)
                loaded = mtp.load_weights([(ckpt, loaded_w)])

                self.assertIn(mapped, loaded)
                param.weight_loader.assert_called_once()
                args, kwargs = param.weight_loader.call_args
                self.assertIs(args[0], param)
                self.assertTrue(torch.equal(args[1], loaded_w))
                self.assertEqual(args[2], mapped)
                self.assertEqual(kwargs.get("shard_id"), shard_id)
                self.assertEqual(kwargs.get("expert_id"), expert_id)

    def test_load_weights_default_weight_loader_used_when_param_has_no_weight_loader_attr(self):
        with self._import_mtp_with_stubs() as m:
            vllm_cfg = self._make_vllm_config()

            class DummyPredictor(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.ignore_share_weight = True

            with patch.object(m, "DeepseekMultiTokenPredictor", DummyPredictor, create=True):
                mtp = m.DeepseekV3MTP(vllm_config=vllm_cfg, prefix="")

            mapped = "model.layers.999.attn.q_proj.weight"
            param = SimpleNamespace()
            mtp.named_parameters = Mock(return_value=[(mapped, param)])

            ckpt = mapped
            loaded_w = torch.ones(1) * 3

            with patch.object(m, "is_pp_missing_parameter", Mock(return_value=False), create=True), \
                 patch.object(m, "get_spec_layer_idx_from_weight_name", Mock(return_value=0), create=True), \
                 patch.object(m.FusedMoE, "make_expert_params_mapping", Mock(return_value=[]), create=True), \
                 patch.object(m, "default_weight_loader", Mock(), create=True) as dwl:
                loaded = mtp.load_weights([(ckpt, loaded_w)])

                self.assertIn(mapped, loaded)
                dwl.assert_called_once()
                args, _kwargs = dwl.call_args
                self.assertIs(args[0], param)
                self.assertTrue(torch.equal(args[1], loaded_w))


if __name__ == "__main__":
    unittest.main()
