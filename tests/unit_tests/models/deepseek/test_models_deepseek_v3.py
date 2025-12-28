import sys
import types
import contextlib
import unittest
import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn


# -----------------------------
# Helpers: ultra-isolated import
# -----------------------------
class _DummyStream:
    def wait_stream(self, other):
        return None


@contextlib.contextmanager
def _maybe_patch_torch_npu_attr():
    """Ensure torch.npu exists during import/runtime, and restore afterwards."""
    existed = hasattr(torch, "npu")
    old = getattr(torch, "npu", None)

    if not existed:

        @contextlib.contextmanager
        def _stream_ctx(_s):
            yield

        torch.npu = SimpleNamespace(  # type: ignore[attr-defined]
            Stream=_DummyStream,
            current_stream=lambda: _DummyStream(),
            stream=_stream_ctx,
        )
    try:
        yield
    finally:
        if not existed:
            try:
                delattr(torch, "npu")
            except Exception:
                pass
        else:
            try:
                torch.npu = old  # type: ignore[attr-defined]
            except Exception:
                pass


@contextlib.contextmanager
def _patch_torch_logging_set_logs_noop():
    """deepseek_v3 calls torch._logging.set_logs(...). Make it no-op temporarily."""
    had_logging = hasattr(torch, "_logging")
    old_logging = getattr(torch, "_logging", None)

    class _DummyLogging:
        @staticmethod
        def set_logs(**kwargs):
            return None

    created = False
    orig_set_logs = None
    if not had_logging:
        torch._logging = _DummyLogging()  # type: ignore[attr-defined]
        created = True
    else:
        try:
            if hasattr(torch._logging, "set_logs"):  # type: ignore[attr-defined]
                orig_set_logs = torch._logging.set_logs  # type: ignore[attr-defined]
                torch._logging.set_logs = lambda **kwargs: None  # type: ignore[attr-defined]
        except Exception:
            orig_set_logs = None

    try:
        yield
    finally:
        if created:
            try:
                delattr(torch, "_logging")
            except Exception:
                pass
        else:
            # restore original object as best-effort
            if old_logging is not None:
                try:
                    torch._logging = old_logging  # type: ignore[attr-defined]
                except Exception:
                    pass
            else:
                # restore set_logs if we patched just the method
                if orig_set_logs is not None:
                    try:
                        torch._logging.set_logs = orig_set_logs  # type: ignore[attr-defined]
                    except Exception:
                        pass


class _AcceptAllNullContext:
    """A no-op context manager that accepts arbitrary args/kwargs."""
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _build_stub_modules_for_import():
    """
    Provide minimal stubs for torch_npu / torchair / vllm / omni.layers* so the module
    can be imported in CPU UT envs. Installed via patch.dict(sys.modules, ...),
    and will be restored after each test.
    """
    stubs = {}

    # --- torch_npu stub ---
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.npu_prefetch = lambda *args, **kwargs: None
    stubs["torch_npu"] = torch_npu

    # --- torchair stub ---
    torchair = types.ModuleType("torchair")

    class _Scope:
        @staticmethod
        @contextlib.contextmanager
        def super_kernel(*args, **kwargs):
            yield

    torchair.scope = _Scope()
    stubs["torchair"] = torchair

    # --- vllm stubs ---
    vllm = types.ModuleType("vllm")

    # IMPORTANT: omni/models/__init__.py imports this from vllm
    class ModelRegistry:
        @staticmethod
        def register_model(*args, **kwargs):
            return None

    vllm.ModelRegistry = ModelRegistry
    stubs["vllm"] = vllm

    vllm_platforms = types.ModuleType("vllm.platforms")
    vllm_platforms.current_platform = SimpleNamespace(device_type="cpu")
    stubs["vllm.platforms"] = vllm_platforms

    vllm_config = types.ModuleType("vllm.config")

    class CacheConfig:
        pass

    class QuantizationConfig:
        pass

    class VllmConfig:
        pass

    vllm_config.CacheConfig = CacheConfig
    vllm_config.QuantizationConfig = QuantizationConfig
    vllm_config.VllmConfig = VllmConfig
    stubs["vllm.config"] = vllm_config

    vllm_comp_dec = types.ModuleType("vllm.compilation.decorators")
    vllm_comp_dec.support_torch_compile = lambda obj: obj
    stubs["vllm.compilation.decorators"] = vllm_comp_dec

    vllm_attention = types.ModuleType("vllm.attention")

    class AttentionMetadata:
        pass

    vllm_attention.AttentionMetadata = AttentionMetadata
    stubs["vllm.attention"] = vllm_attention

    vllm_sequence = types.ModuleType("vllm.sequence")

    class IntermediateTensors(dict):
        pass

    vllm_sequence.IntermediateTensors = IntermediateTensors
    stubs["vllm.sequence"] = vllm_sequence

    vllm_sampling = types.ModuleType("vllm.model_executor.sampling_metadata")

    class SamplingMetadata:
        pass

    vllm_sampling.SamplingMetadata = SamplingMetadata
    stubs["vllm.model_executor.sampling_metadata"] = vllm_sampling

    vllm_sampler = types.ModuleType("vllm.model_executor.layers.sampler")

    class SamplerOutput:
        pass

    class Sampler:
        def __call__(self, logits, sampling_metadata):
            return SamplerOutput()

    vllm_sampler.Sampler = Sampler
    vllm_sampler.SamplerOutput = SamplerOutput
    stubs["vllm.model_executor.layers.sampler"] = vllm_sampler

    vllm_dist = types.ModuleType("vllm.distributed")

    class _Group:
        def __init__(self, world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True):
            self.world_size = world_size
            self.rank_in_group = rank_in_group
            self.is_first_rank = is_first_rank
            self.is_last_rank = is_last_rank

    _default_ep = _Group(world_size=1, rank_in_group=0)
    _default_dp = _Group(world_size=1, rank_in_group=0)
    _default_pp = _Group(world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True)

    vllm_dist.get_ep_group = lambda: _default_ep
    vllm_dist.get_dp_group = lambda: _default_dp
    vllm_dist.get_pp_group = lambda: _default_pp
    vllm_dist.get_tensor_model_parallel_world_size = lambda: 1
    vllm_dist.get_tensor_model_parallel_rank = lambda: 0
    vllm_dist.tensor_model_parallel_all_gather = lambda x, dim=0: x
    stubs["vllm.distributed"] = vllm_dist

    vllm_logits_proc = types.ModuleType("vllm.model_executor.layers.logits_processor")

    class LogitsProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, lm_head, hidden_states, sampling_metadata):
            return lm_head(hidden_states, None)

    vllm_logits_proc.LogitsProcessor = LogitsProcessor
    stubs["vllm.model_executor.layers.logits_processor"] = vllm_logits_proc

    vllm_if = types.ModuleType("vllm.model_executor.models.interfaces")

    class SupportsPP:
        pass

    vllm_if.SupportsPP = SupportsPP
    stubs["vllm.model_executor.models.interfaces"] = vllm_if

    vllm_utils = types.ModuleType("vllm.model_executor.models.utils")

    class PPMissingLayer(nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError("PPMissingLayer called in unit-test stub.")

    def is_pp_missing_parameter(name, module):
        return False

    def make_layers(num_layers, layer_factory, prefix=""):
        return 0, 0, []

    def make_empty_intermediate_tensors_factory(keys, hidden_size):
        def _factory(batch_size, dtype, device):
            return IntermediateTensors({k: torch.zeros((batch_size, hidden_size), dtype=dtype, device=device) for k in keys})
        return _factory

    vllm_utils.PPMissingLayer = PPMissingLayer
    vllm_utils.is_pp_missing_parameter = is_pp_missing_parameter
    vllm_utils.make_layers = make_layers
    vllm_utils.make_empty_intermediate_tensors_factory = make_empty_intermediate_tensors_factory
    stubs["vllm.model_executor.models.utils"] = vllm_utils

    vllm_weight_utils = types.ModuleType("vllm.model_executor.model_loader.weight_utils")
    vllm_weight_utils.default_weight_loader = lambda param, weight: None
    stubs["vllm.model_executor.model_loader.weight_utils"] = vllm_weight_utils

    # --- transformers stub only if transformers missing ---
    transformers = types.ModuleType("transformers")

    class PretrainedConfig:
        pass

    transformers.PretrainedConfig = PretrainedConfig
    stubs["transformers"] = transformers

    # --- omni submodule stubs ---
    def _mk_mod(name):
        return types.ModuleType(name)

    omni_layers_utils = _mk_mod("omni.layers.utils")

    class ConditionalTNGScope:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    omni_layers_utils.ConditionalTNGScope = ConditionalTNGScope
    stubs["omni.layers.utils"] = omni_layers_utils

    omni_ln = _mk_mod("omni.layers.layernorm")

    class RMSNorm(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, residual=None, **kwargs):
            if residual is None:
                return x
            return x, residual

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    omni_ln.RMSNorm = RMSNorm
    stubs["omni.layers.layernorm"] = omni_ln

    omni_vpe = _mk_mod("omni.layers.vocab_parallel_embedding")

    class ParallelLMHead(nn.Module):
        def __init__(self, vocab_size, hidden_size, **kwargs):
            super().__init__()
            self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

        def forward(self, x, embedding_bias=None):
            return self.proj(x)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class VocabParallelEmbedding(nn.Module):
        def __init__(self, vocab_size, hidden_size, **kwargs):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, hidden_size)

        def forward(self, input_ids, reduce=1):
            return self.emb(input_ids)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    omni_vpe.ParallelLMHead = ParallelLMHead
    omni_vpe.VocabParallelEmbedding = VocabParallelEmbedding
    stubs["omni.layers.vocab_parallel_embedding"] = omni_vpe

    omni_linear = _mk_mod("omni.layers.linear")
    omni_linear.AscendMergedColumnParallelLinear = object
    omni_linear.AscendRowParallelLinear = object
    stubs["omni.layers.linear"] = omni_linear

    omni_act = _mk_mod("omni.layers.activation")
    omni_act.SiluAndMul = object
    stubs["omni.layers.activation"] = omni_act

    omni_fused_layer = _mk_mod("omni.layers.moe.fused_moe.layer")

    class FusedMoE:
        @staticmethod
        def make_expert_params_mapping(*args, **kwargs):
            return []

    omni_fused_layer.FusedMoE = FusedMoE
    stubs["omni.layers.moe.fused_moe.layer"] = omni_fused_layer

    omni_fused_moe = _mk_mod("omni.layers.moe.fused_moe.fused_moe")
    omni_fused_moe.set_num_speculative_tokens = lambda *a, **k: None
    stubs["omni.layers.moe.fused_moe.fused_moe"] = omni_fused_moe

    omni_deepseek_moe = _mk_mod("omni.layers.moe.deepseek_moe")

    class DeepseekMoE(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, hidden_states, residual, *args, **kwargs):
            return hidden_states, residual

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class ParallelDeepseekMLP(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.gate_up_proj = SimpleNamespace(weight=torch.empty(1))

        def forward(self, hidden_states, residual, *args, **kwargs):
            return hidden_states, residual

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    omni_deepseek_moe.DeepseekMoE = DeepseekMoE
    omni_deepseek_moe.ParallelDeepseekMLP = ParallelDeepseekMLP
    stubs["omni.layers.moe.deepseek_moe"] = omni_deepseek_moe

    omni_mla = _mk_mod("omni.layers.attention.deepseek_mla")

    class DeepseekMLA(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.q_a_proj = SimpleNamespace(weight=torch.empty(1))
            self.kv_a_proj_with_mqa = SimpleNamespace(weight=torch.empty(1))
            self.q_b_proj = SimpleNamespace(weight=torch.empty(1))
            self.W_UK = torch.empty(1)

        def forward(self, positions, hidden_states, kv_cache, attn_metadata, **kwargs):
            return hidden_states

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    omni_mla.DeepseekMLA = DeepseekMLA
    stubs["omni.layers.attention.deepseek_mla"] = omni_mla

    omni_ps = _mk_mod("omni.adaptors.vllm.distributed.parallel_state")
    omni_ps.get_stream1_attn_group = lambda: None
    omni_ps.get_stream1_mlp_group = lambda: None
    omni_ps.get_stream1_moe_group = lambda: None
    omni_ps.get_mlp_tp_group = lambda: None

    class GroupCoordinator:
        pass

    omni_ps.GroupCoordinator = GroupCoordinator
    stubs["omni.adaptors.vllm.distributed.parallel_state"] = omni_ps

    omni_backend_mla = _mk_mod("omni.layers.attention.backend.mla")
    omni_backend_mla.group_request_list = lambda *a, **k: ([], [], [])
    stubs["omni.layers.attention.backend.mla"] = omni_backend_mla

    omni_cfg_loader = _mk_mod("omni.models.config_loader.loader")
    omni_cfg_loader.model_extra_config = SimpleNamespace(
        parall_config=SimpleNamespace(attn_sp_size=1, attn_dies=1),
        task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False, hardware_platform="A3"),
        operator_opt_config=SimpleNamespace(
            use_super_kernel=False,
            use_mlaprolog=False,
            enable_dsa=False,
            use_prefetch=False,
            dense_mlp_prefetch=0,
            lm_head_prefetch=0,
            enable_prefill_micro_batch=False,
            max_split_token_count_threshold=10**9,
            max_split_token_ratio_threshold=1.0,
            merge_qkv=False,
            prefill_moe_all_to_all=False,
        ),
    )
    stubs["omni.models.config_loader.loader"] = omni_cfg_loader

    return stubs


@contextlib.contextmanager
def _import_under_test(module_name: str):
    """
    Import module under test with minimal stubs and strict restoration to avoid polluting other UTs.
    Additionally removes newly-imported omni.* modules if they were not present before.
    """
    stubs = _build_stub_modules_for_import()

    # If transformers exists, don't override it with our stub.
    try:
        import transformers  # noqa: F401
        stubs.pop("transformers", None)
    except Exception:
        pass

    pre_keys = set(sys.modules.keys())
    with _maybe_patch_torch_npu_attr(), _patch_torch_logging_set_logs_noop():
        with patch.dict(sys.modules, stubs, clear=False):
            try:
                mod = importlib.import_module(module_name)
            except Exception:
                # cleanup partial imports introduced in this block
                post_keys = set(sys.modules.keys())
                new_keys = post_keys - pre_keys
                for k in sorted(new_keys, reverse=True):
                    if k == module_name or k.startswith("omni.models.deepseek") or k.startswith("omni.models"):
                        sys.modules.pop(k, None)
                raise

            try:
                yield mod
            finally:
                # remove module under test and new omni.* parents to avoid leaking stub-bound modules
                post_keys = set(sys.modules.keys())
                new_keys = post_keys - pre_keys
                # Always remove the module under test
                sys.modules.pop(module_name, None)
                # Remove newly created parent modules (only if they didn't exist before)
                for k in sorted(new_keys, reverse=True):
                    if k.startswith("omni.models.deepseek") or k.startswith("omni.models"):
                        sys.modules.pop(k, None)


# -----------------------------
# Test Doubles for forward paths
# -----------------------------
class _DummyAttnMeta:
    def __init__(self, prefill=None):
        self.prefill = prefill


class _DummyNorm:
    """Acts like RMSNorm: either returns x (no residual) or (x', residual')."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if len(args) == 1:
            return args[0]
        x = args[0]
        residual = args[1]
        return x, residual


class _DummyInputNorm:
    """residual=None path returns x; residual!=None returns (x, residual)."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if len(args) == 1:
            return args[0]
        x, residual = args[0], args[1]
        return x, residual


class _DummySelfAttn:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["hidden_states"]


class _DummyDenseMLP:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        hidden_states = args[0]
        residual = args[1]
        return hidden_states, residual


class _DummyMoEMLP:
    def __init__(self, return_tuple=False):
        self.calls = []
        self.return_tuple = return_tuple

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        hs = args[0]
        res = args[1]
        if self.return_tuple:
            return (hs, hs), res
        return hs, res


# -----------------------------
# The actual test suite
# -----------------------------
class TestModelsDeepseekV3(unittest.TestCase):
    _DEEPSEEK_V3_MOD = "omni.models.deepseek.deepseek_v3"

    # -----------------------------
    # 1) _get_pad_size / pad_inputs / generate_sp_inputs
    # -----------------------------
    def test_get_pad_size_scalar_and_tensor_returns_expected_padding(self):
        with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
            self.assertEqual(m._get_pad_size(8, 4), 0)
            self.assertEqual(m._get_pad_size(9, 4), 3)

            qlens = torch.tensor([3, 4, 5], dtype=torch.int64)
            out = m._get_pad_size(qlens, 4)
            self.assertTrue(torch.equal(out, torch.tensor([1, 0, 3], dtype=torch.int64)))

    def test_pad_inputs_pads_each_segment_to_multiple_of_sp_size(self):
        with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
            # Two segments: 3 and 5; sp_size=4 => pad lengths: 1 and 3 => total 12
            query_lens = torch.tensor([3, 5], dtype=torch.int64)
            inp = torch.ones((8, 2), dtype=torch.float32)
            out = m.pad_inputs(inp, query_lens, sp_size=4)
            self.assertEqual(out.shape, (12, 2))

            # segment1 padded row at index 3
            self.assertTrue(torch.all(out[3:4] == 0))
            # segment2 padded rows at indices 9..11
            self.assertTrue(torch.all(out[9:12] == 0))

    def test_generate_sp_inputs_with_prefill_metadata_applies_pad_split_and_zigzag(self):
        with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
            cfg = SimpleNamespace(
                parall_config=SimpleNamespace(attn_sp_size=2, attn_dies=1),
                task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False, hardware_platform="A3"),
                operator_opt_config=SimpleNamespace(
                    use_super_kernel=False, use_mlaprolog=False, enable_dsa=False, use_prefetch=False,
                    dense_mlp_prefetch=0, lm_head_prefetch=0, enable_prefill_micro_batch=False,
                    max_split_token_count_threshold=10**9, max_split_token_ratio_threshold=1.0,
                    merge_qkv=False, prefill_moe_all_to_all=False,
                ),
            )

            hs = torch.zeros((6, 4), dtype=torch.float32)
            hs[:, 0] = torch.arange(1, 7, dtype=torch.float32)

            prefill = SimpleNamespace(
                actual_query_lens=torch.tensor([3, 3], dtype=torch.int64),
                sp_split_list=[4, 4],
                sp_zigzag_index=[1, 0],  # swap chunks
            )
            attn_meta = _DummyAttnMeta(prefill=prefill)

            with patch.object(m, "model_extra_config", cfg):
                out = m.generate_sp_inputs(hs, attn_meta)

            self.assertEqual(out.shape[0], 8)
            nonzero = out[:, 0].tolist()
            self.assertEqual(nonzero, [4.0, 5.0, 6.0, 0.0, 1.0, 2.0, 3.0, 0.0])

    def test_generate_sp_inputs_without_metadata_selects_tp_rank_chunk(self):
        with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
            cfg = SimpleNamespace(
                parall_config=SimpleNamespace(attn_sp_size=2, attn_dies=1),
                task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False, hardware_platform="A3"),
                operator_opt_config=SimpleNamespace(
                    use_super_kernel=False, use_mlaprolog=False, enable_dsa=False, use_prefetch=False,
                    dense_mlp_prefetch=0, lm_head_prefetch=0, enable_prefill_micro_batch=False,
                    max_split_token_count_threshold=10**9, max_split_token_ratio_threshold=1.0,
                    merge_qkv=False, prefill_moe_all_to_all=False,
                ),
            )

            hs = torch.zeros((4, 3), dtype=torch.float32)
            hs[:, 0] = torch.tensor([10, 11, 12, 13], dtype=torch.float32)

            with patch.object(m, "model_extra_config", cfg), patch.object(m, "get_tensor_model_parallel_rank", return_value=1):
                out = m.generate_sp_inputs(hs, None)

            self.assertEqual(out.shape, (2, 3))
            self.assertTrue(torch.equal(out[:, 0], torch.tensor([12.0, 13.0])))

    # -----------------------------
    # 2) should_split_hidden_states
    # -----------------------------
    def test_should_split_hidden_states_handles_negative_ids_and_thresholds(self):
        with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
            model = m.DeepseekV3Model.__new__(m.DeepseekV3Model)
            nn.Module.__init__(model)

            ids = torch.tensor([[-1, -1, -1, 0, 1, 2]], dtype=torch.int64)

            self.assertFalse(model.should_split_hidden_states(ids, ratio_threshold=0.0, count_threshold=100))
            self.assertFalse(model.should_split_hidden_states(ids, ratio_threshold=0.9, count_threshold=0))

            # max_count=3, total=6 => ratio=0.5
            self.assertTrue(model.should_split_hidden_states(ids, ratio_threshold=0.5, count_threshold=999))
            self.assertTrue(model.should_split_hidden_states(ids, ratio_threshold=0.99, count_threshold=3))

    # -----------------------------
    # 3) DeepseekDecoderLayer.forward branches
    # -----------------------------
    def _make_decoder_layer_minimal(self, m, *, is_ffn_die=False, is_moe=False, mlp=None):
        layer = m.DeepseekDecoderLayer.__new__(m.DeepseekDecoderLayer)
        nn.Module.__init__(layer)
        layer.layer_name = "model.layers.0.self_attn.attn"
        layer.is_ffn_die = is_ffn_die
        layer.is_moe = is_moe
        layer.quant_symbol = False
        layer.input_layernorm = _DummyInputNorm()
        layer.post_attention_layernorm = _DummyNorm()
        layer.self_attn = _DummySelfAttn()
        layer.mlp = mlp if mlp is not None else _DummyDenseMLP()
        return layer

    # def test_decoder_layer_forward_dense_prefill_residual_none_path_runs(self):
    #     with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
    #         cfg = SimpleNamespace(
    #             parall_config=SimpleNamespace(attn_sp_size=1, attn_dies=1),
    #             task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False, hardware_platform="A3"),
    #             operator_opt_config=SimpleNamespace(
    #                 use_super_kernel=False, use_mlaprolog=False, enable_dsa=False,
    #                 use_prefetch=False, dense_mlp_prefetch=0, lm_head_prefetch=0,
    #                 enable_prefill_micro_batch=False, max_split_token_count_threshold=10**9,
    #                 max_split_token_ratio_threshold=1.0, merge_qkv=False, prefill_moe_all_to_all=False,
    #             ),
    #         )
    #         layer = self._make_decoder_layer_minimal(m, is_ffn_die=False, is_moe=False)
    #         hs = torch.randn(5, 4)
    #         positions = torch.zeros(5, dtype=torch.int64)
    #         attn = _DummyAttnMeta(prefill=object())

    #         with patch.object(m, "model_extra_config", cfg), patch.object(m, "ConditionalTNGScope", _AcceptAllNullContext):
    #             out_hs, out_res = layer.forward(positions, hs, kv_cache=None, attn_metadata=attn, residual=None)

    #         self.assertEqual(out_hs.shape, hs.shape)
    #         self.assertIsNotNone(out_res)
    #         self.assertTrue(any(len(args) == 1 for args, _ in layer.input_layernorm.calls))

    # def test_decoder_layer_forward_dense_prefill_residual_provided_path_runs(self):
    #     with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
    #         cfg = SimpleNamespace(
    #             parall_config=SimpleNamespace(attn_sp_size=1, attn_dies=1),
    #             task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False, hardware_platform="A3"),
    #             operator_opt_config=SimpleNamespace(
    #                 use_super_kernel=False, use_mlaprolog=False, enable_dsa=False,
    #                 use_prefetch=False, dense_mlp_prefetch=0, lm_head_prefetch=0,
    #                 enable_prefill_micro_batch=False, max_split_token_count_threshold=10**9,
    #                 max_split_token_ratio_threshold=1.0, merge_qkv=False, prefill_moe_all_to_all=False,
    #             ),
    #         )
    #         layer = self._make_decoder_layer_minimal(m, is_ffn_die=False, is_moe=False)
    #         hs = torch.randn(5, 4)
    #         residual = torch.randn(5, 4)
    #         positions = torch.zeros(5, dtype=torch.int64)
    #         attn = _DummyAttnMeta(prefill=object())

    #         with patch.object(m, "model_extra_config", cfg), patch.object(m, "ConditionalTNGScope", _AcceptAllNullContext):
    #             out_hs, out_res = layer.forward(positions, hs, kv_cache=None, attn_metadata=attn, residual=residual)

    #         self.assertEqual(out_hs.shape, hs.shape)
    #         self.assertEqual(out_res.shape, residual.shape)
    #         self.assertTrue(any(len(args) >= 2 for args, _ in layer.input_layernorm.calls))

    # def test_decoder_layer_forward_moe_tuple_output_is_merged_by_add(self):
    #     with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
    #         cfg = SimpleNamespace(
    #             parall_config=SimpleNamespace(attn_sp_size=1, attn_dies=1),
    #             task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False, hardware_platform="A3"),
    #             operator_opt_config=SimpleNamespace(
    #                 use_super_kernel=False, use_mlaprolog=False, enable_dsa=False,
    #                 use_prefetch=False, dense_mlp_prefetch=0, lm_head_prefetch=0,
    #                 enable_prefill_micro_batch=False, max_split_token_count_threshold=10**9,
    #                 max_split_token_ratio_threshold=1.0, merge_qkv=False, prefill_moe_all_to_all=False,
    #             ),
    #         )
    #         moe_mlp = _DummyMoEMLP(return_tuple=True)
    #         layer = self._make_decoder_layer_minimal(m, is_ffn_die=False, is_moe=True, mlp=moe_mlp)

    #         hs = torch.randn(3, 2)
    #         res = torch.randn(3, 2)
    #         positions = torch.zeros(3, dtype=torch.int64)
    #         attn = _DummyAttnMeta(prefill=object())

    #         with patch.object(m, "model_extra_config", cfg), patch.object(m, "ConditionalTNGScope", _AcceptAllNullContext):
    #             out_hs, out_res = layer.forward(positions, hs, kv_cache=None, attn_metadata=attn, residual=res)

    #         self.assertTrue(torch.allclose(out_hs, hs + hs))
    #         self.assertTrue(torch.allclose(out_res, res))

    def test_decoder_layer_forward_prefill_dp_or_split_hidden_states_triggers_chunk_loop(self):
        with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
            cfg = SimpleNamespace(
                parall_config=SimpleNamespace(attn_sp_size=1, attn_dies=1),
                task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False, hardware_platform="A3"),
                operator_opt_config=SimpleNamespace(
                    use_super_kernel=False, use_mlaprolog=False, enable_dsa=False,
                    use_prefetch=False, dense_mlp_prefetch=0, lm_head_prefetch=0,
                    enable_prefill_micro_batch=False, max_split_token_count_threshold=10**9,
                    max_split_token_ratio_threshold=1.0, merge_qkv=False, prefill_moe_all_to_all=False,
                ),
            )

            class _DPGroup:
                world_size = 2

            layer = self._make_decoder_layer_minimal(m, is_ffn_die=False, is_moe=False, mlp=_DummyDenseMLP())
            hs = torch.randn(70, 4)
            res = torch.randn(70, 4)
            positions = torch.zeros(70, dtype=torch.int64)
            attn = _DummyAttnMeta(prefill=object())

            def _fake_all_reduce(tensor, op=None, async_op=False):
                tensor.fill_(80)

            with patch.object(m, "model_extra_config", cfg), \
                 patch.object(m, "get_dp_group", return_value=_DPGroup()), \
                 patch.object(m.current_platform, "device_type", "cpu"), \
                 patch.object(m.dist, "all_reduce", side_effect=_fake_all_reduce), \
                 patch.object(m, "ConditionalTNGScope", _AcceptAllNullContext):
                out_hs, out_res = layer.forward(positions, hs, kv_cache=None, attn_metadata=attn, residual=res)

            self.assertEqual(out_hs.shape[0], 70)
            self.assertEqual(out_res.shape[0], 70)
            self.assertEqual(len(layer.mlp.calls), 2)  # 80 padded => split 64 => 2 chunks

    # def test_decoder_layer_forward_ffn_die_skips_norm_and_mlp_and_returns_inputs(self):
    #     with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
    #         cfg = SimpleNamespace(
    #             parall_config=SimpleNamespace(attn_sp_size=1, attn_dies=1),
    #             task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False, hardware_platform="A3"),
    #             operator_opt_config=SimpleNamespace(
    #                 use_super_kernel=False, use_mlaprolog=False, enable_dsa=False,
    #                 use_prefetch=False, dense_mlp_prefetch=0, lm_head_prefetch=0,
    #                 enable_prefill_micro_batch=False, max_split_token_count_threshold=10**9,
    #                 max_split_token_ratio_threshold=1.0, merge_qkv=False, prefill_moe_all_to_all=False,
    #             ),
    #         )
    #         layer = self._make_decoder_layer_minimal(m, is_ffn_die=True, is_moe=False, mlp=_DummyDenseMLP())
    #         hs = torch.randn(5, 4)
    #         res = torch.randn(5, 4)
    #         positions = torch.zeros(5, dtype=torch.int64)
    #         attn = _DummyAttnMeta(prefill=object())

    #         layer.input_layernorm = Mock(side_effect=RuntimeError("should not be called"))
    #         layer.post_attention_layernorm = Mock(side_effect=RuntimeError("should not be called"))
    #         layer.self_attn = Mock(side_effect=RuntimeError("should not be called"))
    #         layer.mlp = Mock(side_effect=RuntimeError("should not be called"))

    #         with patch.object(m, "model_extra_config", cfg), patch.object(m, "ConditionalTNGScope", _AcceptAllNullContext):
    #             out_hs, out_res = layer.forward(positions, hs, kv_cache=None, attn_metadata=attn, residual=res)

    #         self.assertTrue(torch.equal(out_hs, hs))
    #         self.assertTrue(torch.equal(out_res, res))

    # -----------------------------
    # 4) DeepseekV3Model.forward selection + forward_normal PP / SP reverse
    # -----------------------------
    def test_model_forward_selects_micro_batch_branch_when_enabled_and_multi_seq_prefill(self):
        with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
            cfg = SimpleNamespace(
                parall_config=SimpleNamespace(attn_sp_size=1, attn_dies=1),
                task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False, hardware_platform="A3"),
                operator_opt_config=SimpleNamespace(
                    enable_prefill_micro_batch=True,
                    use_super_kernel=False, use_mlaprolog=False, enable_dsa=False, use_prefetch=False,
                    dense_mlp_prefetch=0, lm_head_prefetch=0,
                    max_split_token_count_threshold=10**9, max_split_token_ratio_threshold=1.0,
                    merge_qkv=False, prefill_moe_all_to_all=False,
                ),
            )

            model = m.DeepseekV3Model.__new__(m.DeepseekV3Model)
            nn.Module.__init__(model)
            model.start_layer = 0
            model.prefix = "model.layers"
            model.postfix = ".self_attn.attn"
            model.start_layer_key = f"{model.prefix}.{model.start_layer}{model.postfix}"

            model.forward_micro_batch = Mock(return_value="MB")
            model.forward_normal = Mock(return_value="NM")

            # must be dict so module-level get_layer_attn_metadata can retrieve it
            layer_attn_meta = SimpleNamespace(prefill=SimpleNamespace(seq_lens=[3, 4]))
            attn_metadata = {model.start_layer_key: layer_attn_meta}

            with patch.object(m, "model_extra_config", cfg):
                out = model.forward(
                    input_ids=torch.tensor([1, 2, 3]),
                    positions=torch.tensor([0, 1, 2]),
                    kv_caches=[],
                    attn_metadata=attn_metadata,
                    intermediate_tensors=None,
                    max_num_tokens=None,
                    lm_head=None,
                )

            self.assertEqual(out, "MB")
            model.forward_micro_batch.assert_called_once()

    def test_model_forward_normal_pp_non_first_rank_uses_intermediate_tensors(self):
        with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
            cfg = SimpleNamespace(
                parall_config=SimpleNamespace(attn_sp_size=1, attn_dies=1),
                task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False, hardware_platform="A3"),
                operator_opt_config=SimpleNamespace(
                    enable_prefill_micro_batch=False,
                    use_super_kernel=False, use_mlaprolog=False, enable_dsa=False, use_prefetch=False,
                    dense_mlp_prefetch=0, lm_head_prefetch=0,
                    max_split_token_count_threshold=10**9, max_split_token_ratio_threshold=1.0,
                    merge_qkv=False, prefill_moe_all_to_all=False,
                ),
            )

            class _PP:
                is_first_rank = False
                is_last_rank = False

            model = m.DeepseekV3Model.__new__(m.DeepseekV3Model)
            nn.Module.__init__(model)

            model.start_layer = 0
            model.end_layer = 0
            model.layers = []
            model.first_k_dense_replace = 0
            model.hidden_size = 4
            model.is_ffn_die = False

            # must match __init__ contract
            model.prefix = "model.layers"
            model.postfix = ".self_attn.attn"
            model.start_layer_key = f"{model.prefix}.{model.start_layer}{model.postfix}"

            hs0 = torch.randn(2, 4)
            res0 = torch.randn(2, 4)
            intermediate = m.IntermediateTensors({"hidden_states": hs0, "residual": res0})

            with patch.object(m, "model_extra_config", cfg), patch.object(m, "get_pp_group", return_value=_PP()):
                out = model.forward_normal(
                    input_ids=torch.tensor([1, 2]),
                    positions=torch.tensor([0, 1]),
                    kv_caches=None,
                    attn_metadata=None,  # ok
                    intermediate_tensors=intermediate,
                    lm_head=None,
                )

            self.assertTrue(torch.equal(out["hidden_states"], hs0))
            self.assertTrue(torch.equal(out["residual"], res0))

    def test_model_forward_normal_prefill_sp_split_and_reverse_restore_order(self):
        with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
            cfg = SimpleNamespace(
                parall_config=SimpleNamespace(attn_sp_size=2, attn_dies=1),
                task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False, hardware_platform="A3"),
                operator_opt_config=SimpleNamespace(
                    enable_prefill_micro_batch=False,
                    use_super_kernel=False, use_mlaprolog=False, enable_dsa=False, use_prefetch=False,
                    dense_mlp_prefetch=0, lm_head_prefetch=0,
                    max_split_token_count_threshold=10**9, max_split_token_ratio_threshold=1.0,
                    merge_qkv=False, prefill_moe_all_to_all=False,
                ),
            )

            class _PP:
                is_first_rank = True
                is_last_rank = True

            model = m.DeepseekV3Model.__new__(m.DeepseekV3Model)
            nn.Module.__init__(model)

            model.start_layer = 0
            model.end_layer = 0
            model.layers = []
            model.first_k_dense_replace = 0
            model.hidden_size = 4
            model.is_ffn_die = False
            model.norm = lambda x, residual=None: x if residual is None else (x, residual)

            model.prefix = "model.layers"
            model.postfix = ".self_attn.attn"
            model.start_layer_key = f"{model.prefix}.{model.start_layer}{model.postfix}"

            def _fake_embeddings(input_ids):
                out = torch.zeros((input_ids.shape[0], 4), dtype=torch.float32)
                out[:, 0] = torch.arange(1, input_ids.shape[0] + 1, dtype=torch.float32)
                return out

            model.get_input_embeddings = _fake_embeddings

            prefill = SimpleNamespace(
                actual_query_lens=torch.tensor([3, 3], dtype=torch.int64),
                sp_split_list=[4, 4],
                sp_zigzag_index=[1, 0],
                sp_reverse_split_list=[4, 4],
                sp_reverse_index=[1, 0],
            )
            attn_first = _DummyAttnMeta(prefill=prefill)
            attn_metadata = {model.start_layer_key: attn_first}

            with patch.object(m, "model_extra_config", cfg), \
                 patch.object(m, "get_pp_group", return_value=_PP()), \
                 patch.object(m, "tensor_model_parallel_all_gather", side_effect=lambda x, dim=0: x):
                out = model.forward_normal(
                    input_ids=torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64),
                    positions=torch.arange(6, dtype=torch.int64),
                    kv_caches=None,
                    attn_metadata=attn_metadata,
                    intermediate_tensors=None,
                    lm_head=None,
                )

            nonzero = [v for v in out[:, 0].tolist() if v != 0.0]
            self.assertEqual(nonzero, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # -----------------------------
    # 5) CausalLM wrappers: forward/compute_lmhead/load_weights/should_use_eager_mode
    # -----------------------------
    def test_causallm_forward_last_pp_rank_produces_logits_and_hidden_states_tuple(self):
        with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
            cfg = SimpleNamespace(
                parall_config=SimpleNamespace(attn_sp_size=1, attn_dies=1),
                task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False, hardware_platform="A3"),
                operator_opt_config=SimpleNamespace(
                    prefill_moe_all_to_all=False,
                    use_super_kernel=False, use_mlaprolog=False, enable_dsa=False, use_prefetch=False,
                    dense_mlp_prefetch=0, lm_head_prefetch=0, enable_prefill_micro_batch=False,
                    max_split_token_count_threshold=10**9, max_split_token_ratio_threshold=1.0,
                    merge_qkv=False,
                ),
            )

            class _PP:
                is_last_rank = True

            causal = m.DeepseekV3ForCausalLM.__new__(m.DeepseekV3ForCausalLM)
            nn.Module.__init__(causal)

            causal.is_ffn_die = False
            causal.return_hidden_states = True
            causal.max_num_token = 999
            causal.lm_head = Mock()
            causal.compute_lmhead = Mock(return_value="LOGITS")
            causal.model = Mock(return_value=torch.randn(5, 8))
            causal.config = SimpleNamespace(vocab_size=32000, hidden_size=8)

            with patch.object(m, "model_extra_config", cfg), patch.object(m, "get_pp_group", return_value=_PP()):
                hidden_states, logits = causal.forward(
                    input_ids=torch.tensor([1, 2, 3, 4, 5]),
                    positions=torch.arange(5),
                    kv_caches=None,
                    attn_metadata=None,  # triggers last-token lm_head
                    selected_indices=None,
                    intermediate_tensors=None,
                )

            self.assertEqual(logits, "LOGITS")
            self.assertEqual(hidden_states.shape[0], 5)
            args, _ = causal.compute_lmhead.call_args
            self.assertEqual(args[0].shape[0], 1)  # hidden_states[-1:, ...]

    def test_causallm_compute_lmhead_applies_selected_indices_shape_contract(self):
        with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
            causal = m.DeepseekV3ForCausalLM.__new__(m.DeepseekV3ForCausalLM)
            nn.Module.__init__(causal)

            seen = {}

            def _lm_head(x, embedding_bias=None):
                seen["shape"] = tuple(x.shape)
                return torch.zeros((x.shape[0], 7), dtype=torch.float32)

            causal.lm_head = _lm_head

            hs = torch.randn(2, 3, 4)  # flattened => 6 x 4
            selected = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)  # want 5 rows
            out = causal.compute_lmhead(hs, selected_indices=selected, embedding_bias=None)
            self.assertEqual(seen["shape"], (5, 4))
            self.assertEqual(out.shape, (5, 7))

    def test_causallm_load_weights_stacked_mapping_and_skip_rules_are_applied(self):
        with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
            cfg = SimpleNamespace(
                parall_config=SimpleNamespace(attn_sp_size=1, attn_dies=1),
                task_config=SimpleNamespace(enable_attn_ffn_disaggregation=False, hardware_platform="A3"),
                operator_opt_config=SimpleNamespace(
                    merge_qkv=True,
                    prefill_moe_all_to_all=False,
                    use_super_kernel=False, use_mlaprolog=False, enable_dsa=False, use_prefetch=False,
                    dense_mlp_prefetch=0, lm_head_prefetch=0, enable_prefill_micro_batch=False,
                    max_split_token_count_threshold=10**9, max_split_token_ratio_threshold=1.0,
                ),
            )

            class _Param:
                def __init__(self, name):
                    self.name = name
                    self.calls = []

                def weight_loader(self, param, weight, *args, **kwargs):
                    self.calls.append((param, weight, args, kwargs))

            p_gate = _Param("model.layers.0.mlp.gate_up_proj.weight")
            p_qkv = _Param("model.layers.0.self_attn.qkv_a_proj.weight")
            p_expert = _Param("model.layers.0.mlp.experts.0.gate_up_proj.weight")

            params_items = [
                (p_gate.name, p_gate),
                (p_qkv.name, p_qkv),
                (p_expert.name, p_expert),
            ]

            causal = m.DeepseekV3ForCausalLM.__new__(m.DeepseekV3ForCausalLM)
            nn.Module.__init__(causal)
            causal.config = SimpleNamespace(
                n_routed_experts=1,
                architectures=["X"],
                num_nextn_predict_layers=0,
                num_hidden_layers=1,
            )
            causal.named_parameters = lambda: iter(params_items)

            # FIX: param_name/weight_name should NOT duplicate "model.layers.0." prefix for replace()
            expert_mapping = [
                ("mlp.experts.0.gate_up_proj", "mlp.experts.0.gate_proj", 0, 0),
            ]

            def _is_pp_missing(name, module):
                return "skipme" in name

            weights = [
                ("xxx.rotary_emb.inv_freq", torch.randn(1)),  # skipped
                ("model.layers.0.mlp.gate_proj.weight", torch.randn(2, 2)),  # stacked -> gate_up_proj shard 0
                ("model.layers.0.self_attn.q_a_proj.weight", torch.randn(2, 2)),  # stacked -> qkv_a_proj shard 0
                ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.randn(2, 2)),  # expert mapping
                ("model.layers.0.mlp.gate_proj.bias", torch.randn(2)),  # bias missing -> skipped
                ("model.layers.0.skipme.some.weight", torch.randn(1)),  # pp missing -> skipped
            ]

            with patch.object(m, "model_extra_config", cfg), \
                 patch.object(m.FusedMoE, "make_expert_params_mapping", return_value=expert_mapping), \
                 patch.object(m, "is_pp_missing_parameter", side_effect=_is_pp_missing):
                loaded = causal.load_weights(weights)

            self.assertGreaterEqual(len(p_gate.calls), 1)
            self.assertGreaterEqual(len(p_qkv.calls), 1)
            self.assertGreaterEqual(len(p_expert.calls), 1)

            _, _, args_gate, _ = p_gate.calls[0]
            self.assertEqual(args_gate[0], 0)  # shard_id

            _, _, args_qkv, _ = p_qkv.calls[0]
            self.assertEqual(args_qkv[0], 0)  # shard_id

            _, _, _, kwargs_ex = p_expert.calls[0]
            self.assertIn("shard_id", kwargs_ex)
            self.assertIn("expert_id", kwargs_ex)

            self.assertFalse(any("rotary_emb.inv_freq" in n for n in loaded))

    def test_should_use_eager_mode_prefill_true_decode_false(self):
        with _import_under_test(self._DEEPSEEK_V3_MOD) as m:
            causal = m.DeepseekV3ForCausalLM.__new__(m.DeepseekV3ForCausalLM)
            nn.Module.__init__(causal)

            layer0 = SimpleNamespace(layer_name="k_layer0")
            causal.model = SimpleNamespace(layers=[layer0], start_layer=0)

            self.assertTrue(causal.should_use_eager_mode(attn_metadata=None))

            meta_prefill = SimpleNamespace(prefill=object())
            meta_decode = SimpleNamespace(prefill=None)

            self.assertTrue(causal.should_use_eager_mode(attn_metadata={layer0.layer_name: meta_prefill}))
            self.assertFalse(causal.should_use_eager_mode(attn_metadata={layer0.layer_name: meta_decode}))


if __name__ == "__main__":
    unittest.main()
