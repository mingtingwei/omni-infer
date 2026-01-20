from collections.abc import Mapping, Set
from dataclasses import dataclass, field
from contextlib import contextmanager

import torch
from torch import nn
from transformers import PretrainedConfig
from unittest.mock import patch
from typing import Callable, Iterator

from omni.models.deepseek.deepseek_v3 import DeepseekV3ForCausalLM
from omni.models.deepseek.deepseek_v32 import DeepseekV32ForCausalLM
from omni.models.pangu.pangu_pro_moe_v2.pangu_moe_v2 import PanguProMoEV2ForCausalLM
from omni.models.pangu.pangu_dense import PanguEmbeddedForCausalLM
from omni.models.pangu.pangu_ultra_moe import PanguUltraMoEForCausalLM
from omni.models.qwen.qwen3_moe import Qwen3MoeForCausalLM
from omni.models.qwen.qwen3 import Qwen3ForCausalLM
from omni.models.bailing.bailing import BailingMoeV2ForCausalLM
@dataclass(frozen=True)
class _HfExamplesInfo:
    model_cls: type[nn.Module]
    hf_config: PretrainedConfig
    init_patch_context_fn: Callable[[], Iterator[None]]
    prompt_token_ids: list[int]

    @contextmanager
    def init_patch_context(self) -> Iterator[None]:
        if self.init_patch_context_fn is None:
            yield
        else:
            with self.init_patch_context_fn():
                yield

config_DeepseekV3ForCausalLM = PretrainedConfig(
    attention_bias=False,
    hidden_size=512,
    num_attention_heads=8,
    num_key_value_heads=8,
    num_nextn_predict_layers=1,
    num_hidden_layers=2,
    intermediate_size=256,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    max_position_embeddings=2048,
    model_type="deepseek_v3",
    n_routed_experts=4,
    n_shared_experts=1,
    moe_intermediate_size=256,
    num_experts_per_tok=2,
    routed_scaling_factor=1.0,
    first_k_dense_replace=1,
    moe_layer_freq=1,
    q_lora_rank=512,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    topk_method="noaux_tc",
    scoring_func="sigmoid",
    seq_aux=True,
    norm_topk_prob=True,
    n_group=1,
    topk_group=1,
    tie_word_embeddings=False,
    torch_dtype=torch.bfloat16,
    use_cache=True,
    vocab_size=10000,
    rope_scaling={
                    "beta_fast": 32,
                    "beta_slow": 1,
                    "factor": 40,
                    "mscale": 1.0,
                    "mscale_all_dim": 1.0,
                    "original_max_position_embeddings": 4096,
                    "type": "yarn"
                },
    enable_speculative=True,
    enable_quantization=True,
    decode_cost_time=0.02,
    quantization_config=None # overwritten when enable_quant=True
)

config_DeepseekV31ForCausalLM = PretrainedConfig(
    attention_bias=False,
    attention_dropout=0.0,
    hidden_size=512,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=8,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    initializer_range=0.02,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    rope_scaling={
        "type": "yarn",
        "factor": 40,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
    },
    q_lora_rank=512,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    n_routed_experts=4,
    n_shared_experts=1,
    moe_intermediate_size=256,
    num_experts_per_tok=2,
    moe_layer_freq=1,
    routed_scaling_factor=1.0,
    first_k_dense_replace=1,
    topk_method="noaux_tc",
    scoring_func="sigmoid",
    norm_topk_prob=True,
    n_group=1,
    topk_group=1,
    num_nextn_predict_layers=1,
    tie_word_embeddings=False,
    use_cache=True,
    torch_dtype=torch.bfloat16,
    vocab_size=10000,
    bos_token_id=0,
    eos_token_id=1,
    model_type="deepseek_v3",
    enable_speculative=True,
    enable_quantization=True,
    decode_cost_time=0.02,
    quantization_config=None,  # overwritten when enable_quant=True
)

config_DeepseekV32ForCausalLM = PretrainedConfig(
    attention_bias=False,
    attention_dropout=0.0,
    hidden_act="silu",
    hidden_size=512,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=8,
    max_position_embeddings=2048,
    rms_norm_eps=1e-6,
    initializer_range=0.02,
    model_type="deepseek_v32",
    n_routed_experts=4,
    n_shared_experts=1,
    moe_intermediate_size=256,
    num_experts_per_tok=2,
    moe_layer_freq=1,
    routed_scaling_factor=1.0,
    first_k_dense_replace=1,
    n_group=1,
    topk_group=1,
    topk_method="noaux_tc",
    scoring_func="sigmoid",
    norm_topk_prob=True,
    q_lora_rank=512,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    rope_theta=10000.0,
    rope_scaling={
        "type": "yarn",
        "factor": 40,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
    },
    vocab_size=10000,
    tie_word_embeddings=False,
    bos_token_id=0,
    eos_token_id=1,
    num_nextn_predict_layers=1,
    use_cache=True,
    torch_dtype=torch.bfloat16,
    enable_speculative=True,
    enable_quantization=True,
    decode_cost_time=0.02,
    quantization_config=None
)

config_PanguEmbeddedForCausalLM = PretrainedConfig(
    architectures=["PanguEmbeddedForCausalLM"],
    model_type="Pangu",
    bos_token_id=0,
    eos_token_id=1,
    pad_token_id=2,
    hidden_size=512,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=8,
    hidden_act="silu",
    rms_norm_eps=1e-5,
    layer_norm_eps=1e-5,
    attention_bias=False,
    bias=False,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    rope_base=10000.0,
    initializer_range=0.02,
    vocab_size=10000,
    tie_word_embeddings=False,
    use_cache=True,
    torch_dtype=torch.bfloat16,
    enable_speculative=True,
    enable_quantization=True,
    decode_cost_time=0.02,
    quantization_config=None,
)

config_PanguProMoEV2ForCausalLM = PretrainedConfig(
    architectures=["PanguProMoEV2ForCausalLM"],
    model_type="PanguProMoE",
    attention_dropout=0.0,
    mlp_only_layers=[0, 1, 2, 3],
    bos_token_id=0,
    eos_token_id=1,
    hidden_act="silu",
    hidden_size=512,
    intermediate_size=256,
    initializer_range=0.02,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    sandwich_norm=True,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=256,
    shared_expert_intermediate_size=512,
    routed_scaling_factor=2.5,
    router_enable_expert_bias=True,
    norm_topk_prob=True,
    output_router_logits=False,
    qk_nope_dim=128,
    qk_rope_dim=64,
    v_channels=128,
    num_mtp_layers=1,
    param_sink_number=128,
    param_sink_with_value=True,
    tie_word_embeddings=False,
    use_cache=True,
    torch_dtype=torch.bfloat16,
    vocab_size=10000,
    enable_speculative=True,
    enable_quantization=True,
    decode_cost_time=0.02,
    quantization_config=None
)
config_PanguUltraMoEForCausalLM = PretrainedConfig(
    architectures=["PanguUltraMoEForCausalLM"],
    model_type="pangu_ultra_moe",
    hidden_act="silu",
    hidden_size=512,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=8,
    num_dense_layers=1,
    moe_intermediate_size=2048,
    num_routed_experts=256,
    num_shared_experts=1,
    num_experts_per_tok=2,
    routed_scaling_factor=2.5,
    attention_bias=False,
    attention_q_lora_dim=512,
    attention_kv_lora_dim=512,
    attention_qk_dim=128,
    attention_qk_rope_dim=64,
    attention_v_dim=128,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    sandwich_norm=True,
    initializer_range=0.02,
    num_mtp_layers=1,
    vocab_size=10000,
    tie_word_embeddings=False,
    use_cache=True,
    torch_dtype=torch.bfloat16,
    enable_speculative=True,
    enable_quantization=True,
    decode_cost_time=0.02,
    quantization_config=None,
)
config_Qwen3MoeForCausalLM = PretrainedConfig(
    architectures=["Qwen3MoeForCausalLM"],
    model_type="qwen3_moe",
    bos_token_id=0,
    eos_token_id=1,
    hidden_size=1024,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=128,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    initializer_range=0.02,
    attention_bias=False,
    attention_dropout=0.0,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    rope_scaling=None,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=256,
    norm_topk_prob=True,
    output_router_logits=False,
    router_aux_loss_coef=0.001,
    decoder_sparse_step=1,
    max_window_layers=2,
    mlp_only_layers=[],
    sliding_window=None,
    use_sliding_window=False,
    num_nextn_predict_layers=1,
    vocab_size=10000,
    tie_word_embeddings=False,
    use_cache=True,
    torch_dtype=torch.bfloat16,
    enable_speculative=True,
    enable_quantization=True,
    decode_cost_time=0.022,
    quantization_config=None,
)
config_Qwen3ForCausalLM = PretrainedConfig(
    architectures=["Qwen3ForCausalLM"],
    model_type="qwen3",
    bos_token_id=0,
    eos_token_id=1,
    vocab_size=10000,
    tie_word_embeddings=False,
    hidden_size=512,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=128,
    attention_bias=False,
    attention_dropout=0.0,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    initializer_range=0.02,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    rope_scaling=None,
    max_window_layers=2,
    sliding_window=None,
    use_sliding_window=False,
    use_cache=True,
    torch_dtype=torch.bfloat16,
    enable_speculative=False,
    enable_quantization=True,
    decode_cost_time=0.022,
    quantization_config=None,
)
config_BailingMoeV2ForCausalLM = PretrainedConfig(
    architectures=["BailingMoeV2ForCausalLM"],
    model_type="bailing_moe",
    eos_token_id=1,
    pad_token_id=2,
    vocab_size=10000,
    tie_word_embeddings=False,
    num_hidden_layers=2,
    hidden_size=512,
    intermediate_size=2048,
    moe_intermediate_size=2048,
    num_attention_heads=8,
    num_key_value_heads=8,
    head_dim=128,
    attention_dropout=0.0,
    use_qkv_bias=False,
    use_qk_norm=True,
    norm_head=False,
    hidden_act="silu",
    use_rmsnorm=True,
    rms_norm_eps=1e-6,
    norm_softmax=False,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    rope_scaling=None,
    rotary_dim=64,
    num_experts=8,
    num_experts_per_tok=2,
    num_shared_experts=1,
    norm_topk_prob=True,
    moe_router_enable_expert_bias=True,
    routed_scaling_factor=2.5,
    router_dtype="fp32",
    score_function="sigmoid",
    n_group=1,
    topk_group=1,
    first_k_dense_replace=1,
    using_split_qkv_in_self_attention=False,
    use_bias=False,
    embedding_dropout=0.0,
    output_dropout=0.0,
    use_cache=True,
    torch_dtype=torch.bfloat16,
    enable_speculative=False,
    enable_quantization=False,
    decode_cost_time=0.01,
    quantization_config=None,
)
@contextmanager
def init_patch_context_DeepseekV3ForCausalLM() -> Iterator[None]:
    class MockDpGroup:
        def __init__(self):
            self.world_size = 2
    with patch("omni.layers.attention.deepseek_mla.get_dp_group", return_value=MockDpGroup()):
        yield

_TRANSFORMERS_MODELS = {
    "DeepseekV3ForCausalLM": _HfExamplesInfo(
        model_cls=DeepseekV3ForCausalLM,
        hf_config=config_DeepseekV3ForCausalLM,
        init_patch_context_fn=init_patch_context_DeepseekV3ForCausalLM,
        prompt_token_ids=[0, 128803, 122294, 1148, 128804, 128798, 201], # 你是谁？
    ),
    "DeepseekV31ForCausalLM": _HfExamplesInfo(
        model_cls=DeepseekV3ForCausalLM,
        hf_config=config_DeepseekV31ForCausalLM,
        init_patch_context_fn=init_patch_context_DeepseekV3ForCausalLM,
        prompt_token_ids=[0, 128803, 122294, 1148, 128804, 128798, 201],
    ),
    "DeepseekV32ForCausalLM": _HfExamplesInfo(
        model_cls=DeepseekV32ForCausalLM,
        hf_config=config_DeepseekV32ForCausalLM,
        init_patch_context_fn=init_patch_context_DeepseekV3ForCausalLM,
        prompt_token_ids=[0, 128803, 122294, 1148, 128804, 128798, 201],
    ),
    "PanguEmbeddedForCausalLM": _HfExamplesInfo(
        model_cls=PanguEmbeddedForCausalLM,
        hf_config=config_PanguEmbeddedForCausalLM,
        init_patch_context_fn=None,
        prompt_token_ids=[0, 128803, 122294, 1148, 128804, 128798, 201],
    ),
    "PanguProMoEV2ForCausalLM": _HfExamplesInfo(
        model_cls=PanguProMoEV2ForCausalLM,
        hf_config=config_PanguProMoEV2ForCausalLM,
        init_patch_context_fn=None,
        prompt_token_ids=[0, 128803, 122294, 1148, 128804, 128798, 201],
    ),
    "PanguUltraMoEForCausalLM":_HfExamplesInfo(
        model_cls=PanguUltraMoEForCausalLM,
        hf_config=config_PanguUltraMoEForCausalLM,
        init_patch_context_fn=init_patch_context_DeepseekV3ForCausalLM,
        prompt_token_ids=[0, 128803, 122294, 1148, 128804, 128798, 201],
    ),
    "Qwen3MoeForCausalLM":_HfExamplesInfo(
        model_cls=Qwen3MoeForCausalLM,
        hf_config=config_Qwen3MoeForCausalLM,
        init_patch_context_fn=None,
        prompt_token_ids=[0, 128803, 122294, 1148, 128804, 128798, 201],
    ),
    "Qwen3ForCausalLM":_HfExamplesInfo(
        model_cls=Qwen3ForCausalLM,
        hf_config=config_Qwen3ForCausalLM,
        init_patch_context_fn=None,
        prompt_token_ids=[0, 128803, 122294, 1148, 128804, 128798, 201],
    ),
    "BailingMoeV2ForCausalLM":_HfExamplesInfo(
        model_cls=BailingMoeV2ForCausalLM,
        hf_config=config_BailingMoeV2ForCausalLM,
        init_patch_context_fn=init_patch_context_DeepseekV3ForCausalLM,
        prompt_token_ids=[0, 128803, 122294, 1148, 128804, 128798, 201],
    ),
}

_EXAMPLE_MODELS = {
    **_TRANSFORMERS_MODELS,
}

class HfExampleModels:
    def __init__(self, hf_models: Mapping[str, _HfExamplesInfo]) -> None:
        super().__init__()

        self.hf_models = hf_models

    def get_supported_archs(self) -> Set[str]:
        return self.hf_models.keys()

    def get_hf_info(self, model_arch: str) -> _HfExamplesInfo:
        return self.hf_models[model_arch]

    def find_hf_info(self, model_id: str) -> _HfExamplesInfo:
        for info in self.hf_models.values():
            if info.default == model_id:
                return info

        # Fallback to extras
        for info in self.hf_models.values():
            if any(extra == model_id for extra in info.extras.values()):
                return info

        raise ValueError(f"No example model defined for {model_id}")


HF_EXAMPLE_MODELS = HfExampleModels(_EXAMPLE_MODELS)

