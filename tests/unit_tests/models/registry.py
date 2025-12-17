from collections.abc import Mapping, Set
from dataclasses import dataclass, field
from contextlib import contextmanager

import torch
from torch import nn
from transformers import PretrainedConfig
from unittest.mock import patch
from typing import Callable, Iterator

from omni.models.deepseek.deepseek_v3 import DeepseekV3ForCausalLM

@dataclass(frozen=True)
class _HfExamplesInfo:
    model_cls: type[nn.Module]
    hf_config: PretrainedConfig
    init_patch_context_fn: Callable[[], Iterator[None]]
    prompt_token_ids: list[int]

    @contextmanager
    def init_patch_context(self) -> Iterator[None]:
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
        quantization_config=None
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

