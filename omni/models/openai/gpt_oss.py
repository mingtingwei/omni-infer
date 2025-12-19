# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Iterable
from typing import Any, Optional, Union, List

import torch
import torch_npu
import itertools
from torch import nn
from transformers import Qwen2Config
from transformers import PretrainedConfig

from vllm.forward_context import get_forward_context, set_forward_context
from vllm.attention import Attention, AttentionType, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig, ModelConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_pp_group, get_ep_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank, tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from omni.layers.layernorm import RMSNormFlashComm
from omni.layers.linear import (RowParallelFlashCommLinear, QKVParallelFlashCommLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from omni.layers.rotary_embedding import get_rope
from omni.layers.fused_mlp import FusedMLP
from omni.layers.attention.backend.attention import AscendAttentionState
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix, extract_layer_index)
from vllm.model_executor.layers.linear import ReplicatedLinear

from torch.nn.parameter import Parameter
from vllm.model_executor.utils import set_weight_attrs
import torchair as tng


logger = init_logger(__name__)

class GptOssExperts(torch.nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        prefix: str = "",
    ):
        super().__init__()
        self.num_total_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        self.swiglu_limit = config.swiglu_limit
        self.ep_size = get_ep_group().world_size
        self.ep_rank = get_ep_group().rank_in_group
        assert self.num_total_experts % self.ep_size == 0
        
        self.num_experts = self.num_total_experts // self.ep_size
        self.experts_start_idx = self.ep_rank * self.num_experts
        self.experts_end_idx = self.experts_start_idx + self.num_experts
        
        self.gate_up_proj = torch.nn.Parameter(
            torch.empty((self.num_experts, config.hidden_size, config.intermediate_size * 2), dtype=torch.int8),
            requires_grad=False)
        set_weight_attrs(self.gate_up_proj, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.gate_up_proj_scale = torch.nn.Parameter(
            torch.empty((self.num_experts, config.intermediate_size * 2), dtype=torch.float32),
            requires_grad=False)
        set_weight_attrs(self.gate_up_proj_scale, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.gate_up_proj_bias = torch.nn.Parameter(
            torch.empty((self.num_experts, config.intermediate_size * 2), dtype=torch.bfloat16),
            requires_grad=False)
        set_weight_attrs(self.gate_up_proj_bias, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.gate_up_smooth_scale = torch.nn.Parameter(
            torch.ones((self.num_experts, config.intermediate_size), dtype=torch.float32),
            requires_grad=False)

        self.down_proj = torch.nn.Parameter(
            torch.empty((self.num_experts, config.intermediate_size, config.hidden_size), dtype=torch.int8),
            requires_grad=False)
        set_weight_attrs(self.down_proj, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.down_proj_scale = torch.nn.Parameter(
            torch.empty((self.num_experts, config.hidden_size), dtype=torch.bfloat16),
            requires_grad=False)
        set_weight_attrs(self.down_proj_scale, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.down_proj_bias = torch.nn.Parameter(
            torch.empty((self.num_experts, config.hidden_size), dtype=torch.bfloat16),
            requires_grad=False)
        set_weight_attrs(self.down_proj_bias, {"output_dim": 0, "weight_loader": self.weight_loader})
        
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = self.ep_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        loaded_weight = torch.squeeze(loaded_weight)
        loaded_weight = loaded_weight.to(param_data.dtype)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
                
    def process_weights_after_loading(self) -> None:
        gate_up_proj = torch_npu.npu_format_cast(self.gate_up_proj.data, 29)
        self.gate_up_proj = torch.nn.Parameter(gate_up_proj, requires_grad=False)
        down_proj = torch_npu.npu_format_cast(self.down_proj.data, 29)
        self.down_proj = torch.nn.Parameter(down_proj, requires_grad=False)


    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor, is_prefill) -> torch.Tensor:

        if is_prefill:
            return self.forward_prefill(hidden_states, router_logits)
        else:
            return self.forward_decode(hidden_states, router_logits)

    def forward_prefill(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(router_logits, self.experts_per_token,
                                                                    k_group=1, group_count=1, group_select_mode=1)
        topk_weights_sum = torch.sum(topk_weights, dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum

        hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        
        expert_range = [self.experts_start_idx, self.experts_end_idx]
        sorted_tokens, expanded_x_idx, expert_tokens, dynamic_quant_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states_int8, topk_ids, scale=pertoken_scale, offset=None, active_num=topk_ids.numel(), expert_capacity=-1, expert_num=self.num_total_experts, 
            drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=True, quant_mode=-1,active_expert_range=expert_range, row_idx_type=0)

        gate_up_proj_output = torch_npu.npu_grouped_matmul([sorted_tokens], [self.gate_up_proj], bias=None, group_list=expert_tokens,
                                                            split_item=3, output_dtype=torch.int32, group_type=0,
                                                            group_list_type=1)[0]

        gate_up_proj_output_int8, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(gate_up_proj_output,
                                                                                      weight_scale=self.gate_up_proj_scale,
                                                                                      activation_scale=dynamic_quant_scale,
                                                                                      bias=self.gate_up_proj_bias,
                                                                                      quant_scale=self.gate_up_smooth_scale,
                                                                                      quant_offset=None,
                                                                                      group_index=expert_tokens,
                                                                                      activate_left=True,
                                                                                      quant_mode=1,
                                                                                      swiglu_mode=1,
                                                                                      clamp_limit=self.swiglu_limit,
                                                                                      glu_alpha=1.702,
                                                                                      glu_bias=1.0)

        down_proj_output = torch_npu.npu_grouped_matmul([gate_up_proj_output_int8], [self.down_proj], scale=[self.down_proj_scale], 
                                            per_token_scale=[pertoken_scale], bias=[self.down_proj_bias],
                                            group_list=expert_tokens, split_item=3, output_dtype=torch.bfloat16,
                                            group_type=0, group_list_type=1)[0]

        output = torch_npu.npu_moe_finalize_routing(down_proj_output.unsqueeze(1), None, None, None,
                                                    topk_weights.to(torch.float32),
                                                    expanded_x_idx, topk_ids, drop_pad_mode=3).to(torch.bfloat16)

        return output

    def forward_decode(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)

        with tng.scope.npu_stream_switch("gating_topk"):
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(router_logits, self.experts_per_token,
                                                                       k_group=1, group_count=1, group_select_mode=1)

        expert_range = [self.experts_start_idx, self.experts_end_idx]
        sorted_tokens, expanded_x_idx, expert_tokens, dynamic_quant_scale = torch_npu.npu_moe_init_routing_v2(
                hidden_states_int8, topk_ids, scale=pertoken_scale, offset=None, active_num=topk_ids.numel(), expert_capacity=-1, expert_num=self.num_total_experts, 
                drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=True, quant_mode=-1,active_expert_range=expert_range, row_idx_type=1)

        with tng.scope.npu_stream_switch("gmmfr_prolog"):
            batch_size, hidden_size = hidden_states.shape
            topk_weights_sum = torch.sum(topk_weights, dim=-1, keepdim=True)
            topk_weights = topk_weights / topk_weights_sum
            range1 = torch.arange(0, expanded_x_idx.shape[0], dtype=torch.int32, device="npu")
            range2 = range1 * torch.tensor(991, dtype=torch.int32, device="npu")
            mask = (range1 >= torch.sum(expert_tokens)).to(torch.int32)
            expanded_x_idx += range2 * mask
            expanded_x_idx = expanded_x_idx % expanded_x_idx.shape[0]
            expanded_x_idx = torch.clamp(expanded_x_idx, min=0, max=expanded_x_idx.shape[0] - 1)
            sorted_topk_weight = torch.index_select(topk_weights.reshape(-1), 0, expanded_x_idx)
            sorted_topk_weight = sorted_topk_weight.to(torch.float32)
            row_index = torch.floor(torch.div(expanded_x_idx, topk_ids.shape[-1])).to(torch.int64)
            share_input = torch.zeros((max(1, batch_size // self.ep_size), hidden_size), dtype=torch.bfloat16, device="npu")

        gate_up_proj_output = torch_npu.npu_grouped_matmul([sorted_tokens], [self.gate_up_proj], bias=None, group_list=expert_tokens,
                                                                split_item=3, output_dtype=torch.int32, group_type=0,
                                                                group_list_type=1)[0]

        gate_up_proj_output_int8, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(gate_up_proj_output,
                                                                                        weight_scale=self.gate_up_proj_scale,
                                                                                        activation_scale=dynamic_quant_scale,
                                                                                        bias=self.gate_up_proj_bias,
                                                                                        quant_scale=self.gate_up_smooth_scale,
                                                                                        quant_offset=None,
                                                                                        group_index=expert_tokens,
                                                                                        activate_left=True,
                                                                                        quant_mode=1,
                                                                                        swiglu_mode=1,
                                                                                        clamp_limit=self.swiglu_limit,
                                                                                        glu_alpha=1.702,
                                                                                        glu_bias=1.0)

        output = torch_npu.npu_grouped_matmul_finalize_routing(gate_up_proj_output_int8,
                                                                self.down_proj,
                                                                scale=self.down_proj_scale,
                                                                bias=self.down_proj_bias,
                                                                pertoken_scale=pertoken_scale,
                                                                group_list=expert_tokens,
                                                                shared_input=share_input,
                                                                logit=sorted_topk_weight,
                                                                row_index=row_index,
                                                                output_bs=batch_size,
                                                                shared_input_weight=1.0,
                                                                group_list_type=1,
                                                                shared_input_offset=0)
        output = output.to(torch.bfloat16)

        return output
    
class GptOssExpertsBF16(torch.nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        prefix: str = "",
    ):
        super().__init__()
        self.num_total_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        self.swiglu_limit = config.swiglu_limit
        self.ep_size = get_ep_group().world_size
        self.ep_rank = get_ep_group().rank_in_group
        assert self.num_total_experts % self.ep_size == 0
        
        self.num_experts = self.num_total_experts // self.ep_size
        self.experts_start_idx = self.ep_rank * self.num_experts
        self.experts_end_idx = self.experts_start_idx + self.num_experts
        
        self.gate_up_proj = torch.nn.Parameter(
            torch.empty((self.num_experts, config.hidden_size, config.intermediate_size * 2), dtype=torch.bfloat16),
            requires_grad=False)
        set_weight_attrs(self.gate_up_proj, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.gate_up_proj_bias = torch.nn.Parameter(
            torch.empty((self.num_experts, config.intermediate_size * 2), dtype=torch.float32),
            requires_grad=False)
        set_weight_attrs(self.gate_up_proj_bias, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.down_proj = torch.nn.Parameter(
            torch.empty((self.num_experts, config.intermediate_size, config.hidden_size), dtype=torch.bfloat16),
            requires_grad=False)
        set_weight_attrs(self.down_proj, {"output_dim": 0, "weight_loader": self.weight_loader})
        self.down_proj_bias = torch.nn.Parameter(
            torch.empty((self.num_experts, config.hidden_size), dtype=torch.float32),
            requires_grad=False)
        set_weight_attrs(self.down_proj_bias, {"output_dim": 0, "weight_loader": self.weight_loader})
        
        
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = self.ep_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        loaded_weight = torch.squeeze(loaded_weight)
        loaded_weight = loaded_weight.to(param_data.dtype)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def process_weights_after_loading(self) -> None:
        gate_up_proj = torch_npu.npu_format_cast(self.gate_up_proj.data, 29)
        self.gate_up_proj = torch.nn.Parameter(gate_up_proj, requires_grad=False)
        down_proj = torch_npu.npu_format_cast(self.down_proj.data, 29)
        self.down_proj = torch.nn.Parameter(down_proj, requires_grad=False)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor, is_prefill) -> torch.Tensor:
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(router_logits, self.experts_per_token,
                                                                    k_group=1, group_count=1, group_select_mode=1)
        topk_weights_sum = torch.sum(topk_weights, dim=-1, keepdim=True)
        topk_weights = topk_weights / topk_weights_sum
            
        expert_range = [self.experts_start_idx, self.experts_end_idx]
        sorted_tokens, expanded_x_idx, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states, topk_ids, offset=None, active_num=topk_ids.numel(), expert_capacity=-1, expert_num=self.num_total_experts, 
            drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=True, quant_mode=-1,active_expert_range=expert_range, row_idx_type=0)
        
        gate_up_proj_output = torch_npu.npu_grouped_matmul([sorted_tokens], [self.gate_up_proj], 
                                                            bias=[self.gate_up_proj_bias], group_list=expert_tokens,
                                                            split_item=3, output_dtype=torch.bfloat16, group_type=0,
                                                            group_list_type=1)[0]
        
        gate_up_proj_output = torch_npu.npu_clipped_swiglu(x=gate_up_proj_output,
                                                                  group_index=expert_tokens,
                                                                  limit=self.swiglu_limit,
                                                                  alpha=1.702,
                                                                  bias=1.0,
                                                                  interleaved=True)
        
        down_proj_output = torch_npu.npu_grouped_matmul([gate_up_proj_output], [self.down_proj],
                                            bias=[self.down_proj_bias],
                                            group_list=expert_tokens, split_item=3, output_dtype=torch.bfloat16,
                                            group_type=0, group_list_type=1)[0]
        
        output = torch_npu.npu_moe_finalize_routing(down_proj_output.unsqueeze(1), None, None, None,
                                                    topk_weights.to(torch.float32),
                                                    expanded_x_idx, topk_ids, drop_pad_mode=3)
        
        return output

class GptOssMoE(torch.nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        prefix: str = "",
    ):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        self.swiglu_limit = config.swiglu_limit
        self.ep_size = get_ep_group().world_size
        
        self.gating_dtype = torch.float32
        self.router = ReplicatedLinear(config.hidden_size,
                                     config.num_local_experts,
                                     bias=True,
                                     quant_config=None,
                                     params_dtype=self.gating_dtype,
                                     prefix=f"{prefix}.router")
        
        if hasattr(config, "quantization_config"):
            self.experts = GptOssExperts(config, prefix=f"{prefix}.experts")
        else :
            self.experts = GptOssExpertsBF16(config, prefix=f"{prefix}.experts")

    def forward(self, hidden_states: torch.Tensor, is_prefill) -> torch.Tensor:
        router_logits, _ = self.router(hidden_states.to(self.gating_dtype))
        output = self.experts(hidden_states, router_logits, is_prefill)
        if self.ep_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output

class GptOssAttention(nn.Module):

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        rope_scaling: Optional[tuple] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.total_num_heads = num_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.dual_chunk_attention_config = dual_chunk_attention_config

        # attention form
        self.layer_idx = extract_layer_index(prefix)
        self.attn_form = config.layer_types[self.layer_idx] # sliding window or full attention
        cache_config.sliding_window = config.sliding_window if "sliding" in self.attn_form else None
        self.max_model_len = model_config.max_model_len

        # attention sink parameter
        self.sinks = torch.nn.Parameter(torch.empty((self.num_heads), dtype=torch.bfloat16), requires_grad=False)
        set_weight_attrs(self.sinks, {"output_dim": 0, "weight_loader": self.weight_loader})

        self.qkv_proj = QKVParallelFlashCommLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.qkv_proj",
            params_dtype=torch.bfloat16
        )
        self.o_proj = RowParallelFlashCommLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.o_proj",
            params_dtype=torch.bfloat16
        )
        
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            dtype=torch.float32,
            rope_scaling={
                "rope_type":
                "yarn",
                "factor":
                config.rope_scaling["factor"],
                "original_max_position_embeddings":
                config.rope_scaling["original_max_position_embeddings"],
                "beta_fast":
                config.rope_scaling["beta_fast"],
                "beta_slow":
                config.rope_scaling["beta_slow"],
            },
            is_neox_style=True,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            attn_sinks=self.sinks,
            prefix=f"{prefix}.attn")

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def apply_rotary_emb(self, query, key, cos, sin):
        query = query.view(query.shape[0], 1, -1, self.head_dim)
        key = key.view(key.shape[0], 1, -1, self.head_dim)

        query_embed, key_embed = torch_npu.npu_apply_rotary_pos_emb(query, key, cos, sin)
        return query_embed.view(query_embed.shape[0], -1), key_embed.view(key_embed.shape[0], -1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.apply_rotary_emb(q, k, cos, sin)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

class GptOssDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.layer_name = f"{prefix}.self_attn.attn"
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = GptOssAttention(
            config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        
        self.mlp = GptOssMoE(
            config,
            prefix=f"{prefix}.mlp",
        )
        
        self.input_layernorm = RMSNormFlashComm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFlashComm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor],
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        is_prefill: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            cos=cos,
            sin=sin
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states, is_prefill)
        return hidden_states, residual


class GptOssModel(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 model_config: Optional[ModelConfig] = None,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "model",
                 decoder_layer_type: type[nn.Module] = GptOssDecoderLayer):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            assert config.max_window_layers == config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported. "
                "This model uses sliding window but `max_window_layers` = {} "
                "is less than `num_hidden_layers` = {}. Please open an issue "
                "to discuss this feature.".format(
                    config.max_window_layers,
                    config.num_hidden_layers,
                ))

        self.config = config
        self.model_config = model_config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        base = getattr(config, "rope_theta", 1000000)
        rotary_dim = getattr(config, "head_dim", 64)
        max_len = config.max_position_embeddings
        self.rotary_emb = get_rope(
            rotary_dim,
            rotary_dim=rotary_dim,
            max_position=max_len,
            base=base,
            dtype=torch.float32,
            rope_scaling={
                "rope_type": "yarn",
                "factor": config.rope_scaling["factor"],
                "original_max_position_embeddings": config.rope_scaling["original_max_position_embeddings"],
                "beta_fast": config.rope_scaling["beta_fast"],
                "beta_slow": config.rope_scaling["beta_slow"],
            },
            is_neox_style=True,
        )
        full_cos, full_sin = self.rotary_emb._compute_cos_sin_cache()
        self.register_buffer("full_cos", full_cos, persistent=False)
        self.register_buffer("full_sin", full_sin, persistent=False)

        # Use the provided decoder layer type or default to GptOssDecoderLayer
        decoder_layer_type = decoder_layer_type or GptOssDecoderLayer
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(config,
                                              model_config,
                                              cache_config,
                                              quant_config,
                                              prefix),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNormFlashComm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        cos = torch.index_select(self.full_cos, dim=0, index=positions).unsqueeze(1).unsqueeze(1)
        sin = torch.index_select(self.full_sin, dim=0, index=positions).unsqueeze(1).unsqueeze(1)

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None and isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata['model.layers.0.self_attn.attn']

        if attn_metadata is not None and attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            hidden_states, residual = self.forward_layers(positions, hidden_states, residual, kv_caches, cos, sin, is_prefill=True)
        else:
            hidden_states, residual = self.forward_layers(positions, hidden_states, residual, kv_caches, cos, sin, is_prefill=False)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual, y_transform=None)
        return hidden_states

    def forward_layers(self, positions, hidden_states, residual, kv_caches, cos, sin, is_prefill):
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                kv_caches[layer_idx] if kv_caches is not None else None,
                cos, sin,
                is_prefill
            )
        return hidden_states, residual

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name.endswith(".dequant_scale") and name not in params_dict:
                name = name.replace("dequant_scale", "weight_scale")
            if name.endswith(".down_proj.scale") and name not in params_dict:
                name = name.replace("down_proj.scale", "down_proj_scale")
            if name.endswith(".gate_up_proj.scale") and name not in params_dict:
                name = name.replace("gate_up_proj.scale", "gate_up_proj_scale")
            if name.endswith(".down_proj.bias") and name not in params_dict:
                name = name.replace("down_proj.bias", "down_proj_bias")
            if name.endswith(".gate_up_proj.bias") and name not in params_dict:
                name = name.replace("gate_up_proj.bias", "gate_up_proj_bias")
            if name.endswith(".down_proj.weight") and name not in params_dict:
                name = name.replace("down_proj.weight", "down_proj")
            if name.endswith(".gate_up_proj.weight") and name not in params_dict:
                name = name.replace("gate_up_proj.weight", "gate_up_proj")
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:     
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params

@support_torch_compile
class GptOssForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        lora_config = None

        self.config = vllm_config.model_config.hf_config
        self.lora_config = lora_config

        self.quant_config = vllm_config.quant_config
        self.model = GptOssModel(self.config,
                                 vllm_config.model_config,
                                 vllm_config.cache_config,
                                 vllm_config.quant_config,
                                 prefix=maybe_prefix(prefix, "model"))
        self.sampler = Sampler()
        self.is_pd_seperate_d = vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.kv_role == "kv_consumer"

        if get_pp_group().is_last_rank:
            if self.config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(self.config.vocab_size,
                                              self.config.hidden_size,
                                              quant_config=None,
                                              params_dtype=torch.bfloat16,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"),
                                              parallel_lmhead=False)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(self.config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor] = None,
        attn_metadata: AttentionMetadata = None,
        selected_indices: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds = None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, None)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
    
    def process_weights_after_loading(self):
        for _, module in self.model.named_modules():
            if isinstance(module, GptOssExperts):
                module.process_weights_after_loading()
            if isinstance(module, GptOssExpertsBF16):
                module.process_weights_after_loading()

    def process_before_sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
            req_ids: list[str],
    ) -> torch.Tensor:
        HIGH_PROBABILITY_CHOICE = 50
        past_token_lists = sampling_metadata.output_token_ids
        for i, past_tokens_ids in enumerate(past_token_lists):
            # Current processing is only applicable to Chat requests in OpenAI's Harmony format.
            if not req_ids[i].startswith('chat'):
                continue
            # The first three tokens must be <|channel|>analysis<|message|> in open ai harmony format.
            if not self.is_pd_seperate_d:
                if len(past_tokens_ids) == 0:
                    logits[i, 200005] = HIGH_PROBABILITY_CHOICE
                if len(past_tokens_ids) == 1:
                    logits[i, 35644] = HIGH_PROBABILITY_CHOICE
                if len(past_tokens_ids) == 2:
                    logits[i, 200008] = HIGH_PROBABILITY_CHOICE

            if len(past_tokens_ids) >= 3:
                # Rule 1: [<|channel|>(200005), final(17196)] → must be followed by <|message|>(200008)
                if past_tokens_ids[-2] == 200005 and past_tokens_ids[-1] == 17196:
                    logits[i, 200008] = HIGH_PROBABILITY_CHOICE
                # Rule 2: <|message|>(200008) → Avoid directly returning <|return|>(200002)
                if past_tokens_ids[-1] == 200008:
                    logits[i, 200002] = float("-inf")
                # Rule 3: <|call|>(200012) → must be followed by <|start|>(200006)
                if past_tokens_ids[-1] == 200012:
                    logits[i, 200006] = HIGH_PROBABILITY_CHOICE
        return logits

    def sample(
            self,
            logits: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
    
    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.model.layers[self.model.start_layer].layer_name]
        return attn_metadata.attn_state != AscendAttentionState.DecodeOnly