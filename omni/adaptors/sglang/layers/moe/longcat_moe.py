# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Inference-only Longcat-flash model."""
from typing import Optional
import torch, torch_npu
from torch import nn
from transformers import PretrainedConfig
torch._logging.set_logs(recompiles=True)
from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_tensor_model_parallel_world_size,
    get_world_group,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.utils import add_prefix

from omni.adaptors.sglang.layers.moe.fused_moe.layer import FusedMoE


"""NPU Stream Switch Names"""
STREAM_SHARED_EXPERT = 'stream_shared_expert'
SEQ_SPLIT_LENGTH = 4096


def rank_log(msg, rank=0):
    if get_world_group().rank_in_group == rank:
        print(msg)


class LongcatFlashTopkRouter(nn.Module):
    def __init__(
        self, 
        config, 
        zero_expert_num: int = 0,
        router_expert_num: int = 256,
        prefix: str = "", 
        rounter_params_dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.config = config
        self.top_k = config.moe_topk
        self.num_experts = router_expert_num + zero_expert_num
        self.routed_scaling_factor = config.routed_scaling_factor

        self.classifier = ReplicatedLinear(
            config.hidden_size,
            self.num_experts,
            bias=config.router_bias,
            params_dtype=rounter_params_dtype,
            quant_config=None,
            prefix=f"{prefix}.classifier",
        )
        self.classifier.quant_method.enable_weight_nz = False
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros((self.num_experts), dtype=rounter_params_dtype)
        )

    def forward(self, hidden_states):
        return self.classifier(hidden_states)

    def get_topk_indices(self, router_logits):
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
            router_logits.float(),
            k=self.top_k,
            bias=self.e_score_correction_bias,  # float32
            k_group=1,
            group_count=1,
            group_select_mode=0,  # 0: maximum in group; 1: topk2.sum(fix)
            renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
            norm_type=0,  # 0: softmax; 1: sigmoid(fix)
            routed_scaling_factor=self.routed_scaling_factor,
            eps=float(1e-20)
        )
        return topk_weights, topk_ids

class LongcatFlashMoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()
        self.prefix = prefix
        self.config = config
        self.layer_id = layer_id
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_size = get_moe_expert_parallel_world_size()
        self.top_k = config.moe_topk
        self.n_routed_experts = config.n_routed_experts
        self.zero_expert_num = config.zero_expert_num
        self.quant_symbol = True if quant_config else False

        assert global_server_args_dict["moe_a2a_backend"].is_deepep()

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        # ========================== init route ==============================
        self.router = LongcatFlashTopkRouter(config=config,
                                             zero_expert_num=self.zero_expert_num,
                                             router_expert_num=self.n_routed_experts,
                                             rounter_params_dtype=torch.float32,
                                             prefix=f"{prefix}.router")

        # ====================== init FusedMoE ======================
        self.ep_num_redundant_experts = global_server_args_dict["ep_num_redundant_experts"]
        is_nextn = kwargs.get("is_nextn",False)
        if is_nextn:
            self.ep_num_redundant_experts = 0
        
        self.num_experts = self.n_routed_experts + self.ep_num_redundant_experts
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.expert_ffn_hidden_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=config.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
        )

        self.smooth_scale = None
        self.in_scale_2 = None
        if self.quant_symbol:
            self.local_expert_num = self.experts.w13_weight.shape[0]

            self.in_scale_2 = torch.ones(
                (self.local_expert_num, config.expert_ffn_hidden_size),
                dtype=torch.float32,
                device="npu")
            epsilon = 1e-2
            self.smooth_scale = torch.nn.Parameter(
                torch.ones(
                    size=(self.num_experts, config.hidden_size),
                    dtype=torch.float32
                ) * (1 - epsilon) + epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        **kwargs
    ) -> torch.Tensor:
        if forward_batch.is_extend_in_batch:
            return self._forward_prefill_norm(hidden_states, forward_batch)
        else:
            return self._forward_decode_dispatch_combine(hidden_states, forward_batch)

    def tranfer_zero_expert_ids(self, topk_ids: torch.Tensor, zero_expert_mask: torch.Tensor, normal_expert_mask: torch.Tensor) -> torch.Tensor:
        device = topk_ids.device
        out_dtype = topk_ids.dtype

        row_idx, slot_idx = torch.nonzero(normal_expert_mask, as_tuple=True)
        used_ids_flat = topk_ids[row_idx, slot_idx].to(torch.long)

        used_mask = torch.zeros((topk_ids.shape[0], self.n_routed_experts), dtype=torch.bool, device=device)
        used_mask.index_put_((row_idx, used_ids_flat), torch.ones_like(used_ids_flat, dtype=torch.bool), accumulate=True)
        available_mask = ~used_mask

        num_zero_each_row = zero_expert_mask.sum(dim=1)
        max_zero_each_row = int(num_zero_each_row.max().item())
        if max_zero_each_row == 0:
            return topk_ids

        idxN = torch.arange(self.n_routed_experts, device=device).float().unsqueeze(0).expand(topk_ids.shape[0], -1)
        score_cand = torch.where(available_mask, -idxN, torch.full_like(idxN, float('-inf')))
        _, cand_idx = torch.topk(score_cand, k=max_zero_each_row, largest=True, sorted=True)

        col_idx = torch.arange(topk_ids.shape[1], device=device).float().unsqueeze(0).expand(topk_ids.shape[0], -1)
        score_zero = torch.where(zero_expert_mask, -col_idx, torch.full_like(col_idx, float('-inf')))
        _, z_indices_pad = torch.topk(score_zero, k=max_zero_each_row, largest=True, sorted=True)

        j = torch.arange(max_zero_each_row, device=device).unsqueeze(0).expand(topk_ids.shape[0], max_zero_each_row)
        valid_assign = j < num_zero_each_row.view(-1, 1)

        rows_expanded = torch.arange(topk_ids.shape[0], device=device).unsqueeze(1).expand_as(j)
        rows_sel = rows_expanded[valid_assign]
        cols_sel = z_indices_pad[valid_assign]
        vals_sel = cand_idx[valid_assign].to(out_dtype)

        topk_ids.index_put_((rows_sel, cols_sel), vals_sel, accumulate=False)
        return topk_ids

    def compute_zero_experts(self, hidden_states, topk_weights, topk_ids):
        zero_expert_mask = topk_ids >= self.n_routed_experts
        normal_expert_mask = topk_ids < self.n_routed_experts

        zero_expert_weights = topk_weights.clone()
        zero_expert_weights[normal_expert_mask] = 0.0
        total_weights = zero_expert_weights.sum(dim=-1, keepdim=True)   # [T, 1]
        zero_expert_output = hidden_states * total_weights              # [T, H]
        zero_expert_output = zero_expert_output.to(hidden_states.dtype)
        
        topk_ids = self.tranfer_zero_expert_ids(topk_ids, zero_expert_mask, normal_expert_mask)
        topk_weights[zero_expert_mask] = 0.0
        return zero_expert_output, topk_ids, topk_weights

    def _forward_prefill_norm(self, hidden_states, forward_batch) -> torch.Tensor:
        if hidden_states.shape[0] > 0 and not forward_batch.is_prefill_idle:
            router_logits, _ = self.router.forward(hidden_states.float())
            topk_weights, topk_ids = self.router.get_topk_indices(router_logits)
            zero_expert_output, topk_ids, topk_weights = self.compute_zero_experts(hidden_states, topk_weights, topk_ids)
        else:
            topk_ids = torch.randperm(self.num_experts)[:hidden_states.size(0) * self.top_k].reshape(
                hidden_states.size(0), self.top_k
                ).npu()

            topk_weights = torch.empty(
                (hidden_states.size(0), self.top_k),
                dtype=torch.float32,
                device="npu")
            zero_expert_output = torch.zeros_like(hidden_states)

        final_hidden_states_list = self.experts(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            forward_batch=forward_batch,
            dynamic_scale=self.smooth_scale,              # TODO: vLLM use None
            comm_group=self.experts.moe_all_to_all_group, # TODO: vLLM use None
        )

        if len(final_hidden_states_list) != 3:
            raise RuntimeError("len(final_hidden_states_list) != 3")

        gathered_tokens = final_hidden_states_list[1]
        expanded_row_idx = final_hidden_states_list[2]

        # finalize-routing
        hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_tokens,
            skip1=zero_expert_output,
            skip2=None,
            bias=None,
            scales=topk_weights.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None,
            drop_pad_mode=2,
        )

        return hidden_states

    def _forward_decode_dispatch_combine(self, hidden_states, forward_batch) -> torch.Tensor:
        router_logits, _ = self.router.forward(hidden_states.float())
        hidden_states = hidden_states.unsqueeze(1).squeeze(1)
        topk_weights, topk_ids = self.router.get_topk_indices(router_logits)

        metadata = forward_batch.attn_backend.forward_metadata
        mc2_mask = metadata.mc2_mask
        
        max_num_deployed_expert = self.num_experts
        act_dtype = hidden_states.dtype
        shared_expert_rank_num = 0
        kwargs = {
            "x": hidden_states,
            "expert_ids": topk_ids,  # [n*topk]
            "expert_shard_type": 0,  # Set it to 0 for now
            "shared_expert_rank_num": shared_expert_rank_num,  # 32
            "moe_expert_num": max_num_deployed_expert, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": 0,  # 0 Default (all); all tokens can be set
        }

        # In fact, what we get is the die number, and the ep group is not adapted by default.
        # The default ep group is experts_num/die_num.
        global_rank = get_world_group().rank_in_group

        kwargs.update({
            "scales": None,  # Quantization coefficient
            "quant_mode": self.experts.quant_mode,  # 0: Non-quantization; 1: Static quantization; 2: Dynamic quantization
            "group_ep": self.experts.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": self.experts.all_to_all_group_size,
            "ep_rank_id": self.experts.all_to_all_rank,
            "group_tp": self.experts.moe_rs_group_name,
            "tp_world_size": self.experts.tp_size,
            "tp_rank_id": self.experts.tp_rank,
            "x_active_mask": mc2_mask,
            "zero_expert_num": self.zero_expert_num,
            "copy_expert_num": 0,
            "const_expert_num": 0,
        })

        output = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)
        expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts = output[0:5]

        group_list = expert_token_nums.to(torch.int64)

        # cal experts
        weight1_3 = self.experts.w13_weight
        weight2 = self.experts.w2_weight
        
        if self.quant_symbol:
            if self.experts.quant_mode != 0:
                pertoken_scale = dynamic_scale
            else:
                expand_x, pertoken_scale = torch_npu.npu_dynamic_quant(expand_x)
            if self.experts.weight_num_bits == 8:
                weight_scale1_3 = self.experts.w13_weight_scale.squeeze(-1).to(torch.float32) # adapt shape and dtype
                weight_scale2 = self.experts.w2_weight_scale.squeeze(-1).to(torch.bfloat16) # adapt shape and dtype
                gate_up_proj = torch_npu.npu_grouped_matmul(
                    [expand_x],
                    [weight1_3],
                    bias=None,
                    group_list=group_list,
                    split_item=3,
                    output_dtype=torch.int32,
                    group_type=0,
                    group_list_type=1)[0]

                gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                    gate_up_proj,
                    weight_scale=weight_scale1_3,
                    activation_scale=pertoken_scale,
                    bias=None,
                    quant_scale=self.in_scale_2,
                    quant_offset=None,
                    group_index=group_list,
                    activate_left=True,
                    quant_mode=1) # 1: dynamic quant; 0: static quant(not supported now)

                hidden_states_experts = torch_npu.npu_grouped_matmul(
                    [gate_up_proj],
                    [weight2],
                    scale=[weight_scale2],
                    per_token_scale=[pertoken_scale],
                    bias=None,
                    group_list=group_list,
                    split_item=3,
                    output_dtype=act_dtype,
                    group_type=0,
                    group_list_type=1)[0]
            else:
                raise NotImplementedError(f"Unsupported compress tensor type. num bits: {self.experts.weight_num_bits}")
        else:
            # bf16
            gate_up_proj = torch_npu.npu_grouped_matmul([expand_x], [weight1_3], bias=None, group_list=group_list,
                                                    split_item=3, group_type=0, group_list_type=1)[0]
        
            gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)

            hidden_states_experts = torch_npu.npu_grouped_matmul([gate_up_proj], [weight2],bias=None,
                                            group_list=group_list, split_item=3, output_dtype=act_dtype,
                                            group_type=0, group_list_type=1)[0]

        # moeCombine
        kwargs = {
            "expand_x": hidden_states_experts,
            "expert_ids": topk_ids,  # [n*topk]
            "assist_info_for_combine": expand_idx,
            "expert_scales": topk_weights.to(torch.float32),  # weight [n*topk]
            "expert_shard_type": 0,
            "shared_expert_rank_num": shared_expert_rank_num,
            "moe_expert_num":  max_num_deployed_expert, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
            "global_bs": 0,  # 0 Default (all); all tokens can be set
        }
        tp_recv_counts = output[5]
        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,  # dispatch's send_counts
            "group_ep": self.experts.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
            "ep_world_size": self.experts.all_to_all_group_size,
            "ep_rank_id": self.experts.all_to_all_rank,
            "tp_send_counts": tp_recv_counts,
            "group_tp": self.experts.moe_rs_group_name,
            "tp_world_size": self.experts.tp_size,
            "tp_rank_id": self.experts.tp_rank,
            "x_active_mask": mc2_mask,
            "zero_expert_num": self.zero_expert_num,
            "copy_expert_num": 0,
            "const_expert_num": 0,
        }
        kwargs.update(stage3_kwargs)

        final_hidden_states = torch_npu.npu_moe_distribute_combine_v2(**kwargs)
        return final_hidden_states
