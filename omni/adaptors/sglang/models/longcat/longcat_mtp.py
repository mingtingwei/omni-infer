import concurrent.futures
import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import (
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    block_quant_to_tensor_quant,
    channel_quant_to_tensor_quant,
    normalize_e4m3fn_to_e4m3fnuz,
    requant_weight_ue8m0_inplace,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import (
    BumpAllocator,
    add_prefix,
    bind_or_assign,
    cpu_has_amx_support,
    get_bool_env_var,
)

from omni.adaptors.sglang.distributed import get_mlp_tp_group
from omni.adaptors.sglang.layers.attention.deepseek_mla import DeepseekMLA
from omni.adaptors.sglang.layers.layernorm import RMSNorm
from omni.adaptors.sglang.layers.vocab_parallel_embedding import ParallelLMHead as OmniParallelLMHead
from omni.adaptors.sglang.models.longcat.longcat_flash import (
    LongcatFlashDecoderLayer,
    LongcatFlashForCausalLM,
    LongcatFlashMLP,
)


logger = logging.getLogger(__name__)


class LongcatFlashDenseDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.alt_stream = alt_stream

        self.self_attn = DeepseekMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            rope_is_neox_style=True,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            quant_config=quant_config,
            # quant_config=None,
            layer_id=layer_id,
            reduce_results=False,
            prefix=add_prefix(f"self_attn", prefix)
        )

        self.mlp = LongcatFlashMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.ffn_hidden_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix(f"mlps", prefix),
            tp_size=get_mlp_tp_group().world_size,
            tp_rank=get_mlp_tp_group().rank_in_group,            
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:        
        # 1 input_layernorm
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states,
                residual,
                quant_symbol=self.quant_symbol and not self.use_mla_prolog,
            )

        # 2 self_attn
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        # 3 post_attention_layernorm
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )

        # 4 mlp
        hidden_states = self.mlp(hidden_states, forward_batch)
        return hidden_states, residual


class LongcatFlashModelNextN(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=True,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            # quant_config=quant_config,
            quant_config=None,
            prefix=add_prefix("eh_proj", ""),
        )
        self.decoder = LongcatFlashDenseDecoderLayer(
            config, 0, quant_config=quant_config,
        )

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        total_num_layers = 1
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        cos, sin = self.decoder.self_attn.rotary_emb.get_cos_sin(positions)
        forward_batch.attn_backend.forward_metadata.cos = cos
        forward_batch.attn_backend.forward_metadata.sin = sin

        tp_size = get_tensor_model_parallel_world_size()
        rank_in_group = get_tensor_model_parallel_rank()
        if forward_batch.is_extend_in_batch :
            token_num = forward_batch.spec_info.hidden_states.shape[0]
            start_range = rank_in_group * (token_num // tp_size)
            end_range = (1 + rank_in_group) * (token_num // tp_size)
            forward_batch.spec_info.hidden_states = forward_batch.spec_info.hidden_states[start_range: end_range, :]

        if hidden_states.shape[0] > 0:
            hidden_states, _ = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )
        residual = None
        # with get_global_expert_distribution_recorder().disable_this_region():
        hidden_states, residual = self.decoder(
            positions, hidden_states, forward_batch, residual, zero_allocator
        )

        # if not forward_batch.forward_mode.is_idle():
        if residual is not None:
            hidden_states, _ = self.final_layernorm(hidden_states, residual)
        else:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class LongcatFlashForCausalLMNextN(LongcatFlashForCausalLM):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = (
            None
            if "mtp" in getattr(config, "disable_quant_module", [])
            else quant_config
        )
        self.model = LongcatFlashModelNextN(config, self.quant_config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:              
        hidden_states = self.model(input_ids, positions, forward_batch)
        if forward_batch.is_extend_in_batch:
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            pruned_states = hidden_states[last_index]
        else:
            pruned_states = hidden_states
        logits = self.compute_lmhead(pruned_states)
        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )

    def compute_lmhead(
        self,
        hidden_states: torch.Tensor,
        selected_indices: Optional[torch.Tensor] = None,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if selected_indices is not None:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            if hidden_states.shape[0] != selected_indices.shape[0]:
                hidden_states = hidden_states.index_select(0, selected_indices)

        logits = self.lm_head(hidden_states, embedding_bias)
        return logits
    

    def post_load_weights(self):
        self_attn = self.model.decoder.self_attn
        w = self_attn.kv_b_proj.weight
        use_deep_gemm_bmm = False
        if w.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            if (
                hasattr(self.quant_config, "weight_block_size")
                and self.quant_config.weight_block_size is not None
            ):
                weight_block_size = self.quant_config.weight_block_size
                assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                weight = w
                weight_scale = self_attn.kv_b_proj.weight_scale_inv

                w, scale = block_quant_to_tensor_quant(
                    weight, weight_scale, weight_block_size
                )
                self_attn.w_scale = scale
            else:
                weight = w
                weight_scale = self_attn.kv_b_proj.weight_scale
                w, scale = channel_quant_to_tensor_quant(weight, weight_scale)
                self_attn.w_scale = scale
        if w.dtype == torch.int8:
            if hasattr(self.quant_config, "weight_block_size"):
                # block-wise int8 need it
                weight_block_size = self.quant_config.weight_block_size
                if weight_block_size is not None:
                    assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                    weight = w
                    weight_scale = self_attn.kv_b_proj.weight_scale_inv
                    w = int8_block_dequant(weight, weight_scale, weight_block_size).to(
                        torch.bfloat16
                    )
            else:
                # channel-wise int8 need it
                w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to(
                    torch.bfloat16
                )
        w_kc, w_vc = w.unflatten(
            0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
        ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
        if not use_deep_gemm_bmm:
            self_attn.w_kc = bind_or_assign(
                self_attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            )
            self_attn.w_vc = bind_or_assign(
                self_attn.w_vc, w_vc.contiguous().transpose(1, 2)
            )
            if (
                hasattr(self_attn.kv_b_proj, "weight_scale")
                and self_attn.w_scale is None
            ):
                self_attn.w_scale = bind_or_assign(
                    self_attn.w_scale, self_attn.kv_b_proj.weight_scale
                )
        else:
            num_tiles_k = self_attn.qk_nope_head_dim // weight_block_size[1]
            num_tiles_n = self_attn.v_head_dim // weight_block_size[0]
            ws_kc, ws_vc = block_scale.unflatten(
                0, (-1, (num_tiles_k + num_tiles_n))
            ).split([num_tiles_k, num_tiles_n], dim=1)
            self_attn.w_scale_k = bind_or_assign(
                self_attn.w_scale_k, ws_kc.transpose(1, 2).contiguous()
            )
            self_attn.w_scale_v = bind_or_assign(
                self_attn.w_scale_v, ws_vc.contiguous()
            )
            self_attn.w_kc = bind_or_assign(
                self_attn.w_kc, w_kc.transpose(1, 2).contiguous()
            )
            self_attn.w_vc = bind_or_assign(self_attn.w_vc, w_vc.contiguous())
            self_attn.use_deep_gemm_bmm = True

        if self.config.mla_scale_q_lora: # True
            self_attn.q_a_layernorm.weight.data *= (
                self.config.hidden_size / self.config.q_lora_rank
            ) ** 0.5
        if self.config.mla_scale_kv_lora: # True
            self_attn.kv_a_layernorm.weight.data *= (
                self.config.hidden_size / self.config.kv_lora_rank
            ) ** 0.5


    def _weight_requant_ue8m0(self):
        weight_block_size = self.quant_config.weight_block_size
        layer = self.model.decoder
        self_attn = layer.self_attn
        module_list = [
            self_attn.kv_b_proj,
            self_attn.o_proj,
        ]

        if self.config.q_lora_rank is not None:
            module_list.append(self_attn.fused_qkv_a_proj_with_mqa)
            module_list.append(self_attn.q_b_proj)
        else:
            module_list.append(self_attn.kv_a_proj_with_mqa)
            module_list.append(self_attn.q_proj)

        for module in module_list:
            if hasattr(module, "weight_scale_inv"):
                requant_weight_ue8m0_inplace(
                    module.weight, module.weight_scale_inv, weight_block_size
                )

        mlp = layer.mlp
        assert isinstance(mlp, LongcatFlashMLP)
        for module in [
            mlp.gate_up_proj,
            mlp.down_proj,
        ]:
            if hasattr(module, "weight_scale_inv"):
                requant_weight_ue8m0_inplace(
                    module.weight, module.weight_scale_inv, weight_block_size
                )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        weight_names_mapping = {
            "model.mtp.layers.0.self_attn.kv_b_proj.weight":"model.decoder.self_attn.kv_b_proj.weight",
            "model.mtp.layers.0.self_attn.q_a_proj.weight":"model.decoder.self_attn.q_a_proj.weight",
            "model.mtp.layers.0.self_attn.q_a_proj.weight_scale":"model.decoder.self_attn.q_a_proj.weight_scale",
            "model.mtp.layers.0.transformer_layer.mlp.down_proj.weight":"model.decoder.mlp.down_proj.weight",
            "model.mtp.layers.0.transformer_layer.mlp.down_proj.weight_scale":"model.decoder.mlp.down_proj.weight_scale",
            "model.mtp.embed_tokens.weight":"model.embed_tokens.weight",
            "model.mtp.layers.0.eh_proj.weight":"model.eh_proj.weight",
            "model.mtp.layers.0.self_attn.q_b_proj.weight":"model.decoder.self_attn.q_b_proj.weight",
            "model.mtp.layers.0.self_attn.q_b_proj.weight_scale":"model.decoder.self_attn.q_b_proj.weight_scale",
            "model.mtp.layers.0.transformer_layer.mlp.gate_proj.weight":"model.decoder.mlp.gate_up_proj.weight",
            "model.mtp.layers.0.transformer_layer.mlp.gate_proj.weight_scale":"model.decoder.mlp.gate_up_proj.weight_scale",
            "model.mtp.layers.0.hnorm.m.weight":"model.hnorm.weight",
            "model.mtp.layers.0.post_attention_layernorm.weight":"model.decoder.post_attention_layernorm.weight",
            "model.mtp.layers.0.self_attn.q_a_layernorm.weight":"model.decoder.self_attn.q_a_layernorm.weight",
            "model.mtp.layers.0.enorm.m.weight":"model.enorm.weight",
            "model.mtp.layers.0.self_attn.kv_a_proj_with_mqa.weight":"model.decoder.self_attn.kv_a_proj_with_mqa.weight",
            "model.mtp.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale":"model.decoder.self_attn.kv_a_proj_with_mqa.weight_scale",
            "model.mtp.layers.0.self_attn.o_proj.weight":"model.decoder.self_attn.o_proj.weight",
            "model.mtp.layers.0.self_attn.o_proj.weight_scale":"model.decoder.self_attn.o_proj.weight_scale",
            "model.mtp.norm.weight":"model.final_layernorm.weight", 
            "model.mtp.layers.0.input_layernorm.weight":"model.decoder.input_layernorm.weight",
            "model.mtp.layers.0.self_attn.kv_a_layernorm.weight":"model.decoder.self_attn.kv_a_layernorm.weight",
            "model.mtp.layers.0.transformer_layer.mlp.up_proj.weight":"model.decoder.mlp.gate_up_proj.weight",
            "model.mtp.layers.0.transformer_layer.mlp.up_proj.weight_scale":"model.decoder.mlp.gate_up_proj.weight_scale",
        }
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            params_dict = dict(self.named_parameters())
            for name, loaded_weight in weights:
                if ".mtp." not in name or name not in weight_names_mapping:
                    continue
                elif "gate_proj" in name:
                    name = weight_names_mapping[name]
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )                    
                    futures.append(
                        executor.submit(weight_loader, param, loaded_weight, 0)
                    )                    
                elif "up_proj" in name:
                    name = weight_names_mapping[name]
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )                    
                    futures.append(
                        executor.submit(weight_loader, param, loaded_weight, 1)
                    )
                else:
                    name = weight_names_mapping[name]
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    futures.append(
                        executor.submit(weight_loader, param, loaded_weight)
                    )

        self.post_load_weights()



EntryClass = LongcatFlashForCausalLMNextN