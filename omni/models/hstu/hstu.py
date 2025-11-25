import math
import torch
import torch.nn.functional as F
import torch_npu

from typing import Iterable, List, Optional, Union, Optional
from torch.autograd.profiler import record_function

from vllm.compilation.decorators import support_torch_compile
from vllm.attention import Attention
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.attention.backends.abstract import AttentionMetadata

from omni.layers.rotary_embedding import get_rope
from omni.models.hstu.hstu_config import (
    HSTUInferenceRankingConfig, InferenceEmbeddingConfig,
    InferenceHSTUConfig, RankingConfig
)
from omni.layers.attention.backend.attention import AscendAttentionState
from vllm.forward_context import ForwardContext, get_forward_context

from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          is_v1_kv_transfer_group)


def init_mlp_weights_optional_bias(
    m: torch.nn.Module,
) -> None:
    """
    Initialize the weights of a linear layer and optionally the bias.

    Args:
        m: The module to initialize.
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # Always initialize bias to zero.
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class MLP(torch.nn.Module):  # type: ignore
    """
    Multi-Layer Perceptron (MLP) module wrapper for processing jagged data.

    Args:
        in_size (int): The input size.
        layer_sizes (List[int]): The sizes of the layers.
        bias (bool, optional): Whether to include bias in the layers. Defaults to True.
        activation (Union[str, Callable[[], torch.nn.Module], torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]], optional): The activation function. Defaults to torch.relu.
        device (Optional[torch.device], optional): The device to use. Defaults to None.
        dtype (torch.dtype, optional): The data type. Defaults to torch.float32.
    """

    def __init__(
        self,
        in_size: int,
        layer_sizes: List[int],
        activation: str = "relu",
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if activation == "relu":
            activation_fn = torch.nn.ReLU
        elif activation == "gelu":
            activation_fn = torch.nn.GELU
        else:
            raise ValueError(f"Activation function {activation} not supported")

        layers = []
        for i in range(len(layer_sizes)):
            layers.extend(
                [
                    torch.nn.Linear(
                        layer_sizes[i - 1] if i > 0 else in_size,
                        layer_sizes[i],
                        bias=bias,
                        device=device,
                        dtype=dtype,
                    ),
                    activation_fn()
                    if i < len(layer_sizes) - 1
                    else torch.nn.Identity(),
                ]
            )

        self._mlp = torch.nn.Sequential(*layers)
        self._mlp.apply(init_mlp_weights_optional_bias)
        self.dtype = dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP module.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        assert input.dim() == 2, "Tensor must be 2-dimensional"
        return self._mlp(input)


class RMSNorm_npu(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, residual: Optional[torch.Tensor]):
        if residual is None:
            y = torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
        else:
            y, _, x = torch_npu.npu_add_rms_norm(residual, x, self.weight, self.eps)
        return y, x


class FFN_npu_swiglu(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False).to(torch.float16)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False).to(torch.float16)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False).to(torch.float16)
        self.dropout = torch.nn.Dropout(dropout).to(torch.bfloat16)
        self.W_1 = torch.cat([self.w3.weight, self.w1.weight], dim = 0).transpose(0,1)
        self.W_2 = self.w2.weight.transpose(0,1)


    def forward(self, x):
        return self.dropout((torch_npu.npu_ffn(x.to(torch.float16), self.W_1, self.W_2, 'swiglu',inner_precise=1)).to(torch.bfloat16))


class InferenceEmbedding(torch.nn.Module):
    """
    InferenceEmbedding is a module for embeddings in the inference stage.

    Args:
        embedding_configs (List[InferenceEmbeddingConfig]): Configuration for the hstu (sharded) embedding.
    """

    def __init__(
        self,
        embedding_configs: List[InferenceEmbeddingConfig],
    ):
        super(InferenceEmbedding, self).__init__()
        sum_vocab_size = 0
        embedding_table_name = []
        for config in embedding_configs:
            if config.table_name not in embedding_table_name:
                sum_vocab_size += config.vocab_size
                dim = config.dim
                embedding_table_name.append(config.table_name)
        
        self._embedding_layer = torch.nn.Embedding(
            num_embeddings=sum_vocab_size,
            embedding_dim=dim,
        )

    def to_empty(self, device: torch.device):
        super().to_empty(device=device)

        @torch.no_grad()
        def truncated_normal_(tensor, mean=0.0, std=0.02, lower=-2.0, upper=2.0):
            """
            Fills the input tensor with values drawn from a truncated normal distribution.

            Args:
                tensor (torch.Tensor): an n-dimensional tensor
                mean (float): mean of the normal distribution
                std (float): standard deviation of the normal distribution
                lower (float): lower bound (in terms of number of standard deviations)
                upper (float): upper bound (in terms of number of standard deviations)

            Returns:
                None. Fills the input tensor in-place.
            """
            size = tensor.size()
            tmp = tensor.new_empty(size).normal_()

            # Clamp to [lower, upper] in standard normal units
            tmp = tmp.clamp(min=lower, max=upper)

            # Scale and shift to desired distribution
            tensor.copy_(tmp)
            tensor.mul_(std).add_(mean)


        @torch.no_grad()
        def init_embedding_weights(m):
            """
            Initialize embedding weights using truncated normal distribution.
            """
            if isinstance(m, torch.nn.Embedding):
                truncated_normal_(m.weight, mean=0.0, std=0.02)

        self.apply(init_embedding_weights)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_layer(input_ids)


class HstuAttention(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: InferenceHSTUConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = config.hidden_size
        self._linear_dim_per_head: int = config.head_dim
        self._attention_dim_per_head: int = config.head_dim
        self._eps = config.layernorm_epsilon
        self._num_heads: int = config.num_heads

        self._split_arg_list = [
            self._linear_dim_per_head * self._num_heads,
            self._linear_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
        ]

        dtype = (
            torch.bfloat16
            if config.bf16
            else torch.float16
            if config.fp16
            else torch.float32
        )
        device = torch_npu.npu.current_device()

        # linear_uvqk
        self._linear_uvqk = torch.nn.Linear(
            self._embedding_dim,
            (self._linear_dim_per_head * 2 + self._attention_dim_per_head * 2)
            * self._num_heads,
            bias=True,
            dtype=dtype,
            device=device,
        )
        for param in self._linear_uvqk.parameters():
            param.requires_grad = False
            param.copy_(torch.empty_like(param).uniform_(-0.5, 0.5))
        self._linear_uvqk_weight = self._linear_uvqk.weight.T.contiguous()

        # input norm
        if config.learnable_input_layernorm:
            self._input_layernorm_weight = torch.nn.Parameter(
                torch.ones(self._embedding_dim, dtype=dtype, device=device),
                requires_grad=False,
            )
            self._input_layernorm_bias = torch.nn.Parameter(
                torch.zeros(self._embedding_dim, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self._input_layernorm_weight = None
            self._input_layernorm_bias = None

        self.rotary_emb = get_rope(
            head_size=config.head_dim,
            rotary_dim=config.head_dim,
            # TODO fix hard code
            max_position=10000000 + 1024,
            base=10000,
        )

        self._layer_name = f"model.layers.{layer_idx}.attn"
        
        self.attn = Attention(
            num_heads=config.num_heads,
            head_size=config.head_dim,
            scale=1.0 / math.sqrt(config.head_dim),
            num_kv_heads=config.num_heads,
            prefix=self._layer_name,
        )
        

    def forward(
        self,
        layer_input: torch.Tensor,
        positions: torch.Tensor,
    ):
        normed_input = F.layer_norm(
            layer_input,
            normalized_shape=[self._embedding_dim],
            weight=self._input_layernorm_weight,
            bias=self._input_layernorm_bias,
            eps=self._eps,
        )

        mixed_uvqk = F.silu(self._linear_uvqk(normed_input))
        (user, value, query, key) = torch.split(
            mixed_uvqk,
            self._split_arg_list,
            dim=-1,
        )

        # query, key = self.rotary_emb(positions, query, key, self._layer_name)
        attn_output = self.attn(query, key, value)

        return attn_output, user


class PagedHSTUInferLayer(torch.nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: InferenceHSTUConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self._embedding_dim: int = config.hidden_size
        # per head dim;
        self._linear_dim_per_head: int = config.head_dim
        self._attention_dim_per_head: int = config.head_dim

        self._num_heads: int = config.num_heads

        self._eps = config.layernorm_epsilon
        self._is_causal = config.is_causal
        self._target_group_size = config.target_group_size
        self._alpha = 1.0 / (self._attention_dim_per_head ** 0.5)
        self._residual = config.residual

        dtype = (
            torch.bfloat16
            if config.bf16
            else torch.float16
            if config.fp16
            else torch.float32
        )
        device = torch_npu.npu.current_device()

        self.self_attn = HstuAttention(
            vllm_config=vllm_config,
            config=config,
            layer_idx=layer_idx
        )

        # output norm
        self._output_layernorm_weight = torch.nn.Parameter(
            torch.ones(
                self._num_heads * self._linear_dim_per_head, dtype=dtype, device=device
            ),
            requires_grad=False,
        )
        self._output_layernorm_bias = torch.nn.Parameter(
            torch.zeros(
                self._num_heads * self._linear_dim_per_head, dtype=dtype, device=device
            ),
            requires_grad=False,
        )

        # linear_proj
        self._linear_proj = torch.nn.Linear(
            self._linear_dim_per_head * self._num_heads,
            self._embedding_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )

        for param in self._linear_proj.parameters():
            param.requires_grad = False
            param.copy_(torch.randn_like(param))

        # ffn
        self.has_ffn = config.has_ffn
        if config.has_ffn:
            self.norm_ffn = RMSNorm_npu(self._embedding_dim, self._eps)
            self.feed_forward = FFN_npu_swiglu(
                dim=self._embedding_dim,
                hidden_dim=self._embedding_dim,
                dropout=config.dropout_ratio,
            )

    @torch.inference_mode()
    def forward(
        self,
        layer_input: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        attn_output, user = self.self_attn(layer_input, position)

        attn_output = attn_output.view(
            -1, self._num_heads * self._linear_dim_per_head
        )
        parallel_input = user * F.layer_norm(
            attn_output,
            normalized_shape=[self._num_heads * self._linear_dim_per_head],
            weight=self._output_layernorm_weight,
            bias=self._output_layernorm_bias,
            eps=self._eps,
        )

        layer_output = self._linear_proj(parallel_input)

        if self.has_ffn:
            ffn_input, _ = self.norm_ffn(layer_output, layer_input if self._residual else None)
            layer_output = self.feed_forward(ffn_input) + layer_output
        else:
            if self._residual:
                torch.add(layer_output, layer_input, out=layer_output)

        return layer_output


class InferenceRankingGR(torch.nn.Module):
    """
    A class representing the ranking model inference.

    Args:
        hstu_config (InferenceHSTUConfig): The HSTU configuration.
        task_config (RankingConfig): The ranking task configuration.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        hstu_config: InferenceHSTUConfig,
        task_config: RankingConfig,
    ):
        super().__init__()
        self._device = torch_npu.npu.current_device()
        self._hstu_config = hstu_config
        self._task_config = task_config

        self._embedding_dim = hstu_config.hidden_size
        for ebc_config in task_config.embedding_configs:
            assert (
                ebc_config.dim == self._embedding_dim
            ), "hstu layer hidden size should equal to embedding dim"

        self._logit_dim_list = [
            layer_sizes[-1] for layer_sizes in task_config.prediction_head_arch
        ]

        self._embedding_collection = InferenceEmbedding(task_config.embedding_configs)
        # temporary using a non-sharing GPU embedding
        self._embedding_collection.to_empty(device=torch_npu.npu.current_device())

        self.layers = torch.nn.ModuleList(
            [
                PagedHSTUInferLayer(vllm_config, hstu_config, layer_idx)
                for layer_idx in range(self._hstu_config.num_layers)
            ]
        )

        self._dense_module = MLP(
            self._embedding_dim,
            task_config.prediction_head_arch[0],
            task_config.prediction_head_act_type,
            task_config.prediction_head_bias,
            device=self._device,
        )

        self.layers.npu()
        self._dense_module = self._dense_module.npu()

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RankingGR: The model with bfloat16 precision.
        """
        self.layers.bfloat16()
        self._dense_module.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RankingGR: The model with half precision.
        """
        self.layers.half()
        self._dense_module.half()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
    ):
        with torch.inference_mode():
            # import ipdb
            # ipdb.set_trace()
            with record_function(f"## embeddings ##"):
                embeddings = self._embedding_collection(input_ids)
            # breakpoint()
            with record_function(f"## hstu_block ##"):
                input = embeddings

                forward_context: ForwardContext = get_forward_context()
                layer_names = list(forward_context.no_compile_layers.keys())

                # if attn_metadata is not None and attn_metadata[layer_names[0]].attn_state == AscendAttentionState.DecodeOnly:
                #     start_load_kv_by_layer_from_connector(forward_context, layer_names[0])

                for index, hstu_layer in enumerate(self.layers):
                    # decode forward前，H2D，load kv cache by layer
                    if attn_metadata is not None and attn_metadata[layer_names[index]].attn_state == AscendAttentionState.DecodeOnly:
                        # if index == 0:
                        #     continue
                        start_load_kv_by_layer_from_connector(forward_context, layer_names[index])
                    
                    input = hstu_layer(input, positions)
                    # prefill forward完成后，D2H，save kv cache
                    if attn_metadata is not None and attn_metadata[layer_names[index]].attn_state == AscendAttentionState.PrefillNoCache:
                        layer_name = layer_names[index]
                        layer = forward_context.no_compile_layers[layer_name]
                        kv_cache_attr = getattr(layer, 'kv_cache', None)
                        if kv_cache_attr is not None:
                            kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]
                            maybe_save_kv_by_layer_to_connector(layer_name, index, kv_cache_layer, attn_metadata[layer_name])

            #  L2 归一化，缩放到单位长度
            hstu_output = input / torch.linalg.norm(
                input, ord=2, dim=-1, keepdim=True
            ).clamp(min=1e-6)
        
            hstu_output = self._dense_module(hstu_output)
            # print(f" - GR hstu_output: {hstu_output}")
            return hstu_output


@support_torch_compile
class HSTUInferenceForCausalLM(torch.nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config: HSTUInferenceRankingConfig = vllm_config.model_config.hf_config
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.model = InferenceRankingGR(
            vllm_config=vllm_config,
            hstu_config=self.config.hstu_config,
            task_config=self.config.task_config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor] = None,
        attn_metadata: Union[AttentionMetadata, dict] = None,
        selected_indices: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds = None,
        **kwargs
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # TODO input_ids经过embedding处理下升维，以及切分candidate
        # TODO 多batch可以通过attn_metadata的seq_lens或者positions判断长度进行切分


        # print(f" - HSTU forward input_ids: {input_ids.shape}")
        # print(f" - HSTU forward positions: {positions.shape}")
        
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # TODO 参考vllm对于logit的计算，可以修改sampling_metadata满足生成一个token的需求
        # return self.model._dense_module(hidden_states)
        return hidden_states

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        # use random model
        if self.config.use_random_model:
            return None
        # 权重加载逻辑保持不变
        params_dict = dict(self.model.named_parameters())
        loaded_weights = set()
        for name, loaded_weight in weights:
            name = name.replace("model.", "", 1)
            if name in params_dict:
                params_dict[name].data.copy_(loaded_weight)
                loaded_weights.add("model." + name)

        for name in params_dict:
            if any(keyword in name for keyword in {"norm_ffn", "feed_forward"}):
                loaded_weights.add("model." + name)
        return loaded_weights

    def should_use_eager_mode(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata", None)
        if not attn_metadata:
            return True
        return False

def start_load_kv_by_layer_from_connector(forward_context, layer_name: str):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    
    layer_names = list(forward_context.no_compile_layers.keys())
    layer_name_index = layer_names.index(layer_name)

    with torch_npu.npu.stream(connector._onload_stream):
        connector.start_load_kv_by_layer(forward_context, layer_name)
        connector._onload_history_kv_events[layer_name_index].record(connector._onload_stream)

    connector._onload_history_kv_events[layer_name_index].wait(torch_npu.npu.current_stream())


def maybe_save_kv_by_layer_to_connector(layer_name: str, layer_index: int, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata"):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return
    
    connector = get_kv_transfer_group()

    with torch_npu.npu.stream(connector._offload_stream):
        connector.save_kv_layer(layer_name, kv_layer, attn_metadata)
        connector._offload_history_kv_events[layer_index].record(connector._offload_stream)
    
    # connector._offload_history_kv_events[layer_index].wait(torch_npu.npu.current_stream())