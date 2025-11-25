# SPDX-License-Identifier: Apache-2.0
import dataclasses
from enum import Enum, unique
from typing import Any, Dict, Iterable, List, Optional, Union, cast, Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import PretrainedConfig

@dataclass
class InferenceEmbeddingConfig:
    """
    Configuration for inference embeddings with dynamic option.
    Args:
        feature_names (List[str]): The name of the features in this embedding.
        table_name (str): The name of the table.
        vocab_size (int): The size of the vocabulary.
        dim (int): The dimension size of the embeddings.
        use_dynamicemb (bool): The option for dynamic embedding.
    """

    feature_names: List[str]
    table_name: str
    vocab_size: int
    dim: int
    use_dynamicemb: bool


@dataclass
class PositionEncodingConfig:
    """
    Configuration data class for position encoding settings.

    Args:
      num_position_buckets: The number of buckets used for position encoding.
      num_time_buckets: The number of buckets used for time encoding.
      use_time_encoding: A boolean flag indicating whether time encoding should be used.

    """

    num_position_buckets: int
    num_time_buckets: int
    use_time_encoding: bool


@dataclass
class HSTUPreprocessingConfig:
    item_embedding_dim: int
    contextual_embedding_dim: int


@unique
class HSTULayerType(Enum):
    """
    Enum class representing different HSTU layer types.

    Attributes:
      FUSED: Represents the fused type. The fused layer is scheduleable and pipelineable. Does not support TP. This will be deprecated in the future.
      NATIVE: Represents the non-fused type. Support TP.
      DEBUG: Represents the debug type. This does not support TP and is used for debugging.
    """

    FUSED = "FUSED"
    NATIVE = "NATIVE"
    DEBUG = "DEBUG"


@unique
class KernelBackend(Enum):
    """
    Enum class representing different kernel backends.

    Attributes:
      TRITON: Represents the TRITON backend.
      PYTORCH: Represents the PYTORCH backend.
      CUTLASS: Represents the CUTLASS backend.
    """

    TRITON = "TRITON"
    PYTORCH = "PYTORCH"
    CUTLASS = "CUTLASS"


@dataclass
class HSTUConfig():
    """
    HSTUConfig is a configuration data class for the HSTU model, inheriting from TransformerConfig.

    Args:
      hstu_preprocessing_config (HSTUPreprocessingConfig): HSTU preprocessing config. Defaults to None.
      position_encoding_config (PositionEncodingConfig): Position embedding config. Defaults to None.
      is_causal (bool): Indicates if the model is causal. Defaults to True.
      enable_relative_attention_bias (bool): Flag to enable relative attention bias. Defaults to False.
      kernel_backend (KernelBackend): Backend for kernel operations. Defaults to KernelBackend.CUTLASS.
      target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
      learnable_input_layernorm (bool): Flag to enable learnable input layernorm. Defaults to True.
      residual (bool): Flag to enable residual connection. Defaults to True.
      async_wgrad (bool): Flag to enable async wgrad. Defaults to False.
      async_wgrad_stream (torch.cuda.Stream): Stream for async wgrad. Defaults to None.
      async_wgrad_event (torch.cuda.Event): Event for async wgrad. Defaults to None.
      recompute_input_layernorm (bool): Flag to enable recompute input layernorm. Defaults to False.
      recompute_input_silu (bool): Flag to enable recompute input silu. Defaults to False.
    """

    hstu_preprocessing_config: Optional[HSTUPreprocessingConfig] = None
    position_encoding_config: Optional[PositionEncodingConfig] = None
    is_causal: bool = True
    enable_relative_attention_bias: bool = False

    kernel_backend: KernelBackend = KernelBackend.CUTLASS
    # TODO deprecate FUSED
    hstu_layer_type: HSTULayerType = HSTULayerType.FUSED

    target_group_size: int = 1
    learnable_input_layernorm: bool = True
    # whether to add residual connection
    residual: bool = True
    # whether to use async wgrad
    async_wgrad: bool = False
    async_wgrad_stream: Optional[torch.cuda.Stream] = None
    async_wgrad_event: Optional[torch.cuda.Event] = None
    # whether to recompute input layernorm
    recompute_input_layernorm: bool = False
    recompute_input_silu: bool = False
    # whether is the inference mode
    is_inference: bool = False
    add_uvqk_bias: bool = True
    fuse_norm_mul_dropout: bool = True

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ShardedEmbeddingConfig:
    """
    Configuration for sharded embeddings with sharding type. Inherits from BaseShardedEmbeddingConfig.

    Args:
        config (EmbeddingConfig): The embedding configuration.
        sharding_type (str): The type of sharding, ``'data_parallel'`` | ``'model_parallel'``.
    """

    """
    Base configuration for sharded embeddings.

    Args:
        feature_names (List[str]): The name of the features in this embedding.
        table_name (str): The name of the table.
        vocab_size (int): The size of the vocabulary.
        dim (int): The dimension size of the embeddings.
        sharding_type (str): The type of sharding, ``'data_parallel'`` | ``'model_parallel'``.
    """

    feature_names: List[str]
    table_name: str
    vocab_size: int
    dim: int
    sharding_type: str

    def __post_init__(self):
        assert self.sharding_type in [
            "data_parallel",
            "model_parallel",
        ], "sharding type should be data_parallel or model_parallel"


@dataclass
class BaseTaskConfig:
    """
    Base configuration for tasks.

    Args:
        embedding_configs (List[ShardedEmbeddingConfig]): A list of embedding configurations.
        user_embedding_norm (str, optional): Normalization for user embeddings. ``'layer_norm'`` | ``'l2_norm'``. Defaults to ``'l2_norm'``.
        item_l2_norm (bool, optional): Whether to apply L2 normalization to item embeddings. Defaults to False.
    """

    embedding_configs: List[InferenceEmbeddingConfig]

    user_embedding_norm: str = "l2_norm"
    item_l2_norm: bool = False

    def __post_init__(self):
        table_names = [emb_config.table_name for emb_config in self.embedding_configs]
        assert len(set(table_names)) == len(
            table_names
        ), f"duplicate table_names in embedding {table_names}"


@dataclass
class RankingConfig(BaseTaskConfig):
    """
    Configuration for ranking tasks.

    Args:
        prediction_head_arch (List[int]): Architecture of the prediction head.
        prediction_head_act_type (str): Activation function type for the prediction head layers. Must be one of: ``'relu'`` | ``'gelu'``. Defaults to ``'relu'``.
        prediction_head_bias (bool): Whether to use bias terms in the prediction head layers. Defaults to ``True``.
        num_tasks (int): Number of tasks. Defaults to ``1``.
        eval_metrics (Tuple[str], optional): Tuple of evaluation metric type str during training. Refer to :obj:`~modules.metrics.metric_modules.MetricType`
          for available metrics. Defaults to ``'AUC'``.
    """

    prediction_head_arch: List[int] = cast(List[int], None)
    prediction_head_act_type: str = "relu"
    prediction_head_bias: bool = True
    num_tasks: int = 1
    eval_metrics: Tuple[str, ...] = ("AUC",)

    def __post_init__(self):
        assert (
            self.prediction_head_arch is not None
        ), "Please provide prediction head arch"
        assert isinstance(
            self.prediction_head_arch, list
        ), "prediction_head_arch should be a list"
        assert isinstance(
            self.prediction_head_act_type, str
        ), "prediction_head_act_type should be a str"
        assert isinstance(
            self.prediction_head_bias, bool
        ), "prediction_head_bias should be a bool"


@dataclass
class InferenceHSTUConfig:
    """
    InferenceHSTUConfig is a configuration data class for the inference HSTU model.

    Args:
        hidden_size (int): The hidden states dimension size.
        num_layers (int): Number of attention layers.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of key-value channels (per attention head).
        layernorm_epsilon (float): Epsilon value for normalization.
        bf16 (bool): Whether to inference in bfloat16.
        fp16 (bool): Whether to inference in float16.

        learnable_input_layernorm (bool): Whether to have input layernorm weights.
        residual (bool): Whether to add residual connection.
        is_causal (bool):Whether the attention is causal.
        target_group_size (int):  The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking).
        position_encoding_config (PositionEncodingConfig, optional): Position embedding config.
    """

    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    layernorm_epsilon: float = 1e-5
    bf16: bool = True
    fp16: bool = False

    learnable_input_layernorm: bool = True
    residual: bool = True
    is_causal: bool = True
    target_group_size: int = 1
    position_encoding_config: Optional[PositionEncodingConfig] = None
    hstu_preprocessing_config: Optional[HSTUPreprocessingConfig] = None
    has_ffn: Optional[bool] = False
    dropout_ratio: Optional[float] = 0.0
    rab: Optional[bool] = False

    def __post_init__(self):
        assert self.is_causal
        assert self.target_group_size == 1

def get_inference_hstu_config(
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    head_dim: int,
    norm_epsilon=1e-5,
    dtype: torch.dtype = torch.bfloat16,
    learnable_input_layernorm: bool = True,
    residual: bool = True,
    is_causal: bool = True,
    target_group_size: int = 1,
    position_encoding_config: Optional[PositionEncodingConfig] = None,
    has_ffn: Optional[bool] = False,
    dropout_ratio: Optional[float] = 0.0,
    rab: Optional[bool] = False,
) -> InferenceHSTUConfig:
    """
    Create the HSTU configuration.

    Args:
        hidden_size (int): The hidden dimension size.
        num_layers (int): Number of attention layers.
        num_attention_heads (int): Number of attention heads.
        head_dim (int): Number of key-value channels (per attention head).
        norm_epsilon (float, optional): Epsilon value for normalization. Defaults to 1e-5.
        dtype (torch.dtype): Data type (e.g., torch.float16).
        learnable_input_layernorm (bool, optional): Whether to have input layernorm weights. Defaults to True.
        residual (bool, optional): Whether to add residual connection. Defaults to True.
        is_causal (bool, optional): Whether the attention is causal. Defaults to False.
        target_group_size (int, optional): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
        position_encoding_config (Optional[PositionEncodingConfig], optional): Position embedding config. Defaults to None.
    Returns:
        HSTUConfig: The HSTU configuration object.
    """
    is_bf16 = dtype == torch.bfloat16
    is_fp16 = dtype == torch.float16
    return InferenceHSTUConfig(  # type: ignore
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_attention_heads,
        head_dim=head_dim,
        layernorm_epsilon=norm_epsilon,
        bf16=is_bf16,
        fp16=is_fp16,
        learnable_input_layernorm=learnable_input_layernorm,
        residual=residual,
        is_causal=is_causal,
        target_group_size=target_group_size,
        position_encoding_config=position_encoding_config,
        has_ffn=has_ffn,
        dropout_ratio=dropout_ratio,
        rab=rab,
    )


class HSTUInferenceRankingConfig(PretrainedConfig):
    model_type = "hstu_inference_ranking"
    hstu_config: InferenceHSTUConfig
    task_config: RankingConfig

    feature_to_max_seqlen: Dict[str, int]
    contextual_feature_names: List[str]
    item_feature_name: str
    action_feature_name: Optional[str]
    max_num_candidates: int
    use_random_model: bool
    hidden_size: int
    vocab_size: int

    def __init__(self,
                 hstu_config: Optional[Dict[str, Any]] = None,
                 task_config: Optional[Dict[str, Any]] = None,
                 feature_to_max_seqlen: Optional[Dict[str, int]] = None,
                 contextual_feature_names: Optional[List[str]] = None,
                 item_feature_name: Optional[str] = None,
                 action_feature_name: Optional[str] = None,
                 max_num_candidates: Optional[int] = None,
                 max_num_users: Optional[int] = None,
                 use_random_model: Optional[bool] = None,
                 hidden_size: Optional[int] = None,
                 vocab_size: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        if hstu_config is None:
            hstu_config_dict = {
                "hidden_size": 1024,
                "num_layers": 8,
                "num_attention_heads": 4,
                "head_dim": 256,
            }
        else:
            hstu_config_dict = hstu_config
        dtype_str = hstu_config_dict.get("dtype")
        if dtype_str == "bfloat16":
            hstu_config_dict["dtype"] = torch.bfloat16
        elif dtype_str == "float16":
            hstu_config_dict["dtype"] = torch.float16
        elif dtype_str == "float32":
            hstu_config_dict["dtype"] = torch.float32
        elif dtype_str is not None:
            raise ValueError(f"Unsupported dtype string '{dtype_str}' in config.json")
        self.hstu_config = get_inference_hstu_config(**hstu_config_dict)

        if task_config is None:
            task_config_dict = {
                "embedding_configs": [], # 默认可以为空列表
                "prediction_head_arch": [[128, 1]] # 至少需要一个任务的arch
            }
        else:
            task_config_dict = task_config.copy()
        embedding_configs_objects = [
            InferenceEmbeddingConfig(**config_dict) for config_dict in task_config_dict.get("embedding_configs", [])
        ]
        task_config_dict["embedding_configs"] = embedding_configs_objects
        self.task_config = RankingConfig(**task_config_dict)

        self.feature_to_max_seqlen = feature_to_max_seqlen
        self.contextual_feature_names = contextual_feature_names
        self.item_feature_name = item_feature_name
        self.action_feature_name = action_feature_name
        self.max_num_users = max_num_users
        self.max_num_candidates = max_num_candidates
        self.use_random_model = use_random_model
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["hstu_config"] = dataclasses.asdict(self.hstu_config)
        output["task_config"] = dataclasses.asdict(self.task_config)

        if "dtype" in output["hstu_config"] and isinstance(output["hstu_config"]["dtype"], torch.dtype):
             output["hstu_config"]["dtype"] = str(output["hstu_config"]["dtype"]).replace("torch.", "")

        output["feature_to_max_seqlen"] = self.feature_to_max_seqlen
        output["contextual_feature_names"] = self.contextual_feature_names
        output["item_feature_name"] = self.item_feature_name
        output["action_feature_name"] = self.action_feature_name
        output["max_num_users"] = self.max_num_users
        output["max_num_candidates"] = self.max_num_candidates
        return output