from copy import copy
from collections.abc import Iterable
from typing import Any, Optional, Union, Tuple
import torch
import math
import torch_npu
import torch.nn as nn
from functools import partial
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder,
    Qwen2_5OmniAudioEncoderLayer,
    Qwen2_5OmniAudioAttention
)
from transformers.modeling_outputs import BaseModelOutput
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerProcessingInfo,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings
)
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLVideoInputs
from .modeling_openpangu_vl import (
    ProjectionSingle,
    OpenPanguVisionTransformer,
)
from collections.abc import Mapping, Sequence
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.models.qwen2_audio import _get_feat_extract_output_lengths
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    PlaceholderFeaturesInfo,
)
from vllm.multimodal.processing import find_token_matches
from vllm.transformers_utils.tokenizer import decode_tokens

from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from .modeling_openpangu_vl import run_dp_sharded_mrope_vision_model
from .processor_openpangu_omni import OpenPanguOmniProcessor
from .imageprocessor_openpangu_vl import rescale_and_normalize
from vllm.version import __version__ as VLLM_VERSION

if "910" in torch.npu.get_device_name():
    NPU_ATTN_INFR = True
    print("[INFO] torch_npu detected. Using NPU fused infer attention.")
else:
    NPU_ATTN_INFR = False

logger = init_logger(__name__)


class OpenPanguOmniAudioAttention(Qwen2_5OmniAudioAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__(config)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)


def all_gather_interleave(local_tensor, hidden_size: int, tp_size: int):
    """All-gather the input tensor interleavely across model parallel group."""
    import torch.distributed as dist
    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(tp_size)]
    dist.all_gather(gathered_tensors,
                    local_tensor,
                    group=parallel_state.get_tp_group().device_group)

    gathered_tensors_split = [
        torch.split(tensor, hidden_size // tp_size, -1)
        for tensor in gathered_tensors
    ]
    ordered_tensors = [
        tensor for pair in zip(*gathered_tensors_split) for tensor in pair
    ]
    result_tensor = torch.cat(ordered_tensors, dim=-1)
    return result_tensor


class OpenPanguOmniAudioFusionAttention(nn.Module):
    """
    Qwen2.5OmniThinker flash attention module. This module inherits from `Qwen2_5OmniAudioAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = False
        self.is_causal = False

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, 3 * head * head_dim]
        seq_len, _ = qkv.shape
        if self.tp_size > 1:
            qkv = all_gather_interleave(qkv, self.qkv_proj.hidden_size,
                                        self.tp_size)

        # [s, 3 * head * head_dim] -> 3 * [s, head * head_dim]
        q, k, v = qkv.chunk(3, dim=1)

        # 3 * [s, head * head_dim]
        if self.tp_size > 1:
            splitter = partial(dist_utils.split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
            v = splitter(v)[self.tp_rank]

        # 3 * [s, head * head_dim] -> 3 * [s, head, head_dim]
        new_shape = (seq_len, self.num_heads,
                     self.head_dim)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        seq_length, all_dim = hidden_states.size()

        qkv, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = self.split_qkv(qkv)

        actual_seq_len = cu_seqlens.tolist()
        attn_output = torch_npu.npu_fusion_attention(
            query_states, key_states, value_states, self.num_heads,
            pse=None,
            atten_mask=None,
            scale=self.scaling,
            input_layout="TND",
            actual_seq_qlen=actual_seq_len,
            actual_seq_kvlen=actual_seq_len,
            pre_tockens=2147483547,
            next_tockens=0,
            keep_prob=1.0,
            inner_precise=0,
            sparse_mode=0)[0]
        attn_output = attn_output.reshape(seq_length, all_dim)
        attn_output, _ = self.out_proj(attn_output)

        return attn_output


OPENPANGU_OMNI_AUDIO_ATTENTION_CLASSES = {
    "attention": OpenPanguOmniAudioAttention,
    "npu_fusion_attention": OpenPanguOmniAudioFusionAttention
}


class OpenPanguOmniAudioEncoderLayer(Qwen2_5OmniAudioEncoderLayer):
    def __init__(self, config):
        super().__init__(config)
        _attn_selection = getattr(config, "_attn_selection", "attention")
        self.self_attn = OPENPANGU_OMNI_AUDIO_ATTENTION_CLASSES[_attn_selection](
            config)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class OpenPanguOmniAudioEncoder(Qwen2_5OmniAudioEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.audio_bos_eos_token = None
        self.layers = nn.ModuleList([OpenPanguOmniAudioEncoderLayer(
            config) for _ in range(config.encoder_layers)])


class OpenPanguOmniProcessorInfo(Qwen2_5OmniThinkerProcessingInfo):
    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_hf_processor(
        self,
        *,
        sampling_rate: Optional[int] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        size: Optional[dict[str, int]] = None,
        fps: Optional[Union[float, list[float]]] = None,
        **kwargs: object,
    ):
        if fps is not None:
            kwargs["fps"] = fps
        # if tuple(int(x.replace('rc3', '')) for x in VLLM_VERSION.split(".")) >= (0, 11, 0):
        #     return self.ctx.get_hf_processor(
        #         OpenPanguOmniProcessor,
        #         use_fast=kwargs.pop("use_fast", True),
        #         **kwargs,
        #     )
        return self.ctx.get_hf_processor(
            OpenPanguOmniProcessor,
            image_processor=self.get_image_processor(
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                size=size,
                use_fast=kwargs.get("use_fast", True),
            ),
            **kwargs,
        )


class OpenPanguOmniThinkerMultiModalProcessor(Qwen2_5OmniThinkerMultiModalProcessor):

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_processor = self.info.get_image_processor(
            **hf_processor_mm_kwargs)
        vocab = tokenizer.get_vocab()

        audio_token = processor.audio_token
        image_token = processor.image_token
        video_token = processor.video_token
        audio_bos_token = processor.audio_bos_token
        audio_eos_token = processor.audio_eos_token
        vision_bos_token = processor.vision_bos_token
        vision_eos_token = processor.vision_eos_token
        audio_token_id = vocab[audio_token]
        image_token_id = vocab[image_token]
        video_token_id = vocab[video_token]
        audio_bos_token_id = vocab[audio_bos_token]
        audio_eos_token_id = vocab[audio_eos_token]
        vision_bos_token_id = vocab[vision_bos_token]
        vision_eos_token_id = vocab[vision_eos_token]
        # if tuple(int(x.replace('rc3', '')) for x in VLLM_VERSION.split(".")) >= (0, 11, 0):
        #     out_mm_kwargs = out_mm_kwargs.get_data()
        audio_feature_lengths = out_mm_kwargs.get("audio_feature_lengths")
        feature_attention_mask = out_mm_kwargs.get("feature_attention_mask")
        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            _, audio_output_lens = _get_feat_extract_output_lengths(
                audio_feature_lengths)
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            if not isinstance(feature_attention_mask, torch.Tensor):
                raise TypeError(
                    "Expected 'feature_attention_mask' to be a Tensor")
            _, audio_output_lens = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1))
            audio_output_lengths = audio_output_lens.tolist()

        # number of audios read from video.
        audio_in_video_item_idx = 0

        def get_replacement_qwen2_audio(item_idx: int):
            item_idx += audio_in_video_item_idx

            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio = audios.get(item_idx)
                raise ValueError(
                    f"The audio {audio} (len={len(audio)}) is too short "
                    "to be represented inside the model")

            return [audio_token_id] * num_features

        def get_replacement_openpangu_vision(item_idx: int, modality: str):
            try:
                grid_thw = out_mm_kwargs[f"{modality}_grid_thw"][item_idx]
            except Exception as e:
                out_item = out_mm_kwargs[modality][item_idx]
                grid_thw = out_item[f"{modality}_grid_thw"].data
            if not isinstance(grid_thw, torch.Tensor):
                raise TypeError("Expected 'grid_thw' to be a Tensor")
            merge_length = image_processor.merge_size**2
            if modality == "image":
                image_token_id_total = [image_token_id] * \
                    (int(grid_thw.prod()) // merge_length)
                return image_token_id_total
            else:
                # When modality is video
                grid_t, grid_h, grid_w = grid_thw
                video_seq_length_per_time = (
                    grid_h * grid_w).item() // merge_length
                video_token_id_per_time = [vision_bos_token_id] + [video_token_id] * video_seq_length_per_time + \
                    [vision_eos_token_id]
                video_token_id_total = video_token_id_per_time * grid_t
                video_token_id_middle = video_token_id_total[1:-1]
                return PromptUpdateDetails.select_token_id(
                    video_token_id_middle,
                    embed_token_id=video_token_id,
                )

        hf_config = self.info.get_hf_config()
        use_audio_in_video = getattr(
            hf_config.vision_config, "use_audio_in_video", False)

        def get_replacement_openpangu_use_audio_in_video(item_idx: int):
            audio_len = audio_output_lengths[item_idx]
            video_grid_thw = out_mm_kwargs["video_grid_thw"][item_idx]
            merge_length = image_processor.merge_size**2

            grid_t = video_grid_thw[0]
            grid_h = video_grid_thw[1]
            grid_w = video_grid_thw[2]
            video_seq_length_per_time = (
                grid_h * grid_w).item() // merge_length
            video_token_id_per_time = (
                [vision_bos_token_id]
                + [video_token_id] * video_seq_length_per_time
                + [vision_eos_token_id]
            )
            audio_seq_length_per_time = audio_len // grid_t
            audio_seq_length_last = audio_len % grid_t
            audio_token_id_per_time = (
                [audio_bos_token_id]
                + [audio_token_id] * audio_seq_length_per_time
                + [audio_eos_token_id]
            )
            audio_token_id_last = (
                [audio_bos_token_id]
                + [audio_token_id] *
                (audio_seq_length_per_time + audio_seq_length_last)
                + [audio_eos_token_id]
            )
            audio_in_video_token_id = (
                (video_token_id_per_time + audio_token_id_per_time) * (grid_t - 1)
                + video_token_id_per_time
                + audio_token_id_last
            )
            placeholder = audio_in_video_token_id[1:-1]

            return PromptUpdateDetails.select_token_id(
                placeholder,
                embed_token_id=video_token_id
            )

        video_replacement_fn = (
            get_replacement_openpangu_use_audio_in_video if use_audio_in_video else
            partial(get_replacement_openpangu_vision, modality="video"))

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            ),
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=partial(get_replacement_openpangu_vision,
                                    modality="image"),
            ),
            PromptReplacement(
                modality="video",
                target=video_token,
                replacement=video_replacement_fn,
            ),
        ]

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargs,
        is_update_applied: bool,
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """
        OpenPanguOmni reimplements this function to handle `use_audio_in_video`.
        """
        unbound_prompt_updates = self._get_prompt_updates(
            mm_items,
            hf_processor_mm_kwargs,
            mm_kwargs,
        )
        mm_prompt_updates = self._bind_and_group_updates(
            unbound_prompt_updates)

        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        hf_config = self.info.get_hf_config()
        use_audio_in_video = getattr(
            hf_config.vision_config, "use_audio_in_video", False)
        if is_update_applied:
            mm_placeholders = self._find_mm_placeholders(
                mm_prompt_updates,
                prompt_ids,
                mm_item_counts,
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts
            )

            tokenizer = self.info.get_tokenizer()
            prompt = decode_tokens(tokenizer, prompt_ids)
        else:
            if use_audio_in_video and "audio" in mm_prompt_updates:
                # Remove audio from prompt updates (it won't match anything)
                filtered_updates = {
                    k: v for k, v in mm_prompt_updates.items() if k != "audio"
                }
                filtered_mm_item_counts = mm_item_counts.copy()
                filtered_mm_item_counts['audio'] -= filtered_mm_item_counts['video']
                (
                    prompt_ids,
                    prompt,
                    mm_placeholders,
                ) = self._apply_prompt_updates(
                    prompt_ids,
                    filtered_updates,
                    filtered_mm_item_counts
                )
                mm_placeholders = self._derive_audio_from_video_placeholders(
                    mm_placeholders, mm_item_counts
                )
            else:
                (
                    prompt_ids,
                    prompt,
                    mm_placeholders,
                ) = self._apply_prompt_updates(
                    prompt_ids,
                    mm_prompt_updates,
                    mm_item_counts
                )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts
            )

        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        audio_token = processor.audio_token
        vision_eos_token = processor.vision_eos_token
        audio_eos_token = processor.audio_eos_token
        audio_token_id = vocab[audio_token]
        vision_eos_token_id = vocab[vision_eos_token]
        audio_eos_token_id = vocab[audio_eos_token]
        if use_audio_in_video:
            for i in range(1, len(prompt_ids)):
                if prompt_ids[i - 1] == audio_token_id and prompt_ids[i] == vision_eos_token_id:
                    prompt_ids[i] = audio_eos_token_id
        prompt = decode_tokens(tokenizer, prompt_ids)

        if use_audio_in_video:
            mm_kwargs["use_audio_in_video"] = True

        return prompt_ids, prompt, mm_placeholders

    def _validate_mm_placeholders(
        self,
        mm_placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_item_counts: Mapping[str, int],
    ) -> None:
        super()._validate_mm_placeholders(mm_placeholders, mm_item_counts)

    def _derive_audio_from_video_placeholders(
        self,
        placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_item_counts: Mapping[str, int],
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        """
        Helper to derive audio placeholders from video placeholders when
        use_audio_in_video=True.
        """
        if "video" not in placeholders:
            return placeholders

        # Validate audio and video counts match
        num_videos = len(placeholders["video"])
        num_audios = mm_item_counts["audio"]
        if num_audios != num_videos:
            raise ValueError(
                f"use_audio_in_video requires equal number of audio and video items, "
                f"got {num_audios=}, {num_videos=}"
            )

        tokenizer = self.info.get_tokenizer()
        processor = self.info.get_hf_processor()
        vocab = tokenizer.get_vocab()
        audio_token = processor.audio_token
        audio_token_id = vocab[audio_token]

        result_placeholders = dict(placeholders)
        audio_placeholders = []

        # Each video is paired with one audio
        for video_idx, video_placeholder in enumerate(placeholders["video"]):
            # Create is_embed mask selecting only audio tokens
            audio_is_embed = torch.tensor(
                video_placeholder.tokens) == audio_token_id

            audio_placeholder = PlaceholderFeaturesInfo(
                modality="audio",
                item_idx=video_idx,
                start_idx=video_placeholder.start_idx,
                tokens=video_placeholder.tokens,
                is_embed=audio_is_embed,
            )
            audio_placeholders.append(audio_placeholder)

        result_placeholders["audio"] = audio_placeholders
        return result_placeholders


@MULTIMODAL_REGISTRY.register_processor(
    OpenPanguOmniThinkerMultiModalProcessor,
    info=OpenPanguOmniProcessorInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder,
)
class OpenPanguOmniForConditionalGeneration(Qwen2_5OmniThinkerForConditionalGeneration):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen2_5OmniThinkerForConditionalGeneration, self).__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.use_data_parallel = getattr(
            vllm_config.parallel_config, 'enable_multimodal_encoder_data_parallel', False)

        self.audio_tower = OpenPanguOmniAudioEncoder(self.config.audio_config)
        self.visual = OpenPanguVisionTransformer(
            vision_config=self.config.vision_config,
            norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "visual"),
            use_data_parallel=self.use_data_parallel
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix("pangu_V5", "language_model"),
            architectures=["PanguEmbeddedForCausalLM"],
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self.visual.vision_projection = ProjectionSingle(
            self.config.vision_config.out_hidden_size,
            vllm_config.model_config.hf_config.hidden_size,
        )
        self._parse_preprocess_params(self.config.vision_config)

    def _parse_preprocess_params(self, vision_config):
        self.channel = vision_config.in_channels
        self.patch_size = vision_config.patch_size
        from vllm.multimodal import MULTIMODAL_REGISTRY
        processor = MULTIMODAL_REGISTRY.create_processor(
            self.vllm_config.model_config)
        self.do_rescale = processor.info.get_hf_processor().image_processor.do_rescale
        self.rescale_factor = processor.info.get_hf_processor().image_processor.rescale_factor
        self.do_normalize = processor.info.get_hf_processor().image_processor.do_normalize
        self.image_mean = tuple(
            processor.info.get_hf_processor().image_processor.image_mean)
        self.image_std = tuple(
            processor.info.get_hf_processor().image_processor.image_std)

    def _process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        if grid_thw.ndim != 2:
            raise ValueError(
                f"grid_thw.ndim must be 2, but it is {grid_thw.ndim}")

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            # rescale and normalize
            pixel_values = pixel_values.reshape(
                -1, self.channel, self.patch_size, self.patch_size)
            pixel_values = rescale_and_normalize(pixel_values, self.do_rescale, self.rescale_factor, self.do_normalize,
                                                 self.image_mean, self.image_std)
            pixel_values = pixel_values.reshape(
                -1, self.channel * self.patch_size * self.patch_size)
            if self.use_data_parallel:
                image_embeds = run_dp_sharded_mrope_vision_model(
                    self.visual, pixel_values, grid_thw, rope_type="rope_3d"
                )
            else:
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

            image_embeds = self.visual.vision_projection(image_embeds)
        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return image_embeds.split(sizes.tolist())

    def _process_video_input(
        self,
        video_input: Qwen2_5_VLVideoInputs,
        video_hashes: list[str] = None,
        cached_video_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        grid_thw = video_input["video_grid_thw"]
        if grid_thw.ndim != 2:
            raise ValueError(
                f"grid_thw.ndim must be 2, but it is {grid_thw.ndim}")

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype)
            if self.use_data_parallel:
                video_embeds = run_dp_sharded_mrope_vision_model(
                    self.visual, pixel_values_videos, grid_thw, rope_type="rope_3d"
                )
            else:
                video_embeds = self.visual(
                    pixel_values_videos, grid_thw=grid_thw)

            video_embeds = self.visual.vision_projection(video_embeds)
        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:

            # TODO (ywang96): support overlapping modalitiy embeddings so that
            # `use_audio_in_video` will work on V1.
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, [
                    self.config.image_token_id,
                    self.config.video_token_id,
                    self.config.audio_token_id
                ])
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is None:
            return inputs_embeds

        for embeddings, modality in multimodal_embeddings:
            if modality == "audio":
                placeholder_token_id = self.config.audio_token_id
            if modality == "image":
                placeholder_token_id = self.config.image_token_id
            if modality == "video":
                placeholder_token_id = self.config.video_token_id
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, embeddings, placeholder_token_id)
        return inputs_embeds

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "[unused18][unused19][unused20]"
        if modality.startswith("video"):
            return "[unused18][unused32][unused20]"
        if modality.startswith("audio"):
            return "[unused28][unused29][unused30]"

        raise ValueError("Only image, video or audio modality is supported")
