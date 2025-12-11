import os
from io import BytesIO
from urllib.parse import urlparse
from typing import Optional, Tuple, Dict, Union, Literal
import numpy.typing as npt
from pathlib import Path
import base64
from vllm.utils import PlaceholderModule
try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

ModalityStr = Literal["image", "audio", "video", "image_embeds"]


def _placeholder_str_add_pangu(self, modality: ModalityStr,
                               current_count: int) -> Optional[str]:
    # TODO: Let user specify how to insert image tokens into prompt
    # (similar to chat template)
    hf_config = self._model_config.hf_config
    model_type = hf_config.model_type

    if modality in ("image", "image_embeds"):
        if model_type == "chatglm":
            return "<|begin_of_image|><|endoftext|><|end_of_image|>"
        if model_type in ("phi3_v", "phi4mm"):
            return f"<|image_{current_count}|>"
        if model_type in ("minicpmo", "minicpmv"):
            return "(<image>./</image>)"
        if model_type in ("blip-2", "florence2", "fuyu", "paligemma",
                          "pixtral", "mistral3"):
            # These models do not use image tokens in the prompt
            return None
        if model_type == "qwen":
            return f"Picture {current_count}: <img></img>"
        if model_type.startswith("llava"):
            return self._cached_token_str(self._tokenizer,
                                          hf_config.image_token_index)

        if model_type in ("aya_vision", "chameleon", "deepseek_vl_v2",
                          "internvl_chat", "ovis", "skywork_chat",
                          "NVLM_D", "h2ovl_chat", "idefics3", "smolvlm"):
            return "<image>"
        if model_type in ("mllama", "llama4"):
            return "<|image|>"
        if model_type in ("qwen2_vl", "qwen2_5_vl"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if model_type == "qwen2_5_omni":
            return "<|vision_start|><|IMAGE|><|vision_end|>"
        if model_type == "molmo":
            return ""
        if model_type == "aria":
            return "<|fim_prefix|><|img|><|fim_suffix|>"
        if model_type == "gemma3":
            return "<start_of_image>"
        if model_type == "kimi_vl":
            return "<|media_start|>image<|media_content|><|media_pad|><|media_end|>"  # noqa: E501
        if model_type == "pangu_v5_vl":
            return ""
        if model_type == "openpangu_vl":
            return ""
        if model_type == "openpangu_omni":
            return ""
        if model_type == "pangu_v5_omni":
            return ""

        raise TypeError(f"Unknown {modality} model type: {model_type}")
    elif modality == "audio":
        if model_type in ("ultravox", "granite_speech"):
            return "<|audio|>"
        if model_type == "phi4mm":
            return f"<|audio_{current_count}|>"
        if model_type in ("qwen2_audio", "qwen2_5_omni"):
            return (f"Audio {current_count}: "
                    f"<|audio_bos|><|AUDIO|><|audio_eos|>")
        if model_type == "minicpmo":
            return "(<audio>./</audio>)"
        if model_type == "pangu_v5_omni":
            return ""
        if model_type == "openpangu_omni":
            return ""
        raise TypeError(f"Unknown model type: {model_type}")
    elif modality == "video":
        if model_type == "internvl_chat":
            return "<video>"
        if model_type in ("qwen2_vl", "qwen2_5_vl"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if model_type == "qwen2_5_omni":
            return "<|vision_start|><|VIDEO|><|vision_end|>"
        if model_type in ("minicpmo", "minicpmv"):
            return "(<video>./</video>)"
        if model_type == "pangu_v5_vl":
            return ""
        if model_type == "openpangu_vl":
            return ""
        if model_type == "pangu_v5_omni":
            return ""
        if model_type == "openpangu_omni":
            return ""
        if model_type.startswith("llava"):
            return self._cached_token_str(self._tokenizer,
                                          hf_config.video_token_index)
        raise TypeError(f"Unknown {modality} model type: {model_type}")
    else:
        raise TypeError(f"Unknown modality: {modality}")


def parse_video(self, video_url: str) -> None:
    num_frames = self._tracker._model_config.hf_image_processor_config.get(
        "num_frames", 32)
    sample_fps = self._tracker._model_config.hf_image_processor_config.get(
        "sample_fps", 1)
    sampling_rate = self._tracker._model_config.hf_image_processor_config.get(
        "sampling_rate", 16000)
    use_audio_in_video = getattr(
        self._tracker._model_config.hf_config.vision_config, "use_audio_in_video", False)
    video = self._connector.fetch_video_async(
        video_url, num_frames=num_frames, sample_fps=sample_fps)
    if use_audio_in_video:
        import ffmpeg
        import tempfile
        allowed_local_media_path = self._connector.allowed_local_media_path
        video_input, kwargs = get_ffmpeg_input(
            video_url, allowed_local_media_path)
        # Execute FFmpeg
        if "input" in kwargs:
            # When video_url is base64 string.
            input_format = kwargs.pop("format", None)
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{input_format}") as tmp:
                    tmp.write(kwargs["input"])
                    tmp_path = tmp.name
                audio_bytes, _ = (
                    ffmpeg
                    .input(tmp_path)
                    .output("pipe:", format="wav", acodec="pcm_s16le", ar=str(sampling_rate), vn=None)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            finally:
                if tmp_path is not None and os.path.exists(tmp_path):
                    os.unlink(tmp_path)  # Clean temporary files
        else:
            # When video_url is local path or HTTP/HTTPS path.
            audio_bytes, _ = (
                ffmpeg.input(video_input)
                .output("pipe:", format="wav", acodec="pcm_s16le", ar=str(sampling_rate), vn=None)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        audio = audio_load_bytes(audio_bytes, sampling_rate)
        self._tracker.add("audio", audio)
    video_placeholder = self._tracker.add("video", video)
    self._add_placeholder(video_placeholder)


def get_ffmpeg_input(video_url: str, allowed_local_media_path: str) -> Tuple[str, Dict[str, Union[bytes, str]]]:
    """返回 (input_arg, run_kwargs)"""
    parsed = urlparse(video_url)
    if parsed.scheme == "data":
        # Data URL: Decode and use pipe.
        data_spec, data = parsed.path.split(",", 1)
        media_type, data_type = data_spec.split(";", 1)
        media_spec, video_format = media_type.split("/", 1)
        video_bytes = base64.b64decode(data)
        return "pipe:", {"input": video_bytes, "format": video_format}
    elif parsed.scheme == "file":
        if allowed_local_media_path is None:
            raise RuntimeError("Cannot load local files without "
                               "`--allowed-local-media-path`.")
        filepath = Path(parsed.path)
        if allowed_local_media_path not in filepath.resolve().parents:
            raise ValueError(
                f"The file path {filepath} must be a subpath "
                f"of `--allowed-local-media-path` {allowed_local_media_path}.")
        return str(Path(parsed.path)), {}
    else:
        # For HTTP/HTTPS or a regular path string, send directly.
        return video_url, {}


async def audio_load_bytes(data: bytes, sampling_rate: int) -> tuple[npt.NDArray, float]:
    return librosa.load(BytesIO(data), sr=sampling_rate)
