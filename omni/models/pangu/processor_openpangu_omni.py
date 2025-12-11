#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The HuggingFace Inc. team
# Adapted from transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py
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

from typing import Any
import re
from transformers.utils import logging
from transformers import AutoTokenizer
import numpy as np
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor

logger = logging.get_logger(__name__)


class OpenPanguOmniProcessor(Qwen2_5OmniProcessor):
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("AutoTokenizer")

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.tokenizer=tokenizer
        self.audio_bos_token = "[unused28]"
        self.audio_token = "[unused29]"
        self.audio_eos_token = "[unused30]"
        self.video_token = "[unused32]"
        self.vision_bos_token = "[unused18]"
        self.image_token = "[unused19]"
        self.vision_eos_token = "[unused20]"

        self.audio_token_id = self.get_token_id("audio_token_id", self.audio_token)
        self.image_token_id = self.get_token_id("image_token_id", self.image_token)
        self.video_token_id = self.get_token_id("video_token_id", self.video_token)
        self.audio_bos_token_id = self.get_token_id("audio_bos_token_id", self.audio_bos_token)
        self.audio_eos_token_id = self.get_token_id("audio_eos_token_id", self.audio_eos_token)
        self.vision_bos_token_id = self.get_token_id("vision_bos_token_id", self.vision_bos_token)
        self.vision_eos_token_id = self.get_token_id("vision_eos_token_id", self.vision_eos_token)

        self.image_processor = image_processor
        self.video_processor = video_processor
        self.feature_extractor = feature_extractor
        self.chat_template = chat_template

    def get_token_id(self, str_token_id, token):
        if getattr(self.tokenizer, str_token_id, None) is not None:
            return getattr(self.tokenizer, str_token_id)
        if hasattr(self.tokenizer, "video_token_id"):
            return self.tokenizer.video_token_id
        return self.tokenizer.convert_tokens_to_ids(token)

    def replace_multimodal_special_tokens(
        self,
        text,
        audio_lengths,
        image_grid_thw,
        video_grid_thw,
        video_second_per_grid,
        use_audio_in_video,
        position_id_per_seconds,
        seconds_per_chunk,
    ):
        # Extend mm token length
        merge_length_image = self.image_processor.merge_size**2
        merge_length_video = self.video_processor.merge_size**2

        processed_text = []
        for sample in text:
            positions = []
            special_tokens = [re.escape(tok) for tok in [self.audio_token, self.image_token, self.video_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])

            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * next(audio_lengths), 1)
                elif special_token == self.image_token:
                    image_seq_length = next(image_grid_thw).prod() // merge_length_image
                    sample = sample.replace(self.image_token, "<|image_placeholder|>" * image_seq_length, 1)
                elif special_token == self.video_token:
                    if not use_audio_in_video:
                        curr_video_grid_thw = next(video_grid_thw)
                        grid_t = curr_video_grid_thw[0]
                        grid_h = curr_video_grid_thw[1]
                        grid_w = curr_video_grid_thw[2]
                        video_seq_length_per_time = (grid_h * grid_w).item() // merge_length_video
                        placeholder_string = (
                            self.vision_bos_token
                            + "<|video_placeholder|>" * video_seq_length_per_time
                            + self.vision_eos_token
                        )
                        placeholder_string = placeholder_string * grid_t
                        placeholder_string = placeholder_string.removeprefix(self.vision_bos_token)
                        placeholder_string = placeholder_string.removesuffix(self.vision_eos_token)
                        sample = sample.replace(self.video_token, placeholder_string, 1)
                    else:
                        curr_video_grid_thw = next(video_grid_thw)
                        # Get the number of video frames (time steps).
                        grid_t = curr_video_grid_thw[0]
                        grid_h = curr_video_grid_thw[1]
                        grid_w = curr_video_grid_thw[2]
                        video_seq_length_per_time = (grid_h * grid_w).item() // merge_length_video

                        # Calculate the number of audio tokens corresponding to each frame (round up or down;
                        # here, we use integer division plus remainder for allocation).
                        audio_seqlen = next(audio_lengths)
                        # Allocation strategy: The first T-1 frames are evenly distributed,
                        # and the last frame takes up the remainder.
                        base_audio_per_frame = audio_seqlen // grid_t
                        remainder = audio_seqlen % grid_t
                        audio_tokens_per_frame_list = (
                            [base_audio_per_frame] * (grid_t - 1)
                            + [base_audio_per_frame + remainder]
                        )

                        mix_placeholder_string = []
                        # Build the placeholder string for each frame.
                        for t in range(grid_t):
                            placeholder_string_per_frame = (
                                self.vision_bos_token +
                                "<|video_placeholder|>" * video_seq_length_per_time +
                                self.vision_eos_token +
                                self.audio_bos_token +
                                "<|audio_placeholder|>" * audio_tokens_per_frame_list[t] +
                                self.audio_eos_token
                            )
                            mix_placeholder_string.append(placeholder_string_per_frame)
                        placeholder_string = "".join(mix_placeholder_string)
                        # Replace the video tags in the original template (only once).
                        sample = sample.replace(
                            self.vision_bos_token + self.video_token + self.vision_eos_token,
                            placeholder_string,
                            1,
                        )

            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            sample = sample.replace("<|image_placeholder|>", self.image_token)
            sample = sample.replace("<|video_placeholder|>", self.video_token)
            processed_text.append(sample)

        return processed_text


__all__ = ["OpenPanguOmniProcessor"]
