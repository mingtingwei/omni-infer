import os
import torch
import torch.nn as nn
from vllm.sequence import SequenceData

from omni.accelerators.reasoning_compression.config import ThinkCompressDict
from omni.accelerators.reasoning_compression.utils import (
    tokenize_without_special_tokens,
)

import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ThinkCompressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_parameters()

    def init_parameters(self) -> None:

        self.enable_early_stop = ThinkCompressDict.reasoner_early_think_stopping_enabled == 1
        self.tokenizer_path = ThinkCompressDict.PRECHECKPOINT_PATH

        # Rollback and disable all think compression features due to invalid tokenizer
        if self.tokenizer_path is None or not os.path.exists(self.tokenizer_path):
            ThinkCompressDict.reasoner_early_think_stopping_enabled = False
            self.enable_early_stop = False

        self.reasoner_early_think_stopping_step = ThinkCompressDict.reasoner_early_think_stopping_step
        self.reasoner_early_think_stopping_string = ThinkCompressDict.reasoner_early_think_stopping_string
        self.reasoner_early_think_stopping_tags = ThinkCompressDict.reasoner_early_think_stopping_tags

        print(
            "self.enable_early_stop=",
            self.enable_early_stop,
            flush=True,
        )

        print(
            "self.reasoner_early_think_stopping_step=",
            self.reasoner_early_think_stopping_step,
            flush=True,
        )
        print(
            "self.reasoner_early_think_stopping_string=",
            self.reasoner_early_think_stopping_string,
            flush=True,
        )
        print(
            "self.reasoner_early_think_stopping_tags=",
            self.reasoner_early_think_stopping_tags,
            flush=True,
        )
        print(
            "ThinkCompressDict.reasoner_early_think_stopping_think_start_string",
            ThinkCompressDict.reasoner_early_think_stopping_think_start_string,
            flush=True,
        )

        if ThinkCompressDict.reasoner_early_think_stopping_think_start_string == "":
            # When this is not set, we always do early stopping
            # This is for backward compatibility
            self.enforce_early_stopping = True
        else:
            # convert to tuple so that it can be directly compared
            self.think_start_token_ids = tuple(
                tokenize_without_special_tokens(self.tokenizer, ThinkCompressDict.reasoner_early_think_stopping_think_start_string)
            )
            self.enforce_early_stopping = False

        self.think_end_token_ids = eval(self.reasoner_early_think_stopping_string)
        print("self.think_end_token_ids", self.think_end_token_ids, flush=True)

        # convert to tuple so that it can be directly compared
        self.think_stop_tag_ids = tuple(eval(self.reasoner_early_think_stopping_tags))
        print("self.think_stop_tag_ids", self.think_stop_tag_ids, flush=True)

        self.initialized = True

    def get_think_stop_step(self, seq_data: SequenceData):
        # control thinking budget for each request
        if hasattr(seq_data, "thinking_budget") and seq_data.thinking_budget is not None:
            default_think_stop_step = seq_data.thinking_budget
        else:
            default_think_stop_step = self.reasoner_early_think_stopping_step

        if seq_data.stop_length is not None:
            if seq_data.stop_type == "repeated_output_in_summary":
                # 对summary中的终止需要特殊处理
                # 无论reasoner_early_think_stopping_step是什么，都应该用seq_data.stop_length
                think_stop_step = seq_data.stop_length
            else:
                # 如果stop length被设置得比reasoner_early_think_stopping_step更大，
                # 那么它之前就已经开始了硬终止，此处应当保持最小的那个
                think_stop_step = min(default_think_stop_step, seq_data.stop_length)
        else:
            think_stop_step = default_think_stop_step
        return think_stop_step

    def update_guided_decoding_info(self, seq_data, tags_to_fix) -> None:
        tags_to_fix.append(self._get_guided_decoding_token_id(seq_data))

    def update_sampled_token_ids(self, token_ids: torch.Tensor, seq_datas) -> torch.Tensor:
        """
        For the already sampled token ids, directly update the sampling token ids based on guided decoding info
        """
        num_tokens = token_ids.shape[-1]
        if token_ids.ndim == 1:
            for req_idx, _seq_data in enumerate(seq_datas):
                target_token_id = self._get_guided_decoding_token_id(_seq_data, 0)
                if target_token_id is not None:
                    token_ids[req_idx] = target_token_id
        else:
            for req_idx, _seq_data in enumerate(seq_datas):
                # handle end of guided_decoding_token_ids for speculative decoding
                has_valid_guided_tokens = False
                # record the thinking stop position and the current possible token positions to
                # reduce overhead for speculative decoding, as multiple tokens need to be checked
                think_stop_step = self.get_think_stop_step(_seq_data)
                sampled_output_token_length = len(_seq_data.output_token_ids)
                max_cur_sampled_output_token_length = sampled_output_token_length + num_tokens
                for offset in range(num_tokens):
                    target_token_id = self._get_guided_decoding_token_id(
                        _seq_data, offset, think_stop_step=think_stop_step)

                    if target_token_id is not None:
                        token_ids[req_idx, offset] = target_token_id
                        # label that guided decoding for SD is enabled
                        has_valid_guided_tokens = True
                    
                    """
                    for requests with both SD and guided decoding enabled,
                    if over-length tokens over the defined guided_decoding_token_ids, 
                    set them as invalid tokens to avoid chaos in output semantics
                    """
                    if target_token_id is None:
                        if has_valid_guided_tokens:
                            token_ids[req_idx, offset] = -1
                        elif sampled_output_token_length > think_stop_step or \
                            max_cur_sampled_output_token_length < think_stop_step:
                            # if guided decoding is not enabled for all possible tokens, exit now to reduce overhead
                            break

        return token_ids

    def _get_guided_decoding_token_id(self, seq_data, offset=0, think_stop_step=None):
        """ """
        seq_id = seq_data.seq_id

        if seq_data.is_think_finished and seq_data.stop_type != "repeated_output_in_summary":
            return None

        think_end_token_ids_in_use = self.think_end_token_ids

        prompt_token_ids = seq_data.prompt_token_ids
        output_token_ids = seq_data.output_token_ids

        if not seq_data.is_think_started:
            if self.enforce_early_stopping:
                # For backward compatibility, some old scripts do not have think start tokens
                seq_data.is_think_started = True
            else:
                if output_token_ids[-len(self.think_start_token_ids) :] == self.think_start_token_ids:
                    # When think start token is detected
                    seq_data.is_think_started = True
                    seq_data.is_think_finished = False

        if not seq_data.is_think_started:
            # When think is not started, skip
            return None

        if output_token_ids[-len(self.think_stop_tag_ids) :] == self.think_stop_tag_ids:
            seq_data.is_think_finished = True
            return None

        current_len = len(output_token_ids) + offset

        # Get think stop step
        if think_stop_step is None:
            think_stop_step = self.get_think_stop_step(seq_data)

        if current_len < think_stop_step:
            # Reduce resource use
            return None

        # Catch over-length requests
        if current_len >= think_stop_step and current_len - think_stop_step < len(think_end_token_ids_in_use):

            tag_index = current_len - think_stop_step
            tag_idx = think_end_token_ids_in_use[tag_index]
            print(
                f"PID: {os.getpid()}, seq_id {seq_id}, reason {seq_data.stop_type} current_len: {current_len}, "
                f"think_stop_step: {think_stop_step}, Appending {tag_idx} "
                f"from think_end_token_ids_in_use {think_end_token_ids_in_use}",
                flush=True,
            )
            return tag_idx
        else:
            return None

    def get_tokenizer(self, tokenizer_path):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
