import fcntl
import logging
import pickle
import time
import gzip
import struct
import os
from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from typing import Optional
from unittest.mock import patch

import numpy as np
import torch

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from vllm.sampling_params import RequestOutputKind

_global_experts_capturer: Optional["RoutedExpertsCapturer"] = None
_global_experts_reader: Optional["RoutedExpertsReader"] = None
SAVE_DIR = os.getenv("EXPORT_MOE_EXPERTS_TEST_PATH", None)
logger = logging.getLogger(__name__)

FMT, SIZE = "i", struct.calcsize("i")
LOCK_FILE_PREFIX = "/tmp/vllm_routed_experts"
BUFFER_PREFIX = "vllm_routed_experts_buffer"


def save_data(indices: np.ndarray, data: np.ndarray, save_dir: str) -> None:
    """Save indices and data to a compressed pickle file."""
    if save_dir is None:
        return
    
    data_dict = {int(indices[i]): v for i, v in enumerate(data)}
    save_name = os.path.join(save_dir, f"data_{time.time()}.pkl")
    
    with gzip.open(save_name, "wb") as file:
        pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


def save_indices(indices: np.ndarray, request_id: str, save_dir: str) -> None:
    """Save request indices to a compressed pickle file."""
    if save_dir is None:
        return
    
    save_name = os.path.join(save_dir, f"indices_{time.time()}.pkl")
    with gzip.open(save_name, "wb") as file:
        pickle.dump({request_id: indices.tolist()}, file, protocol=pickle.HIGHEST_PROTOCOL)


def lock_file(file_pointer) -> None:
    """Acquire an exclusive lock on a file."""
    fcntl.flock(file_pointer, fcntl.LOCK_EX)


def unlock_file(file_pointer) -> None:
    """Release a file lock."""
    fcntl.flock(file_pointer, fcntl.LOCK_UN)


class RoutedExpertsCapturer(ABC):
    """Abstract base class for routed experts capturer."""
    
    @staticmethod
    def create() -> "RoutedExpertsCapturer":
        """Create a singleton instance of the experts capturer."""
        global _global_experts_capturer
        if _global_experts_capturer is not None:
            raise RuntimeError("Experts capturer already created.")
        _global_experts_capturer = _RoutedExpertsCapturer()
        return _global_experts_capturer

    @staticmethod
    def get_instance() -> Optional["RoutedExpertsCapturer"]:
        """Get the singleton instance of the experts capturer."""
        return _global_experts_capturer


class _RoutedExpertsCapturer:
    """Concrete implementation of routed experts capturer."""
    
    def __init__(self) -> None:
        self._experts_capturer_device_buffer: Optional[torch.Tensor] = None
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._host_buffer_view: Optional[np.ndarray] = None
        self._shm_total_token_num: Optional[shared_memory.SharedMemory] = None
        self.instance_id: Optional[str] = None
        self.num_moe_layers: int = 0
        self.num_selected_experts: int = 0
        self.torch_dtype: torch.dtype = torch.uint8
        self.np_dtype: np.dtype = np.uint8
        self.max_num_kv_tokens: int = 0
        self.lock_file: Optional[str] = None

    def init_buffer(
        self,
        max_num_batched_tokens: int,
        num_blocks: int,
        num_cache_groups: int,
        block_size: int,
        num_hidden_layers: int,
        mlp_only_layers: list,
        num_experts_per_tok: int,
        instance_id: str,
        enable_init_shared_memory: bool,
    ) -> None:
        """Initialize the experts capturer buffer."""
        self.instance_id = instance_id
        self.num_moe_layers = num_hidden_layers - len(mlp_only_layers)
        self.num_selected_experts = num_experts_per_tok
        
        self.torch_dtype = (
            torch.uint8
            if self.num_selected_experts - 1 <= torch.iinfo(torch.uint8).max
            else torch.uint16
        )
        self.np_dtype = np.uint8 if self.torch_dtype == torch.uint8 else np.uint16
        self.max_num_kv_tokens = (num_blocks // num_cache_groups + 1) * block_size

        if self._experts_capturer_device_buffer is None:
            self._experts_capturer_device_buffer = torch.zeros(
                (max_num_batched_tokens, self.num_moe_layers, self.num_selected_experts),
                dtype=self.torch_dtype,
                device="npu",
            )
            self.lock_file = f"{LOCK_FILE_PREFIX}_{instance_id}.lock"

            if enable_init_shared_memory:
                shape = (
                    self.max_num_kv_tokens,
                    self.num_moe_layers,
                    self.num_selected_experts,
                )
                nbytes = int(np.prod(shape)) * np.dtype(self.np_dtype).itemsize
                
                with open(self.lock_file, "wb") as fp:
                    lock_file(fp)
                    try:
                        self._shm = shared_memory.SharedMemory(
                            create=True,
                            size=nbytes,
                            name=f"{BUFFER_PREFIX}_{instance_id}",
                        )
                        self._host_buffer_view = np.ndarray(
                            shape, dtype=self.np_dtype, buffer=self._shm.buf
                        )
                        self._host_buffer_view.fill(0)
                        
                        self._shm_total_token_num = shared_memory.SharedMemory(
                            create=True,
                            size=SIZE,
                            name=f"{BUFFER_PREFIX}_{instance_id}_num",
                        )
                        struct.pack_into(FMT, self._shm_total_token_num.buf, 0, 0)
                    finally:
                        unlock_file(fp)

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Capture experts for a specific layer."""
        if layer_id is None or self._experts_capturer_device_buffer is None:
            return
        
        batch_size = topk_ids.shape[0]
        self._experts_capturer_device_buffer[:batch_size, layer_id, :] = topk_ids

    def clear_buffer(self) -> None:
        """Clear the device buffer."""
        if self._experts_capturer_device_buffer is not None:
            self._experts_capturer_device_buffer.zero_()

    def save_captured_experts(
        self,
        slot_mapping: np.ndarray,
        len_req_ids: int,
        max_num_seqs: int,
        enable_torchair_graph_mode: bool,
    ) -> None:
        """Save captured experts to shared memory and disk."""
        real_mapping = torch.from_numpy(slot_mapping)
        indices = torch.where(real_mapping == -1)[0]
        num_valid_tokens = (
            indices[0].item() if indices.shape[0] > 0 else real_mapping.numel()
        )
        
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        
        if enable_torchair_graph_mode and num_valid_tokens == len_req_ids:
            split_size = (max_num_seqs - 1) // tp_size + 1
        else:
            split_size = (num_valid_tokens - 1) // tp_size + 1
        
        rank_lower_bound = split_size * tp_rank
        rank_upper_bound = min(split_size * tp_rank + split_size, num_valid_tokens)
        
        if rank_upper_bound <= rank_lower_bound:
            return

        indices = real_mapping[rank_lower_bound:rank_upper_bound].numpy()
        
        if self._host_buffer_view is None:
            self._attach_shared_memory()
        
        with open(self.lock_file, "wb+") as fp:
            lock_file(fp)
            try:
                if (
                    self._host_buffer_view is not None
                    and self._experts_capturer_device_buffer is not None
                ):
                    num_tokens = len(indices)
                    data = self._experts_capturer_device_buffer[:num_tokens, :, :].cpu().numpy()
                    self._host_buffer_view[indices, :, :] = data
                    save_data(indices, data, SAVE_DIR)
                
                if self._shm_total_token_num is not None:
                    current_count = struct.unpack_from(
                        FMT, self._shm_total_token_num.buf, 0
                    )[0]
                    struct.pack_into(
                        FMT,
                        self._shm_total_token_num.buf,
                        0,
                        current_count + len(data),
                    )
            finally:
                unlock_file(fp)

    def _attach_shared_memory(self) -> None:
        """Attach to existing shared memory."""
        with open(self.lock_file, "rb+") as fp:
            lock_file(fp)
            try:
                with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
                    if self._shm is None:
                        self._shm = shared_memory.SharedMemory(
                            name=f"{BUFFER_PREFIX}_{self.instance_id}"
                        )
                    if self._shm_total_token_num is None:
                        self._shm_total_token_num = shared_memory.SharedMemory(
                            name=f"{BUFFER_PREFIX}_{self.instance_id}_num"
                        )
                    
                    buffer_size = (
                        self._shm.size
                        // self.num_moe_layers
                        // self.num_selected_experts
                        // np.dtype(self.np_dtype).itemsize
                    )
                    self._host_buffer_view = np.ndarray(
                        (buffer_size, self.num_moe_layers, self.num_selected_experts),
                        dtype=self.np_dtype,
                        buffer=self._shm.buf,
                    )
            finally:
                unlock_file(fp)

    def __del__(self) -> None:
        """Cleanup shared memory resources."""
        try:
            if self._shm is not None:
                self._shm.close()
                self._shm.unlink()
                if self._shm_total_token_num is not None:
                    self._shm_total_token_num.close()
                    self._shm_total_token_num.unlink()
        except Exception:
            logger.debug("Exception during __del__ cleanup for capturer", exc_info=True)


class RoutedExpertsReader(ABC):
    """Abstract base class for routed experts reader."""
    
    @staticmethod
    def create() -> "RoutedExpertsReader":
        """Create a singleton instance of the experts reader."""
        global _global_experts_reader
        if _global_experts_reader is None:
            _global_experts_reader = _RoutedExpertsReader()
        return _global_experts_reader

    @staticmethod
    def get_instance() -> Optional["RoutedExpertsReader"]:
        """Get the singleton instance of the experts reader."""
        if _global_experts_reader is None:
            logger.info("Experts reader not initialized.")
        return _global_experts_reader


class _RoutedExpertsReader:
    """Concrete implementation of routed experts reader."""
    
    def __init__(self) -> None:
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._host_buffer_view: Optional[np.ndarray] = None
        self._shm_total_token_num: Optional[shared_memory.SharedMemory] = None
        self.is_pd_disaggregation: bool = False
        self.block_size: int = 0
        self.lock_file: Optional[str] = None
        self.np_dtype: np.dtype = np.uint8

    def attach_buffer(
        self,
        num_hidden_layers: int,
        mlp_only_layers: list,
        num_experts_per_tok: int,
        block_size: int,
        instance_id: str,
    ) -> None:
        """Attach to the shared memory buffer."""
        self.is_pd_disaggregation = bool(os.getenv("ROLE"))
        if self._shm is not None:
            return

        self.block_size = block_size
        self.lock_file = f"{LOCK_FILE_PREFIX}_{instance_id}.lock"
        num_moe_layers = num_hidden_layers - len(mlp_only_layers)
        self.np_dtype = (
            np.uint8
            if num_experts_per_tok - 1 <= np.iinfo(np.uint8).max
            else np.uint16
        )

        with open(self.lock_file, "rb+") as fp:
            lock_file(fp)
            try:
                with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
                    self._shm = shared_memory.SharedMemory(
                        name=f"{BUFFER_PREFIX}_{instance_id}"
                    )
                    buffer_size = (
                        self._shm.size
                        // num_moe_layers
                        // num_experts_per_tok
                        // np.dtype(self.np_dtype).itemsize
                    )
                    self._host_buffer_view = np.ndarray(
                        (buffer_size, num_moe_layers, num_experts_per_tok),
                        dtype=self.np_dtype,
                        buffer=self._shm.buf,
                    )
                    self._shm_total_token_num = shared_memory.SharedMemory(
                        name=f"{BUFFER_PREFIX}_{instance_id}_num"
                    )
            finally:
                unlock_file(fp)

    def get_routed_experts(
        self, request, kv_cache_manager, stopped: bool
    ) -> Optional[np.ndarray]:
        """Get routed experts for a request."""
        is_stream = request.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY
        if not stopped and not is_stream:
            return None

        block_ids = kv_cache_manager.get_block_ids(request.request_id)[0]
        num_tokens = request.num_tokens - 1
        is_prefill = num_tokens == request.num_prompt_tokens
        start_tokens = (
            request.num_prompt_tokens
            if not is_prefill and self.is_pd_disaggregation
            else 0
        )
        
        block_ids_array = np.array(block_ids, dtype=np.uint32)
        indices = (
            block_ids_array.reshape((-1, 1)) * self.block_size
            + np.arange(self.block_size)
        ).flatten()[start_tokens:num_tokens]
        
        if not is_prefill and is_stream:
            indices = indices[-1:]

        while (
            struct.unpack_from(FMT, self._shm_total_token_num.buf, 0)[0] < indices.size
        ):
            pass
        
        struct.pack_into(
            FMT,
            self._shm_total_token_num.buf,
            0,
            struct.unpack_from(FMT, self._shm_total_token_num.buf, 0)[0] - indices.size,
        )

        with open(self.lock_file, "rb+") as fp:
            lock_file(fp)
            try:
                if self._host_buffer_view is None:
                    raise RuntimeError("Buffer not attached.")
                routed_experts = self._host_buffer_view[indices, :, :].copy()
            finally:
                unlock_file(fp)

        if routed_experts is not None:
            save_indices(indices, request.request_id, SAVE_DIR)
        
        return routed_experts

    def __del__(self) -> None:
        """Cleanup shared memory resources."""
        try:
            if self._shm is not None:
                self._shm.close()
                self._shm_total_token_num.close()
                os.remove(self.lock_file)
        except Exception:
            logger.debug("Exception during __del__ cleanup for reader", exc_info=True)