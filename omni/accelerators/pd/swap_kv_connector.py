# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor

import safetensors
import torch
import torch_npu

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)

@dataclass
class SwapKVReqMeta:
    # req id
    req_id: str
    # user id
    uid: int
    # Request tokens
    token_ids: torch.Tensor
    # Slot mappings, should have the same length as token_ids
    slot_mapping: torch.Tensor
    # Is store or load
    is_store: bool
    # Prefill block offset
    prefill_block_offset: int

    @staticmethod
    def make_meta(token_ids: list[int], block_ids: list[int], block_size: int,
                  is_store: bool, uid: int, req_id: str) -> "SwapKVReqMeta":
        token_ids_tensor = torch.as_tensor(token_ids, device='cpu')
        prefill_block_offset = 0
        if is_store:
            prefill_block_offset = block_ids[0]
        return SwapKVReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=None,
            is_store=is_store,
            uid=uid,
            req_id=req_id,
            prefill_block_offset=prefill_block_offset
        )


@dataclass
class SwapKVConnectorMetadata(KVConnectorMetadata):
    requests: list[SwapKVReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        uid: int,
        req_id: str
    ) -> None:
        self.requests.append(
            SwapKVReqMeta.make_meta(token_ids, block_ids, block_size, is_store, uid, req_id))


class SwapKVConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        self._connector_metadata = SwapKVConnectorMetadata()

        self._block_size = vllm_config.cache_config.block_size

        self._requests_need_load: dict[str, Request] = {}

        self._dram_tensor_cache: dict[int, dict] = {}

        logger.info(vllm_config.kv_transfer_config)

        self._onload_history_kv_events = [torch_npu.npu.Event() for _ in range(8)]
        self._onload_stream = torch_npu.npu.Stream()

        # offload stream
        self._offload_history_kv_events = [torch_npu.npu.Event() for _ in range(8)]
        self._offload_stream = torch_npu.npu.Stream()

        # req_id to user id
        self.req_to_user_mapping: dict[str, int] = {}

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        return

    def start_load_kv_by_layer(self, forward_context: "ForwardContext", layer_name:str,
                      **kwargs) -> None:
        print(f"start to load kv from connector.")
        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_k_cache: torch.Tensor,
            src_v_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
            ) -> None:
            dst_kv_cache_layer_shape = dst_kv_cache_layer[0].shape
            
            num_pages = dst_kv_cache_layer_shape[0]
            page_size = dst_kv_cache_layer_shape[1]

            dst_k_cache_layer = dst_kv_cache_layer[0].reshape(num_pages * page_size, -1)
            dst_v_cache_layer = dst_kv_cache_layer[1].reshape(num_pages * page_size, -1)

            index_start = int(slot_mapping[0])
            index_end = int(slot_mapping[-1] + 1)

            dst_k_cache_layer[index_start:index_end].copy_(src_k_cache, non_blocking=True)
            dst_v_cache_layer[index_start:index_end].copy_(src_v_cache, non_blocking=True)
            
            dst_k_cache_layer = dst_k_cache_layer.reshape(dst_kv_cache_layer_shape)  # 分页格式
            dst_v_cache_layer = dst_v_cache_layer.reshape(dst_kv_cache_layer_shape)
        # Get the metadata
        attn_metadata = forward_context.attn_metadata[layer_name]
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, SwapKVConnectorMetadata)
        if metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the connector metadata is None"
            )
            return

        if attn_metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the attn_metadata is None")
            return

        uid = attn_metadata.additional_metadata["uid"].item()
        if uid is None:
            logger.warning("In connector.start_load_kv_by_layer, but user id is None.")
            return
        # Load the KV for each request each layer
        for request in metadata.requests:
            if not request.is_store or uid != request.uid:
                continue

            layer = forward_context.no_compile_layers[layer_name]
            kv_cache_attr = getattr(layer, 'kv_cache', None)
            if kv_cache_attr is None:
                continue

            kv_cache_layer = kv_cache_attr[ \
                    forward_context.virtual_engine]
            k_cache = self._dram_tensor_cache[uid][layer_name]["k_cache"]
            v_cache = self._dram_tensor_cache[uid][layer_name]["v_cache"]
            slot_mapping = request.slot_mapping
            if attn_metadata.block_tables is not None and request.prefill_block_offset != 0:
                slot_mapping = slot_mapping + (attn_metadata.block_tables[0][0].item() - request.prefill_block_offset) * self._block_size
            inject_kv_into_layer(kv_cache_layer, k_cache, v_cache, slot_mapping)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs) -> None:
        print(f"start to save kv layer to connector.")
        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            num_pages, page_size = layer.shape[0], layer.shape[1]
            index_start = slot_mapping[0]
            index_end = slot_mapping[-1] + 1
            return layer.reshape(num_pages * page_size, -1)[index_start:index_end, ...]

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, SwapKVConnectorMetadata)

        uid = attn_metadata.additional_metadata["uid"].item()
        if uid is None:
            logger.warning("uids in attn_metadata is None.")
            return
        
        for request in connector_metadata.requests:
            if not request.is_store or uid != request.uid:
                continue
            slot_mapping = attn_metadata.slot_mapping
            k_cache = extract_kv_from_layer(kv_layer[0], slot_mapping)
            v_cache = extract_kv_from_layer(kv_layer[1], slot_mapping)

            k_cache_pinned = k_cache.detach().to('cpu')
            v_cache_pinned = v_cache.detach().to('cpu')

            if request.uid not in self._dram_tensor_cache:
                self._dram_tensor_cache[request.uid] = {}
            self._dram_tensor_cache[request.uid][layer_name] = {
                "k_cache": k_cache_pinned,
                "v_cache": v_cache_pinned
            }
            # breakpoint()
            request.slot_mapping = slot_mapping
        
    def wait_for_save(self):
        return

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        uid = request.prompt_token_ids[0]
        self.req_to_user_mapping[request.request_id] = uid

        # 判断是否为prefill阶段，如果是prefill阶段，不需要拉取，直接返回0
        if request.prompt_token_ids[1] != -1:
            return 0, False
        
        connector_metadata = self._get_connector_metadata()
        for req in connector_metadata.requests:
            if uid != req.uid or req.is_store is False:
                continue
            logger.info(f"User Hit in Connector. uid: {uid}")
            
            num_matched_token = len(req.token_ids)

            request.prompt_token_ids = num_matched_token * [0] + request.prompt_token_ids
            request._all_token_ids = num_matched_token * [0] + request._all_token_ids
            request.all_token_ids = ConstantList(request._all_token_ids)

            return num_matched_token, False
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):          
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = self._get_connector_metadata()
        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            uid = self.req_to_user_mapping.get(new_req.req_id, None)
            if uid is None:
                continue

            if new_req.req_id in self._requests_need_load:
                meta.add_request(token_ids=new_req.prompt_token_ids,
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size,
                                 is_store=False,
                                 uid=uid,
                                 req_id=new_req.req_id)
                total_need_load += 1
            else:
                meta.add_request(token_ids=new_req.prompt_token_ids[1 : -1],
                                 block_ids=new_req.block_ids[0],
                                 block_size=self._block_size,
                                 is_store=True,
                                 uid=uid,
                                 req_id=new_req.req_id)

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta
    
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        done_sending = set()
        done_recving = set()

        connector_metadata = self._get_connector_metadata()
        for finished_req_id in finished_req_ids:
            for request in connector_metadata.requests:
                if request.req_id != finished_req_id:
                    continue
                if request.is_store and request.uid in self._dram_tensor_cache:
                    done_sending.add(finished_req_id)
                elif not request.is_store:
                    done_recving.add(finished_req_id)
        return done_sending, done_recving