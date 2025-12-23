# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
from collections.abc import Iterator
import math
import threading
from typing import TYPE_CHECKING, Any, Optional, Union, Mapping, Dict, List
import zmq
import os
import pickle
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

import socket
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.envs import VLLM_RPC_TIMEOUT
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput

from omni.accelerators.pd.utils import get_config_from_dict_or_env

if TYPE_CHECKING:
    from vllm.config import VllmConfig, KVTransferConfig
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request
from vllm.v1.request import Request
from vllm.utils import round_down
from dataclasses import dataclass
from collections import defaultdict
import torch
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group, get_pp_group)

from vllm.utils import get_open_port
from vllm.v1.request import RequestStatus
import queue
from concurrent.futures import ThreadPoolExecutor

profiling_is_set = os.getenv("PROFILING_NAMELIST", None) is not None

from .llmdatadist_connector_v1 import (
    PrefillConnectorScheduler as PrefillConnectorScheduler_V1,
    DecodeConnectorScheduler as DecodeConnectorScheduler_V1,
    PrefillConnectorWorker, DecodeConnectorWorker,
    DatadistConnectorMetadata, DatadistConnectorMetadataPrefill,
    ReqMeta, ReqMetaPrefill,
    handle_exception,
    dump_thread_to_file,
    get_local_ip,
)

GET_META_MSG = b"get_meta_msg"

thread_dump_path = os.environ.get("VLLM_THREAD_DUMP_PATH", "/tmp/vllm_thread_info")
BLOCK_RELEASE_DELAY = int(os.environ.get("BLOCK_RELEASE_DELAY", 600))  # seconds, use to free blocks when the request is finished for a long time 

if os.getenv("ENABLE_DYNAMIC_LLMDATADIST", "0") == "1":
    FLAG_ENABLE_DYNAMIC_LLMDATADIST = True
    LLMDATADIST_BASE_PORT = int(os.environ.get("VLLM_LLMDATADIST_BASE_PORT", 15567))
    from omni.accelerators.pd.llmdatadist_manager_v1 import LLMDataDistManager, LLMDataDistConfig
else:
    FLAG_ENABLE_DYNAMIC_LLMDATADIST = False
    from omni.accelerators.pd.llmdatadist_manager import LLMDataDistManager, LLMDataDistConfig

BASE_PORT = int(os.environ.get("BASE_PORT", "15077"))
ZMQ_BASE_PORT = int(os.environ.get("ZMQ_BASE_PORT", "17555"))
ZMQ_DECODE_PUSH_REQUEST_TO_PREFILL_PORT = int(
    os.environ.get("ZMQ_DECODE_PUSH_REQUEST_TO_PREFILL_PORT", "17556")
)
ZMQ_PREFILL_PUSH_INFORMATION_TO_DECODE_BASE_PORT = int(
    os.environ.get("ZMQ_PREFILL_PUSH_INFORMATION_TO_DECODE_BASE_PORT", "17557")
)

class LLMDataDistConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        if vllm_config.kv_transfer_config is None:
            raise RuntimeError("vllm_config.kv_transfer_config cannot be None")

        if vllm_config.model_config.is_deepseek_mla:
            vllm_config.kv_transfer_config.kv_parallel_size = 1
            logger.info("Set kv_parallel_size to 1 when use deepseek mla model.")

        if FLAG_ENABLE_DYNAMIC_LLMDATADIST:
            local_host_ip = get_local_ip()
            local_host_port = LLMDATADIST_BASE_PORT
            self.datadist_config = LLMDataDistConfig(vllm_config, local_host_ip, local_host_port, ignore_load_rank=True)
            self.host_cluster_id = self.datadist_config.host_cluster_id
            self.host_ip = local_host_ip
        else:
            self.datadist_config = LLMDataDistConfig(vllm_config, ignore_load_rank=True)
            self.host_cluster_id = self.datadist_config.cluster_id_start
            self.host_ip = self.datadist_config.local_group.host_ip
        # Introduce the environment variable VLLM_LLMDATADIST_ZMQ_PORT to resolve ZMQ connection conflicts during
        # multi-P deployments on the same machine.
        # This variable should not be set separately unless specifically required for this scenario.
        self.host_port = get_config_from_dict_or_env(vllm_config.kv_transfer_config, "kv_port",
                                                     "VLLM_LLMDATADIST_ZMQ_PORT", "5568", int)
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.host_port += dp_rank
        self.is_prefill = vllm_config.kv_transfer_config.kv_role == "kv_producer"

        if role == KVConnectorRole.SCHEDULER:
            if self.is_prefill:
                self.connector_scheduler = PrefillConnectorScheduler(vllm_config, self.host_cluster_id, self.host_ip, str(self.host_port))
            else:
                self.connector_scheduler = DecodeConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            if self.is_prefill:
                self.connector_worker = PrefillConnectorWorker(vllm_config, str(self.host_ip), str(self.host_port))
            else:
                self.connector_worker = DecodeConnectorWorker(vllm_config, str(self.host_ip), self.host_cluster_id)
            self.connector_scheduler = None

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
            self,
            scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.build_connector_metadata(scheduler_output)

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
            spec_token_ids: Optional[list[int]] = []
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if self.connector_scheduler is None:
            raise RuntimeError("self.connector_scheduler cannot be None")
        return self.connector_scheduler.request_finished(request, block_ids, spec_token_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.register_kv_caches(kv_caches)

    def unregister_kv_caches(self):
        logger.info("not support")

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        return self.connector_worker.get_finished(self._connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        if self.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        if not isinstance(self._connector_metadata, Union[DatadistConnectorMetadata, DatadistConnectorMetadataPrefill]):
            raise RuntimeError("self._connector_metadata must be an instance of DatadistConnectorMetadata or DatadistConnectorMetadataPrefill")
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Connector does not do layerwise saving."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Connector does not save explicitly."""
        pass

    def wait_for_save(self):
        """Connector does not save explicitly."""
        pass


# --------------------------------------------------------------------------
# Prefill side
# --------------------------------------------------------------------------

class PrefillConnectorScheduler(PrefillConnectorScheduler_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        host_cluster_id: str,
        host_ip: str,
        host_port: str,
    ):
        super().__init__(vllm_config, host_cluster_id, host_ip, host_port)

        self.decode_requests_dict: Dict[str, Dict[str, Union[str, int]]] = {}
        self._decode_requests_lock = threading.Lock()

        self.ctx = zmq.Context()
        self.input_socket = self.ctx.socket(zmq.PULL)
        self.input_socket.bind(f"tcp://{self.host_ip}:{ZMQ_DECODE_PUSH_REQUEST_TO_PREFILL_PORT}")
        logger.info(
            "PrefillConnectorScheduler bind tcp://%s:%d",
            self.host_ip, ZMQ_DECODE_PUSH_REQUEST_TO_PREFILL_PORT,
        )
        self.decode_zmq_socket_map: Dict[str, zmq.Socket] = {}

        t_name = "prefill_receives_decode_requests"
        self.receive_decode_requests_thread = threading.Thread(
            target=self._receive_and_record_decode_requests,
            daemon=True,
            name=t_name,
        )
        self.receive_decode_requests_thread.start()

        self._prefill_pending: Dict[str, dict] = {}
        self._prefill_pending_lock = threading.Lock()
        self._prefill_cv = threading.Condition(self._prefill_pending_lock)
        self._prefill_sender_stop = threading.Event()

        self._prefill_sender_thread = threading.Thread(
            target=self._prefill_sender_loop,
            daemon=True,
            name="prefill_sender_loop",
        )
        self._prefill_sender_thread.start()

    def _receive_and_record_decode_requests(self) -> None:
        logger.info("Prefill begins to receive decode requests")
        while True:
            try:
                if self.input_socket.poll(timeout=100) > 0:
                    msg = self.input_socket.recv_string()
                    decode_request = json.loads(msg)
                    logger.debug("Received decode request: %s", decode_request)
                    with self._decode_requests_lock:
                        self.decode_requests_dict.update(decode_request)
            except Exception as e:
                logger.error("Failed to receive the decode request message: %s", e)

    def _prefill_sender_loop(self) -> None:
        poll_interval = 1.0
        while not self._prefill_sender_stop.is_set():
            with self._prefill_cv:
                if not self._prefill_pending:
                    self._prefill_cv.wait(timeout=poll_interval)
                pending_ids = list(self._prefill_pending.keys())

            if not pending_ids:
                continue

            for req_id in pending_ids:
                if self._prefill_sender_stop.is_set():
                    break

                with self._decode_requests_lock:
                    decode_request = self.decode_requests_dict.pop(req_id, None)
                if not decode_request:
                    continue

                decode_ip = decode_request.get("ip")
                decode_rank = decode_request.get("rank")
                if decode_ip is None or decode_rank is None:
                    logger.warning("Bad decode_request for %s: %s", req_id, decode_request)
                    continue

                path = f"tcp://{decode_ip}:{ZMQ_PREFILL_PUSH_INFORMATION_TO_DECODE_BASE_PORT + decode_rank}"

                try:
                    socket_ = self.decode_zmq_socket_map.get(path)
                    if socket_ is None:
                        socket_ = self.ctx.socket(zmq.PUSH)
                        socket_.connect(path)
                        self.decode_zmq_socket_map[path] = socket_
                        logger.info("Create new socket path: %s", path)

                    with self._prefill_cv:
                        payload = self._prefill_pending.get(req_id)
                        if payload is None:
                            continue

                    data_to_be_sent = {req_id: payload}
                    json_data = json.dumps(data_to_be_sent)
                    socket_.send_string(json_data)
                    logger.info("Send required data for request %s to path: %s", req_id, path)

                    try:
                        if getattr(self, "metadata", None) is not None:
                            self.metadata.add_new_req(
                                request_id=req_id,
                                local_block_ids=payload.get("remote_block_ids"),
                                kv_transfer_params=payload,
                            )
                    except Exception:
                        logger.exception("metadata.add_new_req failed for %s", req_id)

                    with self._prefill_cv:
                        self._prefill_pending.pop(req_id, None)

                except Exception as e:
                    logger.error(
                        "Failed to send required data for request %s to %s: %s",
                        req_id, path, e,
                    )

    def request_finished(
            self,
            request: "Request",
            block_ids: list[int],
            spec_token_ids: Optional[list[int]] = []
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        delay_free_blocks = len(block_ids) > 0
        # record the finish time of the request
        if delay_free_blocks:
            self.requests_finish_time[request.request_id] = time.monotonic()

        payload = dict(
            remote_block_ids=block_ids,
            remote_cluster_id=self.host_cluster_id,
            remote_host_ip=f"tcp://{self.host_ip}:{self.host_port}",
            spec_token_ids=spec_token_ids,
            remote_dp_rank=self.vllm_config.parallel_config.data_parallel_rank,
            remote_request_id=request.request_id,
        )

        if delay_free_blocks:
            with self._prefill_cv:
                self._prefill_pending[request.request_id] = payload
                self._prefill_cv.notify()

        return delay_free_blocks, payload


class DecodeConnectorScheduler(DecodeConnectorScheduler_V1):

    def __init__(self, vllm_config: "VllmConfig"):
        super().__init__(vllm_config)

        self.prefill_info_dict: Dict[str, Dict[str, Union[None, str, List[int]]]] = {}
        self._prefill_info_dict_lock = threading.Lock()

        self.ctx = zmq.Context()
        self.local_ip = get_local_ip()
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        self.input_socket = self.ctx.socket(zmq.PULL)
        self.input_socket.bind(
            f"tcp://{self.local_ip}:{ZMQ_PREFILL_PUSH_INFORMATION_TO_DECODE_BASE_PORT + self.dp_rank}"
        )
        logger.info(
            "DecodeConnectorScheduler bind tcp://%s:%d",
            self.local_ip,
            ZMQ_PREFILL_PUSH_INFORMATION_TO_DECODE_BASE_PORT + self.dp_rank,
        )

        self.prefill_zmq_socket_map: Dict[str, zmq.Socket] = {}

        t_name = "decode_receives_prefill_information"
        self.receive_prefill_information_thread = threading.Thread(
            target=self._receive_and_record_prefill_information,
            daemon=True,
            name=t_name,
        )
        self.receive_prefill_information_thread.start()

    def _receive_and_record_prefill_information(self) -> None:
        logger.info("Decode begins to receive prefill information")
        while True:
            try:
                if self.input_socket.poll(timeout=100) > 0:
                    msg = self.input_socket.recv_string()
                    prefill_information = json.loads(msg)
                    logger.debug("Received prefill information: %s", prefill_information)
                    with self._prefill_info_dict_lock:
                        self.prefill_info_dict.update(prefill_information)
            except Exception as e:
                logger.error("Failed to receive the prefill information: %s", e)


    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        if request.request_id in self.processed_request:
            return 0, False
        params = request.kv_transfer_params
        if params is None:
            return 0, False

        prefill_ip = params["remote_cluster_id"]
        req_id = params["remote_request_id"]

        path = f"tcp://{prefill_ip}:{ZMQ_DECODE_PUSH_REQUEST_TO_PREFILL_PORT}"
        socket_ = self.prefill_zmq_socket_map.get(path)
        if socket_ is None:
            socket_ = self.ctx.socket(zmq.PUSH)
            socket_.connect(path)
            self.prefill_zmq_socket_map[path] = socket_
            logger.info("Create new socket path:%s", path)

        try:
            decode_inf = {req_id: {"ip": self.local_ip, "rank": self.dp_rank}}
            socket_.send_string(json.dumps(decode_inf))
            with self._prefill_info_dict_lock:
                self.prefill_info_dict.setdefault(req_id, {})
            logger.info("Send request to prefill %s", path)
        except Exception as e:
            logger.error("Failed to send request to prefill %s: %s", path, e)

        logger.debug(
            "DatadistConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens, params)

        if num_computed_tokens % self.block_size != 0:
            raise RuntimeError("num_computed_tokens must be divisible by self.block_size")
        rounded_num_prompt_tokens = self._round_up(
            len(request.prompt_token_ids), self.block_size)
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        return count, count > 0

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        logger.debug(
            "Request id %s: blocks length is %d",
            request.request_id,
            len(blocks.blocks),
        )
        params = request.kv_transfer_params
        logger.debug(
            "DatadistConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens, params)
        
        if params is not None:
            self._reqs_need_recv[request.request_id] = (request, blocks)

    def build_connector_metadata(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        metadata = DatadistConnectorMetadata()

        for req_id, (req, blocks) in list(self._reqs_need_recv.items()):
            if req.kv_transfer_params is None:
                logger.warning("For request %s: kv_transfer_params now is None", req_id)
                continue

            with self._prefill_info_dict_lock:
                prefill_information = self.prefill_info_dict.pop(req_id, None)

            if not prefill_information:
                continue

            req.kv_transfer_params = prefill_information
            params = req.kv_transfer_params
            block_ids = None
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_cluster_id", "remote_host_ip")):
                    block_ids = blocks.get_unhashed_block_ids()
                else:
                    logger.warning("Got invalid KVTransferParams: %s.", params)

            self.processed_request.add(req_id)
            metadata.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )
            req.kv_transfer_params = None
            self._reqs_need_recv.pop(req_id, None)

        if self.async_pull_kv and scheduler_output is None and metadata.requests:
            serialized_data = pickle.dumps(metadata)
            self.pub.send(serialized_data)

        return metadata