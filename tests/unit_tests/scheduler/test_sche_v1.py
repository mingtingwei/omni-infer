# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Union, Any
from unittest.mock import Mock, patch
from pathlib import Path
import pytest
import torch
import os

from vllm.config import (CacheConfig, KVTransferConfig, ModelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

import huggingface_hub

_CACHED_NO_EXIST_T = Any
EOS_TOKEN_ID = 50256
ENV_MOCK_FILE_PATH: str = "/"

# initialize the model path
def get_model_mock_file_path():
    work_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(work_dir)
    redirect_target_dir = os.path.join(parent_dir, "accelerators", "mock_model")
    global ENV_MOCK_FILE_PATH
    ENV_MOCK_FILE_PATH = redirect_target_dir
    return ENV_MOCK_FILE_PATH

# origin model mock: facebook/opt-125m

def mock_try_to_load_from_cache(
        repo_id: str,
        filename: str,
        cache_dir: Union[str, Path, None] = None,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
) -> Union[str, Any, None]:
    return repo_id


global_patch = patch(
    target="huggingface_hub.try_to_load_from_cache",
    side_effect=mock_try_to_load_from_cache
)

mock_func = global_patch.start()


# model path set by setup_vllm_mock.sh
def create_scheduler(
        model: str = get_model_mock_file_path(),
        max_num_seqs: int = 16,
        max_num_batched_tokens: int = 8192,
        enable_prefix_caching: Optional[bool] = None,
        long_prefill_token_threshold: int = 0,
        disable_chunked_mm_input: bool = False,
        use_kv_connector: bool = False,
        num_blocks: int = 10000,
        block_size: int = 16,
        max_model_len: Optional[int] = None,
        num_speculative_tokens: Optional[int] = None,
        mock_functions: mock_func = Any
) -> Scheduler:
    if max_model_len is None:
        max_model_len = max_num_batched_tokens
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        disable_chunked_mm_input=disable_chunked_mm_input,
        enable_chunked_prefill=True,
    )

    model_config = ModelConfig(
        model=model,
        task="auto",
        tokenizer=model,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )

    kwargs_cache = ({} if enable_prefix_caching is None else {
        'enable_prefix_caching': enable_prefix_caching
    })
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        **kwargs_cache,
    )

    kv_transfer_config = KVTransferConfig(
        kv_connector="SharedStorageConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"shared_storage_path": "local_storage"},
    ) if use_kv_connector else None

    speculative_config: Optional[SpeculativeConfig] = None
    if num_speculative_tokens is not None:
        speculative_config = SpeculativeConfig(
            model="ngram", num_speculative_tokens=num_speculative_tokens)

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        speculative_config=speculative_config,
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        tensors={},
        kv_cache_groups=[
            KVCacheGroupSpec(['layer'],
                             FullAttentionSpec(block_size, 1, 1, torch.float32,
                                               False))
        ],
    )

    cache_config.num_gpu_blocks = num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


class MockConnector:
    def request_finished(self, *args: Any, **kwargs: Any) -> None:
        return

    def update_state_after_alloc(self, *args: Any, **kwargs: Any) -> None:
        return

    def get_num_new_matched_tokens(self, *args: Any, **kwargs: Any) -> tuple[int, bool]:
        return 0, True

    def build_connector_meta(self, *args: Any, **kwargs: Any) -> KVConnectorMetadata:
        return KVConnectorMetadata()


from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1, KVConnectorRole


def mock_create_connector_v1(
        *args, **kwargs) -> KVConnectorBase_V1:
    return MockConnector


def create_kv_manager_scheduler(
        model: str = get_model_mock_file_path(),
        max_num_seqs: int = 16,
        max_num_batched_tokens: int = 8192,
        enable_prefix_caching: Optional[bool] = None,
        long_prefill_token_threshold: int = 0,
        disable_chunked_mm_input: bool = False,
        use_kv_connector: bool = False,
        num_blocks: int = 10000,
        block_size: int = 16,
        max_model_len: Optional[int] = None,
        num_speculative_tokens: Optional[int] = None,
        mock_functions: mock_func = Any
) -> Scheduler:
    with patch("vllm.distributed.kv_transfer.kv_connector.factory.KVConnectorFactory.create_connector_v1",
               side_effect=mock_create_connector_v1):
        if max_model_len is None:
            max_model_len = max_num_batched_tokens
        scheduler_config = SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            long_prefill_token_threshold=long_prefill_token_threshold,
            disable_chunked_mm_input=disable_chunked_mm_input,
            enable_chunked_prefill=True,
            policy="fcfs",
            preemption_mode="swap",
        )

        model_config = ModelConfig(
            model=model,
            task="auto",
            tokenizer=model,
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="float16",
            seed=42,
        )

        # Cache config, optionally force APC
        kwargs_cache = ({} if enable_prefix_caching is None else {
            'enable_prefix_caching': enable_prefix_caching
        })
        cache_config = CacheConfig(
            block_size=block_size,
            gpu_memory_utilization=0.9,
            swap_space=2,
            cache_dtype="auto",
            **kwargs_cache,
        )

        kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        ) if use_kv_connector else None

        speculative_config: Optional[SpeculativeConfig] = None
        if num_speculative_tokens is not None:
            speculative_config = SpeculativeConfig(
                model="ngram", num_speculative_tokens=num_speculative_tokens)

        vllm_config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            kv_transfer_config=kv_transfer_config,
            speculative_config=speculative_config,
            additional_config={
                "async_schedule": False,
                "enable_omni_attn": True,
                "combine_block": "1",
            }
        )
        os.environ["LLM_WAITING_OUT"] = "86400"

        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,  # A large number of blocks to hold all requests
            tensors={"test": torch.tensor(1)},
            kv_cache_groups=[
                KVCacheGroupSpec(['layer'],
                                 FullAttentionSpec(block_size, 1, 1, torch.float32,
                                                   False))
            ],
        )

        cache_config.num_gpu_blocks = num_blocks
        return Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            log_stats=True,
            structured_output_manager=StructuredOutputManager(vllm_config),
        )


def create_requests(num_requests: int,
                    num_tokens: int = 10,
                    mm_positions: Optional[list[PlaceholderRange]] = None,
                    max_tokens: int = 16,
                    stop_token_ids: Optional[list[int]] = None,
                    prompt_logprobs: Optional[int] = None):
    sampling_params = SamplingParams(ignore_eos=False,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids,
                                     prompt_logprobs=prompt_logprobs)
    requests = []
    for i in range(num_requests):
        if mm_positions is not None:
            mm_position = mm_positions[i]
            mm_inputs = [MultiModalKwargs({})] * len(mm_position)
        else:
            mm_position = None
            mm_inputs = None
        request = Request(
            request_id=f"{i}",
            prompt_token_ids=[i] * num_tokens,
            sampling_params=sampling_params,
            multi_modal_inputs=mm_inputs,
            multi_modal_placeholders=mm_position,
            multi_modal_hashes=None,
            eos_token_id=EOS_TOKEN_ID,
            arrival_time=0,
        )
        requests.append(request)
    return requests


def test_requests_add_finish_abort():
    '''
    Test normal request add, finish, abort
    '''
    scheduler = Mock(name="Scheduler", mock_functions=mock_func)
    scheduler.requests = []
    scheduler.waiting = []

    def mock_add_request(request: Request):
        scheduler.requests.append(request.request_id)
        scheduler.waiting.append(request.request_id)

    scheduler.add_request.side_effect = mock_add_request

    requests = create_requests(num_requests=10)
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        assert request.request_id in scheduler.requests
        assert len(scheduler.waiting) == i + 1

    def mock_finish_requests(request: Request, status: RequestStatus):
        scheduler.requests.remove(request)
        scheduler.waiting.remove(request)

    scheduler.finish_requests.side_effect = mock_finish_requests

    def mock_get_num_unfinished_requests():
        return len(scheduler.requests)

    scheduler.get_num_unfinished_requests.side_effect = mock_get_num_unfinished_requests

    for i, request in enumerate(requests):
        scheduler.finish_requests(request.request_id,
                                  RequestStatus.FINISHED_ABORTED)
        assert request.request_id not in scheduler.requests
        assert len(scheduler.waiting) == 9 - i
        assert scheduler.get_num_unfinished_requests() == len(requests) - i - 1


@pytest.mark.parametrize("enable_prefix_caching, prompt_logprobs", [
    (None, None),
    (True, 5),
])
def test_schedule(enable_prefix_caching: Optional[bool],
                  prompt_logprobs: Optional[int]):
    '''
    Test scheduling
    '''
    scheduler = create_scheduler(enable_prefix_caching=enable_prefix_caching)
    requests = create_requests(num_requests=10,
                               prompt_logprobs=prompt_logprobs)
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert len(output.scheduled_cached_reqs) == 0
    assert len(output.finished_req_ids) == 0
    # Verify all requests are scheduled.
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)
    for i, request in enumerate(requests):
        assert request in scheduler.running


def test_schedule_kvmanager():
    '''
    Test schedule init with kvmanager config
    '''
    scheduler = create_kv_manager_scheduler(use_kv_connector=True)
    requests = create_requests(num_requests=10)
    for request in requests:
        request.arrival_time = -1147483648.0
        scheduler.add_request(request)

    from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                                  create_request_queue)
    from vllm.v1.core.sched.scheduler import PreemptionMode
    scheduler.policy == SchedulingPolicy.PRIORITY
    scheduler.preemption_mode == PreemptionMode.SWAP

    output = scheduler.schedule()
    assert len(output.scheduled_cached_reqs) == 0
    assert len(output.finished_req_ids) == 10


def test_schedule_multimodal_requests():
    scheduler = create_scheduler()
    mm_positions = [[PlaceholderRange(offset=i, length=100)]
                    for i in range(10)]
    requests = create_requests(
        num_requests=10,
        num_tokens=200,

    )
    for request in requests:
        request.status == RequestStatus.WAITING
        scheduler.add_request(request)

    from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                                  create_request_queue)
    from vllm.v1.core.sched.scheduler import PreemptionMode

    scheduler.policy == SchedulingPolicy.PRIORITY
    scheduler.preemption_mode == PreemptionMode.SWAP
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 10
    assert len(output.scheduled_cached_reqs) == 0
    assert len(output.finished_req_ids) == 0

    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)
        pass
    assert len(output.scheduled_encoder_inputs) == 0
    for req_id, encoder_input in output.scheduled_encoder_inputs.items():
        assert len(encoder_input) == 1


def test_stop_via_update_from_output():
    """Test stopping behavior through update_from_output"""
    scheduler = create_scheduler(num_speculative_tokens=1)

    # Test case 1: Stop on EOS token
    requests = create_requests(num_requests=2, max_tokens=10)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)

    scheduler_output = SchedulerOutput(scheduled_new_reqs=[],
                                       scheduled_cached_reqs=[],
                                       num_scheduled_tokens={
                                           requests[0].request_id: 1,
                                           requests[1].request_id: 2
                                       },
                                       total_num_scheduled_tokens=3,
                                       scheduled_encoder_inputs={},
                                       scheduled_spec_decode_tokens={
                                           requests[0].request_id: [],
                                           requests[1].request_id: [10]
                                       },
                                       num_common_prefix_blocks=0,
                                       finished_req_ids=set(),
                                       free_encoder_input_ids=[],
                                       structured_output_request_ids={},
                                       grammar_bitmask=None)

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={
            req.request_id: i
            for i, req in enumerate(requests)
        },
        sampled_token_ids=[[EOS_TOKEN_ID],
                           [10,
                            11]],  # First request hits EOS, second continues
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={})

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped, second continues
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_STOPPED
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [EOS_TOKEN_ID]
    assert list(requests[1].output_token_ids) == [10, 11]

    # Test case 2: Stop on custom stop token
    scheduler = create_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=2,
                               max_tokens=10,
                               stop_token_ids=[42, 43])
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)

    scheduler_output = SchedulerOutput(scheduled_new_reqs=[],
                                       scheduled_cached_reqs=[],
                                       num_scheduled_tokens={
                                           requests[0].request_id: 3,
                                           requests[1].request_id: 2
                                       },
                                       total_num_scheduled_tokens=5,
                                       scheduled_encoder_inputs={},
                                       scheduled_spec_decode_tokens={
                                           requests[0].request_id: [10, 42],
                                           requests[1].request_id: [13]
                                       },
                                       num_common_prefix_blocks=0,
                                       finished_req_ids=set(),
                                       free_encoder_input_ids=[],
                                       structured_output_request_ids={},
                                       grammar_bitmask=None)

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={
            req.request_id: i
            for i, req in enumerate(requests)
        },
        sampled_token_ids=[[10, 42, 12],
                           [13, 14]],  # First request hits stop token
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={})

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped on custom token
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_STOPPED
    assert requests[0].stop_reason == 42
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [10, 42]
    assert list(requests[1].output_token_ids) == [13, 14]

    # Test case 3: Stop on max tokens
    scheduler = create_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=2, max_tokens=2)
    for req in requests:
        req.num_computed_tokens = req.num_tokens
        scheduler.requests[req.request_id] = req
        scheduler.running.append(req)

    scheduler_output = SchedulerOutput(scheduled_new_reqs=[],
                                       scheduled_cached_reqs=[],
                                       num_scheduled_tokens={
                                           requests[0].request_id: 3,
                                           requests[1].request_id: 1
                                       },
                                       total_num_scheduled_tokens=4,
                                       scheduled_encoder_inputs={},
                                       scheduled_spec_decode_tokens={
                                           requests[0].request_id: [10, 11],
                                           requests[1].request_id: []
                                       },
                                       num_common_prefix_blocks=0,
                                       finished_req_ids=set(),
                                       free_encoder_input_ids=[],
                                       structured_output_request_ids={},
                                       grammar_bitmask=None)

    model_output = ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index={
            req.request_id: i
            for i, req in enumerate(requests)
        },
        sampled_token_ids=[[10, 11, 12],
                           [13]],  # First request exceeds max_tokens
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={})

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify first request stopped due to length
    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == requests[1].request_id
    assert requests[0].status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert requests[0].request_id in scheduler.finished_req_ids
    assert list(requests[0].output_token_ids) == [10, 11
                                                  ]  # Truncated to max_tokens
    assert list(requests[1].output_token_ids) == [13]

    # Test case 4: Ignore EOS flag
    scheduler = create_scheduler(num_speculative_tokens=2)
    requests = create_requests(num_requests=1, max_tokens=10)
    requests[0].sampling_params.ignore_eos = True
    requests[0].num_computed_tokens = requests[0].num_tokens
    scheduler.requests[requests[0].request_id] = requests[0]
    scheduler.running.append(requests[0])

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={requests[0].request_id: 3},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={
            requests[0].request_id: [EOS_TOKEN_ID, 10]
        },
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None)

    model_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[EOS_TOKEN_ID, 10, 11]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={})

    scheduler.update_from_output(scheduler_output, model_output)

    # Verify request continues past EOS
    assert len(scheduler.running) == 1
    assert not requests[0].is_finished()
    assert list(requests[0].output_token_ids) == [EOS_TOKEN_ID, 10, 11]


@pytest.mark.parametrize("enable_prefix_caching, prompt_logprobs", [
    (None, None),
    (True, 5),
])
def test_schedule_concurrent_batches(enable_prefix_caching: Optional[bool],
                                     prompt_logprobs: Optional[int]):
    scheduler = create_scheduler(
        max_num_batched_tokens=1024,
        max_num_seqs=2,
        enable_prefix_caching=enable_prefix_caching,
    )
    requests = create_requests(
        num_requests=2,
        num_tokens=512,
        prompt_logprobs=prompt_logprobs,
    )

    # Schedule the first request.
    scheduler.add_request(requests[0])
    scheduler_output0 = scheduler.schedule()
    assert len(scheduler_output0.scheduled_new_reqs) == 1
    assert scheduler_output0.num_scheduled_tokens[
               requests[0].request_id] == 512

    # The first request is still running, so only schedule the second request.
    scheduler.add_request(requests[1])
    scheduler_output1 = scheduler.schedule()
    assert len(scheduler_output1.scheduled_new_reqs) == 1
    assert scheduler_output1.num_scheduled_tokens[
               requests[1].request_id] == 512

    # Model output of the first request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[0].request_id],
        req_id_to_index={requests[0].request_id: 0},
        sampled_token_ids=[[0]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(scheduler_output0, model_runner_output)

    # Schedule the next step.
    # The first request can be scheduled again while the second
    # request is still running.
    scheduler_output2 = scheduler.schedule()
    assert scheduler_output2.num_scheduled_tokens[requests[0].request_id] == 1

    # Model output of the second request.
    model_runner_output = ModelRunnerOutput(
        req_ids=[requests[1].request_id],
        req_id_to_index={requests[1].request_id: 0},
        sampled_token_ids=[[0]],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    scheduler.update_from_output(scheduler_output1, model_runner_output)


# Note - these test cases mirror some of those in test_rejection_sampler.py
@pytest.mark.parametrize(
    "spec_tokens,output_tokens,expected",
    [
        ([[1, 2, 3]], [[1, 2, 3, 4]], (1, 3, 3, [1, 1, 1])),  # perfect match
        ([[1, 2, 3]], [[1, 5]], (1, 3, 1, [1, 0, 0])),  # early mismatch
        ([[1, 2], [3]], [[1, 2, 5], [3, 4]],
         (2, 3, 3, [2, 1])),  # multiple sequences
        ([[1]], [[1, 2]], (1, 1, 1, [1])),  # single token sequence
        ([[]], [[5]], (0, 0, 0, [0])),  # empty sequence
        ([[1, 2, 3], [4, 5, 6]], [[1, 2, 7], [4, 8]],
         (2, 6, 3, [2, 1, 0])),  # multiple mismatches
    ])
def test_schedule_spec_decoding_stats(spec_tokens, output_tokens, expected):
    """
    Test scheduling behavior with speculative decoding.

    """
    num_spec_tokens = max(1, max(len(t) for t in spec_tokens))
    scheduler = create_scheduler(num_speculative_tokens=num_spec_tokens)
    requests = create_requests(num_requests=len(spec_tokens), num_tokens=1)
    req_ids = []
    req_to_index = {}
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        req_ids.append(request.request_id)
        req_to_index[request.request_id] = i

    # Schedule a decode, which will also draft speculative tokens
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.total_num_scheduled_tokens == len(requests)
    for i in range(len(requests)):
        req_id = requests[i].request_id
        assert output.num_scheduled_tokens[req_id] == 1
        assert req_id not in output.scheduled_spec_decode_tokens

    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=[[0] for _ in range(len(requests))],
        spec_token_ids=spec_tokens,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    engine_core_outputs = scheduler.update_from_output(output,
                                                       model_runner_output)

    for i in range(1, len(requests)):
        running_req = scheduler.running[i]
        # The prompt token
        assert running_req.num_computed_tokens == 1
        # The prompt token and the sampled token
        assert running_req.num_tokens == 2

    # No draft or accepted tokens counted yet
    assert engine_core_outputs.scheduler_stats.spec_decoding_stats is None

    # Schedule the speculated tokens for validation
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 0
    # The sampled token and speculated tokens
    assert output.total_num_scheduled_tokens == \
           len(requests) + sum(len(ids) for ids in spec_tokens)
    for i in range(len(requests)):
        req_id = requests[i].request_id
        assert output.num_scheduled_tokens[req_id] == 1 + len(spec_tokens[i])
        if spec_tokens[i]:
            assert len(output.scheduled_spec_decode_tokens[req_id]) == \
                   len(spec_tokens[i])
        else:
            assert req_id not in output.scheduled_spec_decode_tokens

    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_to_index,
        sampled_token_ids=output_tokens,
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )
    engine_core_outputs = scheduler.update_from_output(output,
                                                       model_runner_output)

    scheduler_stats = engine_core_outputs.scheduler_stats
    if expected[0] == 0:
        assert scheduler_stats.spec_decoding_stats is None
    else:
        assert scheduler_stats.spec_decoding_stats is not None
        stats = scheduler_stats.spec_decoding_stats
        assert stats.num_drafts == expected[0]


def _assert_right_scheduler_output(
        output: SchedulerOutput,
        num_requests: int,
        expected_num_scheduled_tokens: int,
):
    """Check if SchedulerOutput is correct after remote KV cache hit."""

    # We should inject the kv_connector_metadata.
    assert len(output.kv_connector_metadata.requests) == num_requests

    # Only num_tokens - matched_num_new_tokens should be scheduled.
    for _, num_scheduled_tokens in output.num_scheduled_tokens.items():
        assert num_scheduled_tokens == expected_num_scheduled_tokens


def _assert_right_kv_cache_manager(
        scheduler: Scheduler,
        req_ids: list[str],
        num_tokens: int,
        block_size: int,
        num_requests: int,
        num_total_blocks: int,
):
    """Check whether KVCacheManager is correct after allocate."""

    # Make sure the request stats are right.
    EXPECTED_TOTAL_BLOCKS = num_tokens // block_size
    for req_id in req_ids:
        blocks = (scheduler.kv_cache_manager.single_type_manager.
        req_to_blocks[req_id])
        hashes = scheduler.kv_cache_manager.req_to_block_hashes[req_id]
        assert (scheduler.kv_cache_manager.single_type_manager.
                num_cached_block[req_id] == EXPECTED_TOTAL_BLOCKS)
        assert len(blocks) == EXPECTED_TOTAL_BLOCKS
        assert len(hashes) == EXPECTED_TOTAL_BLOCKS

    # Make sure we actually touched all the blocks.
    BLOCKS_PER_REQ = num_tokens / block_size
    assert (scheduler.kv_cache_manager.block_pool.get_num_free_blocks() ==
            num_total_blocks - num_requests * BLOCKS_PER_REQ)


def _step_until_done(
        scheduler: Scheduler,
        output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
):
    """Loop over schedule(), update_from_output() until finished."""

    all_finished = False
    _ = scheduler.update_from_output(output, model_runner_output)
    while not all_finished:
        # Schedule + a few iterations until stopping.
        output = scheduler.schedule()
        assert len(scheduler.running)
        for _, num_scheduled_tokens in output.num_scheduled_tokens.items():
            # We should be in the decode phase now.
            assert num_scheduled_tokens == 1
        assert len(output.kv_connector_metadata.requests) == 0
        ecos = scheduler.update_from_output(output, model_runner_output)
        all_done = True
        for eco in ecos.outputs:
            if eco.finish_reason is None:
                all_done = False
        all_finished = all_done


def make_output(scheduler: Scheduler):
    return ModelRunnerOutput(
        req_ids=[req.request_id for req in scheduler.running],
        req_id_to_index={
            req.request_id: i
            for i, req in enumerate(scheduler.running)
        },
        sampled_token_ids=[[1000]] * len(scheduler.running),
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )


def assert_scheduler_empty(scheduler: Scheduler):
    """Confirm the scheduler is "empty" - i.e. no leaks."""
    # Scheduler Metadata.
    assert len(scheduler.requests) == 0
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.finished_req_ids) == 0
    assert len(scheduler._cached_reqs_data) == 0

    # EncoderCacheManager.
    assert len(scheduler.encoder_cache_manager.freed) == 0
    assert len(scheduler.encoder_cache_manager.cached) == 0

    # KVCache Manager.
    assert len(
        scheduler.kv_cache_manager.single_type_manager.req_to_blocks) == 0
    assert len(scheduler.kv_cache_manager.req_to_block_hashes) == 0
    assert len(
        scheduler.kv_cache_manager.single_type_manager.num_cached_block) == 0
    num_free_blocks = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks)
    assert num_free_blocks == (
            scheduler.kv_cache_manager.block_pool.num_gpu_blocks - 1)

    for block in scheduler.kv_cache_manager.block_pool.blocks:
        assert block.ref_cnt == 0


def test_memory_leak():
    """Test that we do not have a memory leak."""

    scheduler = create_scheduler(enable_prefix_caching=True)

    NUM_REQUESTS = 5
    NUM_TOKENS = 10
    MAX_TOKENS = 10
    requests = create_requests(num_requests=NUM_REQUESTS,
                               num_tokens=NUM_TOKENS,
                               max_tokens=MAX_TOKENS)

    # Add each request.
    for request in requests:
        scheduler.add_request(request)
        scheduler_output = scheduler.schedule()
        model_runner_output = make_output(scheduler)
        scheduler.update_from_output(scheduler_output, model_runner_output)

    # Iterate until done.
    while True:
        scheduler_output = scheduler.schedule()
        if len(scheduler.running) == 0:
            break
        model_runner_output = make_output(scheduler)
        scheduler.update_from_output(scheduler_output, model_runner_output)

    # Confirm no memory leak.
    assert_scheduler_empty(scheduler)
