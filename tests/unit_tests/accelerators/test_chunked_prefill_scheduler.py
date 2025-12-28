import os
from pathlib import Path
from unittest.mock import patch
import sys
import torch
from typing import Optional
from unittest.mock import Mock

from vllm.config import (CacheConfig, KVTransferConfig, ModelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

EOS_TOKEN_ID = 50256


def create_scheduler(
    model: str = "facebook/opt-125m",
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
) -> Scheduler:
    '''Create scheduler under test.

    Args:
      model: model under test
      max_num_seqs: max sequences to schedule
      max_num_batch_tokens: max num tokens to batch
      enable_prefix_caching: optionally force APC config
                             (True/False) or use default
                             (None)

    Returns:
      {class}`Scheduler` instance
    '''
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
    # Cache config, optionally force APC
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

def test_no_chunk():
    """Prefills should not be chunked without `FORCE_ENABLE_CHUNK_PREFILL` set."""

    block_size = 4
    max_seqs = 60
    max_model_len = 80
    max_num_batched_tokens = 64

    # Use cached model 
    CUR_DIR = Path(__file__).parent
    model_path=f"{CUR_DIR}/mock_model/"

    scheduler = create_scheduler(model=model_path,
                                max_num_seqs=max_seqs,
                                max_num_batched_tokens=max_num_batched_tokens,
                                num_blocks=32,
                                block_size= block_size,
                                max_model_len=max_model_len,)
    requests = create_requests(num_requests=2,
                            num_tokens=60)
    for request in requests:
        scheduler.add_request(request)

    output = scheduler.schedule()
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 1
    assert len(output.scheduled_new_reqs) == 1
    assert len(output.scheduled_cached_reqs) == 0
    assert output.num_scheduled_tokens[requests[0].request_id] == 60
    assert output.total_num_scheduled_tokens == 60
    requests[0].append_output_token_ids(1)

    output = scheduler.schedule()
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == len(requests)
    assert len(output.scheduled_new_reqs) == 1
    assert len(output.scheduled_cached_reqs) == 1
    assert output.num_scheduled_tokens[requests[1].request_id] == 60
    assert output.num_scheduled_tokens[requests[0].request_id] == 1
    assert output.total_num_scheduled_tokens == 61

def test_chunk():
    """Verify prefills are chunked properly."""

    env_vars = {
        "FORCE_ENABLE_CHUNK_PREFILL": "1",
    }

    block_size = 4
    max_seqs = 60
    max_model_len = 80
    max_num_batched_tokens = 64

    # Use cached model 
    CUR_DIR = Path(__file__).parent
    model_path=f"{CUR_DIR}/mock_model/"

    with patch.dict(os.environ, env_vars):
        # Reload module with new env var
        from importlib import reload
        import vllm.v1.core.sched.scheduler

        reload(vllm.v1.core.sched.scheduler)

        scheduler = create_scheduler(model=model_path,
                                    max_num_seqs=max_seqs,
                                    max_num_batched_tokens=max_num_batched_tokens,
                                    num_blocks=32,
                                    block_size= block_size,
                                    max_model_len=max_model_len,)
        requests = create_requests(num_requests=2,
                                num_tokens=60)
        for request in requests:
            scheduler.add_request(request)

        output = scheduler.schedule()
        assert len(scheduler.waiting) == 0
        assert len(scheduler.running) == len(requests)
        assert len(output.scheduled_new_reqs) == len(requests)
        assert len(output.scheduled_cached_reqs) == 0
        assert output.num_scheduled_tokens[requests[0].request_id] == 60
        assert output.num_scheduled_tokens[requests[1].request_id] == 4
        assert output.total_num_scheduled_tokens == 64
        requests[0].append_output_token_ids(1)

        output = scheduler.schedule()
        assert len(scheduler.waiting) == 0
        assert len(scheduler.running) == len(requests)
        assert len(output.scheduled_new_reqs) == 0
        assert len(output.scheduled_cached_reqs) == len(requests)
        assert output.num_scheduled_tokens[requests[0].request_id] == 1
        assert output.num_scheduled_tokens[requests[1].request_id] == 56
        assert output.total_num_scheduled_tokens == 57
        
    if 'vllm.v1.core.sched.scheduler' in sys.modules:
        del sys.modules['vllm.v1.core.sched.scheduler']
