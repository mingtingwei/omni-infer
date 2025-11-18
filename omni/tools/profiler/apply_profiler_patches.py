# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import os
from pathlib import Path
import logging
import yaml, json
import sys
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from tools.profiler.utils import safe_print, ip_str, trace_output_directory
from tools.profiler.prof_wrapper import (torchnpu_prof_wrapper, 
    timer_prof_wrapper, viztracer_prof_wrapper, marker_prof_wrapper)
import time
from typing import Optional, List, Tuple
from omni.tools.profiler.trace.tracing import init_tracer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

wrapper_dict = {
    "torchnpu": torchnpu_prof_wrapper, 
    "timer": timer_prof_wrapper, 
    "viztracer": viztracer_prof_wrapper, 
    "marker": marker_prof_wrapper
}

# Parse config from namelist, apply profiler monkey patch
def apply_patches(namelist_path: str):
    try:
        namelist_file = Path(__file__).parent / namelist_path

        # Load namelist
        with namelist_file.open('r') as f:
            config = yaml.safe_load(f)

        profiler_type = config.get('type')
        if not (profiler_type=='torchnpu' or 
                profiler_type=='timer' or 
                profiler_type=='viztracer' or
                profiler_type=='marker'):
            logger.error(f"<<<type of namelist invalid, should be one of torchnpu/timer/viztracer/marker")
            raise RuntimeError("<<<type of namelist invalid, should be one of torchnpu/timer/viztracer/marker")
        logger.info(f"<<<Applying {profiler_type} profiler patches from {namelist_path}")
        wrapper_method = wrapper_dict[profiler_type]
        
        base_params = config.get("base_params", {})

        # Extract target modules and methods
        targets: List[Tuple[str, Optional[str]]] = []
        for target in config.get('targets', []):
            module_name = target.get('module')
            class_name = None
            if ":" in module_name:
                module_name, class_name = module_name.split(":")
            function_name = target.get('function_name')
            entry_operation = target.get('entry_operation', None)
            exit_operation = target.get('exit_operation', None)
            entry_message = target.get('entry_message', None)
            exit_message = target.get('exit_message', None)
            if module_name:
                targets.append(
                    (
                        module_name, 
                        class_name, 
                        function_name, 
                        (entry_operation, exit_operation), 
                        (entry_message, exit_message)
                    )
                )            
            else:
                logger.warning(f"<<<Skipping target with missing 'module': {target}")

        if not targets:
            logger.warning(f"<<<No valid targets found in {namelist_path}")
            return

        for module_name, class_name, function_name, \
                (entry_operation, exit_operation), \
                (entry_message, exit_message) in targets:
            logger.info(f"<<<Patching {module_name}.{function_name or 'all methods'}")
            try:
                original_module = importlib.import_module(module_name)

                base_params['entry_operation'] = entry_operation
                base_params['exit_operation'] = exit_operation
                base_params['entry_message'] = entry_message
                base_params['exit_message'] = exit_message
                if class_name:
                    try:
                        target_class = getattr(original_module, class_name)
                        try:
                            original_function = getattr(target_class, function_name)
                            wrapped_function = wrapper_method(original_function, base_params)
                            setattr(target_class, function_name, wrapped_function)
                            logger.info(f"<<<<{module_name}.{class_name}.{function_name} is wrapped")
                        except AttributeError:
                            logger.warning(
                                f"<<<Function '{function_name}' not found in class '{class_name}' "
                                f"of module '{module_name}'"
                            )
                            continue
                    except AttributeError:
                        logger.warning(f"<<<Class '{class_name}' not found in module '{module_name}'")
                        continue
                else:
                    try:
                        original_function = getattr(original_module, function_name)
                        wrapped_function = wrapper_method(original_function, base_params)
                        setattr(original_module, function_name, wrapped_function)
                        logger.info(f"<<<<{module_name}.{function_name} is wrapped")
                    except AttributeError:
                        logger.warning(f"<<<Function '{function_name}' not found in module '{module_name}'")
                        continue
            except ImportError as e:
                logger.warning(f"<<<Failed to import module '{module_name}': {str(e)}")
                continue
            except Exception as e:
                logger.warning(
                    f"<<<Unexpected error while wrapping {module_name}.{class_name or ''}."
                    f"{function_name}: {str(e)}"
                )
                continue

    except (FileNotFoundError, ImportError, AttributeError, RuntimeError, yaml.YAMLError) as e:
        logger.error(f"<<<Failed to apply model patches: {e}")
        raise

# following monkey patch is for triggering a printing message 
#   when a request enter WAITING_FOR_REMOTE_KVS status
def monkey_patch_request_status():
    patch_request()
    from vllm.v1.request import Request
    from vllm.v1.request import RequestStatus
    from omni.tools.profiler.trace.tracing import create_span
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    import time
    original_status = Request.__dict__.get('status', None)
    def status_getter(self):
        return self._status

    def status_setter(self, value):
        self._status = value
        self.waiting_pull_len += 1
        if value == RequestStatus.WAITING_FOR_REMOTE_KVS:
            if self.trace_headers:
                start_time = int(self.trace_headers.get("start_time_ns"))
                trace_headers = {"traceparent": self.trace_headers["traceparent"]}
                timestamp = time.time_ns()
                ctx = create_span(start_time_ns=start_time, end_time=timestamp, action_name="Add need pullling sequence", request_id=self.request_id, ip_str=ip_str, ctx_flag=False, trace_headers=trace_headers,)
                carrier = {}
                TraceContextTextMapPropagator().inject(carrier, context=ctx)
                carrier['start_time_ns'] = str(timestamp)
                carrier['ttft_start_time'] = self.trace_headers.get('ttft_start_time')
                carrier['ttft_traceparent'] = self.trace_headers.get('ttft_traceparent')
                self.trace_headers = carrier
            safe_print(
                trace_output_directory, 
                f"<<<Action: Add need pulling sequence; "
                f"Timestamp:{time.time()}; "
                f"RequestID:{self.request_id}; "
                f"Role:{os.getenv('ROLE')}_{ip_str}"
            )
    Request.status = property(status_getter, status_setter)
    original_init = Request.__init__
    def new_init(self, *args, **kwargs):
        self.waiting_pull_len = 0
        original_init(self, *args, **kwargs)
        self._status = self.status

    Request.__init__ = new_init
    print("<<< Monkey patch request status is applied")

# following monkey patch is for marking the first token, second token and last token
def monkey_patch_async_generator_io_logger():
    from functools import wraps
    from typing import AsyncGenerator
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from omni.tools.profiler.trace.tracing import create_span, parent_ctx_var, clean_ctx, ttft_start_time, ttft_end_time, ttft_trace_id
    from vllm.logger import init_logger
    import logging, time
    logging.basicConfig(level=logging.DEBUG)
    logger = init_logger(__name__)

    original_method = OpenAIServingChat.chat_completion_stream_generator
    @wraps(original_method)
    async def new_method(self, *args, **kwargs) -> AsyncGenerator:
        yield_count = 0
        request_id = args[2] # get request_id
        async for item in original_method(self, *args, **kwargs):
            yield_count += 1
            if yield_count == 1:
                # First chat_completion_stream_generator yield.
                pass
            elif yield_count == 2:
                # Second chat_completion_stream_generator yield.
                headers = self.headers_map.get(request_id) or {}
                if headers.get('traceparent'):
                    logger.info(f"<<< output_trace_headers, req_id:{request_id}, trace_headers:{headers}")
                    timestamp = time.time_ns()
                    ttft_trace_id.set(headers.get("ttft_traceparent"))
                    ttft_start_time.set(headers.get('ttft_start_time'))
                    ttft_end_time.set(timestamp)

                    start_time = int(headers.get("start_time_ns"))
                    trace_headers = {"traceparent": headers["traceparent"]}
                    create_span(start_time_ns=start_time, end_time=timestamp, action_name="First decode output token", request_id=request_id, ip_str=ip_str, ctx_flag=False, trace_headers=trace_headers,)
                safe_print(trace_output_directory, f"<<<Action: First decode output token; Timestamp:{time.time()}; RequestID:{request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            elif yield_count == 3:
                if parent_ctx_var.get() is not None: 
                    timestamp = time.time_ns()
                    create_span(start_time_ns=0, end_time=timestamp, action_name="Second decode output token", request_id=request_id, ip_str=ip_str, ctx_flag=True, trace_headers=None,)
                # Third chat_completion_stream_generator yield.
                safe_print(trace_output_directory, f"<<<Action: Second decode output token; Timestamp:{time.time()}; RequestID:{request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            elif yield_count == 4:
                if parent_ctx_var.get() is not None:
                    timestamp = time.time_ns()
                    create_span(start_time_ns=0, end_time=timestamp, action_name="Third decode output token", request_id=request_id, ip_str=ip_str, ctx_flag=True, trace_headers=None,)
                # Third chat_completion_stream_generator yield.
                safe_print(trace_output_directory, f"<<<Action: Third decode output token; Timestamp:{time.time()}; RequestID:{request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            if item == "data: [DONE]\n\n":
                if parent_ctx_var.get() is not None:
                    timestamp = time.time_ns()
                    create_span(start_time_ns=0, end_time=timestamp, action_name="Finish decode pickle and start response", request_id=request_id, ip_str=ip_str, ctx_flag=True, trace_headers=None,)

                    ttft_trace_header = {"traceparent": ttft_trace_id.get()}
                    # start_time = int(ttft_start_time.get())
                    ttft_val = ttft_start_time.get()
                    start_time = int(ttft_val) if ttft_val is not None else 0
                    end_time = ttft_end_time.get()
                    if ttft_trace_header != "None" and start_time != 0:
                        ctx = create_span(start_time_ns=start_time, end_time=timestamp, action_name="Vllm tracing", request_id=request_id, ip_str=ip_str, ctx_flag=False, trace_headers=ttft_trace_header,)
                        carrier = {}
                        TraceContextTextMapPropagator().inject(carrier, context=ctx)
                        create_span(start_time_ns=start_time, end_time=end_time, action_name="TTFT (Time of first token measured at D node)", request_id=request_id, ip_str=ip_str, ctx_flag=False, trace_headers=carrier,)
                    clean_ctx()
                safe_print(trace_output_directory, f"<<<Action: Finish decode pickle and start response; Timestamp:{time.time()}; RequestID:{request_id}; Role:{os.getenv('ROLE')}_{ip_str}")
            yield item

    OpenAIServingChat.chat_completion_stream_generator = new_method
    print("<<< Monkey patch monkey_patch_async_generator_io_logger is applied")


# following monkey patch is for passing tracer span_context in Enginecore process
def patch_enginecorerequest():
    import vllm.v1.engine
    from typing import Optional, Mapping
    from msgspec import Struct
    class PatchedEngineCoreRequest(vllm.v1.engine.EngineCoreRequest, Struct, kw_only=True):
        trace_headers: Optional[Mapping[str, str]] = None
    vllm.v1.engine.EngineCoreRequest = PatchedEngineCoreRequest
    vllm.v1.engine.core.EngineCoreRequest = PatchedEngineCoreRequest
    import vllm.v1.request
    vllm.v1.request.EngineCoreRequest = PatchedEngineCoreRequest
    print("<<< Monkey patch patch_enginecorerequest is applied")

def patch_enginecoreoutput():
    import vllm.v1.engine
    import msgspec, time
    from typing import Optional, Mapping
    from msgspec import Struct
    from vllm.v1.metrics.stats import SchedulerStats
    class PatchedEngineCoreOutput(vllm.v1.engine.EngineCoreOutput, Struct, kw_only=True):
        trace_headers: Optional[Mapping[str, str]] = None
    vllm.v1.engine.EngineCoreOutput = PatchedEngineCoreOutput
    vllm.v1.engine.core.EngineCoreOutput = PatchedEngineCoreOutput
    vllm.v1.core.sched.scheduler.EngineCoreOutput = PatchedEngineCoreOutput
    import vllm.v1.engine.core_client
    vllm.v1.engine.core_client.EngineCoreOutput = PatchedEngineCoreOutput
    from vllm.v1.engine import UtilityOutput
    class PatchedEngineCoreOutputs(msgspec.Struct,array_like=True,  omit_defaults=True,  gc=False): 
        engine_index: int = 0
        outputs: list[PatchedEngineCoreOutput] = []
        scheduler_stats: Optional[SchedulerStats] = None
        timestamp: float = 0.0
        utility_output: Optional[UtilityOutput] = None
        finished_requests: Optional[set[str]] = None
        wave_complete: Optional[int] = None
        start_wave: Optional[int] = None
        def __post_init__(self):
            if self.timestamp == 0.0:
                self.timestamp = time.monotonic()
    vllm.v1.engine.EngineCoreOutputs = PatchedEngineCoreOutputs
    vllm.v1.engine.core.EngineCoreOutputs = PatchedEngineCoreOutputs
    vllm.v1.core.sched.scheduler.EngineCoreOutputs = PatchedEngineCoreOutputs
    vllm.v1.engine.core_client.EngineCoreOutputs = PatchedEngineCoreOutputs
    print("<<< Monkey patch patch_enginecoreoutput is applied")

def patch_request():
    from vllm.v1.request import Request
    original_init = Request.__init__
    def new_init(self, *args, **kwargs):
        trace_headers = kwargs.pop('trace_headers', None)
        original_init(self, *args, **kwargs)
        self.trace_headers = trace_headers
    Request.__init__ = new_init
    import vllm.v1.request
    vllm.v1.request.Request.__init__ = new_init
    patch_enginecorerequest()
    from vllm.v1.engine import EngineCoreRequest
    from vllm.multimodal.inputs import MultiModalKwargs
    from vllm.utils import is_list_of
    from vllm.v1.structured_output.request import StructuredOutputRequest
    @classmethod
    def patch_from_engine_core_request(cls, request: EngineCoreRequest) -> "Request":
        if request.mm_inputs is not None:
            assert isinstance(request.mm_inputs, list)
            assert is_list_of(request.mm_inputs, MultiModalKwargs), (
                "mm_inputs was not updated in EngineCore.add_request")
        return cls(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            multi_modal_inputs=request.mm_inputs,
            multi_modal_hashes=request.mm_hashes,
            multi_modal_placeholders=request.mm_placeholders,
            sampling_params=request.sampling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            structured_output_request=StructuredOutputRequest(sampling_params=request.sampling_params),
            cache_salt=request.cache_salt,
            trace_headers = request.trace_headers,
        )
    Request.from_engine_core_request = patch_from_engine_core_request
    print("<<< Monkey patch patch_request is applied")

def patch_requestoutput():
    from vllm.outputs import RequestOutput
    original_init = RequestOutput.__init__
    def new_init(self, *args, **kwargs):
        trace_headers = kwargs.pop('trace_headers', None)
        original_init(self, *args, **kwargs)
        self.trace_headers = trace_headers
    RequestOutput.__init__ = new_init
    import vllm.outputs
    vllm.outputs.RequestOutput.__init__ = new_init

    from vllm.logger import init_logger
    import logging, time
    logging.basicConfig(level=logging.INFO)
    logger = init_logger(__name__)
    patch_enginecoreoutput()
    from vllm.v1.engine import FinishReason
    from vllm.v1.core.sched.scheduler import EngineCoreOutputs
    from vllm.v1.core.sched.scheduler import EngineCoreOutput
    import vllm.v1.engine.output_processor as otpro
    from vllm.v1.engine.output_processor import OutputProcessorOutput
    def _patch_process_outputs(self, engine_core_outputs, engine_core_timestamp, iteration_stats):
        request_outputs: list[RequestOutput] = []
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)

            # if getattr(engine_core_output, "trace_headers", {}):
            trace_headers = engine_core_output.trace_headers

            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # 1) Compute stats for this iteration.
            self._update_stats_from_output(req_state, engine_core_output,
                                           engine_core_timestamp,
                                           iteration_stats)

            new_token_ids = engine_core_output.new_token_ids
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason
            kv_transfer_params = engine_core_output.kv_transfer_params
            num_cached_tokens = engine_core_output.num_cached_tokens
            req_state.is_prefilling = False

            # 2) Detokenize the token ids into text and perform stop checks.
            stop_string = req_state.detokenizer.update(
                new_token_ids, finish_reason == FinishReason.STOP)
            if stop_string:
                finish_reason = FinishReason.STOP
                stop_reason = stop_string

            # 3) Compute sample and prompt logprobs for request, if required.
            req_state.logprobs_processor.update_from_output(engine_core_output)

            # 4) Create and handle RequestOutput objects.
            if request_output := req_state.make_request_output(
                    new_token_ids, finish_reason, stop_reason,
                    kv_transfer_params, num_cached_tokens):

                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    if trace_headers is not None:
                        req_state.trace_headers = trace_headers
                    request_output.trace_headers = getattr(req_state, "trace_headers", None)
                    req_state.queue.put(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    if trace_headers is not None:
                        req_state.trace_headers = trace_headers
                    request_output.trace_headers = getattr(req_state, "trace_headers", None)
                    request_outputs.append(request_output)

            # Free completed requests.
            if finish_reason is not None:
                self.request_states.pop(req_id)
                # Remove parent request if applicable.
                parent_req = req_state.parent_req
                if parent_req and not parent_req.child_requests:
                    self.parent_requests.pop(parent_req.request_id, None)
                if not engine_core_output.finished:
                    # If req not finished in EngineCore, but Detokenizer
                    # detected stop string, abort needed in EngineCore.
                    reqs_to_abort.append(req_id)

                # Track per-request stats
                self._update_stats_from_finished(req_state, finish_reason,
                                                 iteration_stats)

        self.lora_states.update_iteration_stats(iteration_stats)
        # for  pro_output in request_outputs:
            # logger.info(f"<<< RequestOutput.trace_headers, req_id:{req_id}, trace_headers:{pro_output.trace_headers}")
        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )
    otpro.OutputProcessor.process_outputs = _patch_process_outputs

    print("<<< Monkey patch patch_requestoutput is applied")

def patch_scheduler():
    import vllm.v1.core.sched.scheduler as sch
    from typing import Optional, Mapping
    _orig_init = sch.Scheduler.__init__
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.traced_waiting_ids: set[str] = set()
    sch.Scheduler.__init__ = _patched_init

    patch_enginecoreoutput()
    from vllm.v1.core.sched.scheduler import EngineCoreOutputs
    from vllm.v1.core.sched.scheduler import EngineCoreOutput
    import vllm.v1.core.sched.output as otp
    orig_init = otp.NewRequestData.__init__
    def new_init(self, *args, **kwargs):
        trace_headers = kwargs.pop('trace_headers', None)
        orig_init(self, *args, **kwargs)
        self.trace_headers = trace_headers
    otp.NewRequestData.__init__ = new_init
    @classmethod
    def patch_from_request(cls, request, block_ids,):
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_inputs=request.mm_inputs,
            mm_hashes=request.mm_hashes,
            mm_positions=request.mm_positions,
            sampling_params=request.sampling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            trace_headers=request.trace_headers,
        )
    otp.NewRequestData.from_request = patch_from_request

    _orig_update_from_output = sch.Scheduler.update_from_output
    def _patch_update_from_output(self, scheduler_output, model_output, num_steps):
        engine_core_outputs = _orig_update_from_output(self, scheduler_output, model_output, num_steps)
        headers_map = {}
        for req in scheduler_output.scheduled_new_reqs:
            if getattr(req, "trace_headers", None): #req.trace_headers:
                headers_map[req.req_id] = req.trace_headers
        
        for core_output in engine_core_outputs.outputs:
            core_output.trace_headers = headers_map.get(core_output.request_id)
        return engine_core_outputs
    sch.Scheduler.update_from_output = _patch_update_from_output
    print("<<< Monkey patch patch_scheduler is applied")


def patch_create_chat_completion_api():
    from vllm.entrypoints.openai.api_server import chat, completion, base
    from vllm.entrypoints.openai.protocol import ErrorResponse
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from vllm.entrypoints.openai import api_server
    from fastapi import Request, HTTPException
    from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                CompletionRequest, ChatCompletionResponse)
    from typing import Union
    from http import HTTPStatus
    from vllm.logger import init_logger
    import logging, time
    logging.basicConfig(level=logging.DEBUG)
    logger = init_logger(__name__)

    async def new_create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
        handler = chat(raw_request)
        if handler is None:
            return base(raw_request).create_error_response(
                message="The model does not support Chat Completions API")

        generator = await handler.create_chat_completion(request, raw_request)
        # logger.warning(f"<<< generator's type is {type(generator)}")
        if isinstance(generator, ErrorResponse):
            response = JSONResponse(content=generator.model_dump(),
                                status_code=generator.code)
            response.headers["X-Request-Id"] = generator.id
            return response
        elif isinstance(generator, ChatCompletionResponse):
            response = JSONResponse(content=generator.model_dump())
            trace_headers = (generator.kv_transfer_params or {}).get("trace_headers", {})
            if trace_headers:
                response.headers["X-Request-Id"] = generator.id
                response.headers["Traceparent"] = trace_headers.get("traceparent")
                response.headers["Start_time_ns"] = trace_headers.get("start_time_ns")
                # start_time = trace_headers.get("start_time_ns")
                # logger.info(f"*****1111111 Add header P to proxy, req_id:{request.request_id}, trace_headers:{trace_headers}, Start_time_ns:{start_time}")
            return response
        return StreamingResponse(content=generator, media_type="text/event-stream")

    for route in api_server.router.routes:
        if getattr(route, "path", None) == "/v1/chat/completions":
            route.endpoint = new_create_chat_completion
            print("<<< Monkey patch route.endpoint is applied")

    import vllm.entrypoints.openai.api_server
    vllm.entrypoints.openai.api_server.create_chat_completion = new_create_chat_completion

    print("<<< Monkey patch patch_create_chat_completion_api is applied")



# following monkey patch is for passing tracer span_context inside process_input_socket function
def patched_process_inputs():
    from vllm.v1.engine.processor import Processor
    import time
    from collections.abc import Mapping, Sequence
    from typing import Any, Literal, Optional, Union

    from vllm.inputs import ProcessorInputs, PromptType
    from vllm.inputs.parse import split_enc_dec_inputs
    from vllm.inputs.preprocess import InputPreprocessor
    from vllm.lora.request import LoRARequest
    from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalKwargs,
                             MultiModalRegistry)
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.multimodal.utils import merge_and_sort_multimodal_metadata
    from vllm.pooling_params import PoolingParams
    from vllm.prompt_adapter.request import PromptAdapterRequest
    from vllm.sampling_params import SamplingParams
    # from vllm.transformers_utils.tokenizer_group import TokenizerGroup
    from vllm.v1.engine import EngineCoreRequest
    # from vllm.v1.engine.mm_input_cache import MirroredProcessingCache
    def my_process_inputs(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> tuple[Optional[str], EngineCoreRequest]:

        # TODO(woosuk): Support pooling models.
        # TODO(woosuk): Support encoder-decoder models.
        self._validate_lora(lora_request)
        self._validate_params(params, lora_request)
        if priority != 0:
            raise ValueError("V1 does not support priority yet.")
        if trace_headers is not None:
            pass
            # raise ValueError("V1 does not support tracing yet.")
        if prompt_adapter_request is not None:
            raise ValueError("V1 does not support prompt_adapter_request.")

        if arrival_time is None:
            arrival_time = time.time()

        processed_inputs: ProcessorInputs = self.input_preprocessor.preprocess(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            return_mm_hashes=self.use_hash,
        )
        from vllm.platforms import current_platform
        current_platform.validate_request(
            prompt=prompt,
            params=params,
            processed_inputs=processed_inputs,
        )
        eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

        self._validate_model_inputs(processed_inputs, lora_request)

        encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)

        # TODO: Impl encoder-decoder
        if encoder_inputs is not None:
            raise NotImplementedError

        assert isinstance(params, SamplingParams)
        # TODO: can we avoid cloning here in multiproc case?
        sampling_params = params.clone()
        # If unset max tokens, then generate up to the max_model_len.
        if sampling_params.max_tokens is None:
            sampling_params.max_tokens = (
                self.model_config.max_model_len -
                len(decoder_inputs["prompt_token_ids"]))
        sampling_params.update_from_generation_config(
            self.generation_config_fields, eos_token_id)
        sampling_params.update_from_tokenizer(
            self.tokenizer.get_lora_tokenizer(lora_request))

        # Multimodal related.
        sorted_mm_inputs: Optional[Sequence[Optional[MultiModalKwargs]]] = None
        sorted_mm_positions: Optional[list[PlaceholderRange]] = None
        sorted_mm_hashes: Optional[list[str]] = None
        if decoder_inputs["type"] == "multimodal":
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]

            (
                sorted_item_modalities,
                sorted_mm_positions,
                sorted_mm_hashes,
            ) = merge_and_sort_multimodal_metadata(
                decoder_inputs["mm_placeholders"],
                decoder_inputs["mm_hashes"] if self.use_hash else None,
            )

            unique_modalities = set(sorted_item_modalities)
            if len(unique_modalities) > 1:
                orig_sorted_mm_inputs = []
                used_indices = {modality: 0 for modality in unique_modalities}

                for modality in sorted_item_modalities:
                    items = decoder_mm_inputs.get_items(modality)
                    item = items[used_indices[modality]]

                    orig_sorted_mm_inputs.append(
                        MultiModalKwargs.from_items([item]))
                    used_indices[modality] += 1
            else:
                orig_sorted_mm_inputs = [
                    MultiModalKwargs.from_items([item]) for item in
                    decoder_mm_inputs.get_items(sorted_item_modalities[0])
                ]

            if sorted_mm_hashes is not None:
                sorted_mm_inputs = self.mm_input_cache_client.get_and_update_p0(
                    orig_sorted_mm_inputs, sorted_mm_hashes)
            else:
                sorted_mm_inputs = orig_sorted_mm_inputs

        return decoder_inputs.get("prompt"), EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=decoder_inputs["prompt_token_ids"],
            mm_inputs=sorted_mm_inputs,
            mm_hashes=sorted_mm_hashes,
            mm_placeholders=sorted_mm_positions,
            sampling_params=sampling_params,
            eos_token_id=eos_token_id,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_inputs.get("cache_salt"),
            trace_headers=trace_headers,
        )
    Processor.process_inputs = my_process_inputs
    print("<<< Monkey patch patched_process_inputs is applied")

# following monkey patch is for passing raw request_id inside _preprocess_chat function
def patch_chatcompletionrequest():
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    OriginalChatCompletionRequest = ChatCompletionRequest
    class PatchedChatCompletionRequest(OriginalChatCompletionRequest):
        req_id: Optional[str] = None
    ChatCompletionRequest = PatchedChatCompletionRequest
    print("<<< Monkey patch patch_chatcompletionrequest is applied")


profiling_namelist = os.getenv("PROFILING_NAMELIST", None)
if profiling_namelist is not None:
    if os.path.isfile(profiling_namelist):
        init_tracer(service_name="vllm-service")
        patch_chatcompletionrequest()
        patch_create_chat_completion_api()
        monkey_patch_request_status()
        patch_enginecorerequest()
        patch_enginecoreoutput()
        patched_process_inputs()
        patch_request()
        patch_scheduler()
        patch_requestoutput()
        apply_patches(profiling_namelist)
        monkey_patch_async_generator_io_logger()
    else:
        logger.error(f"'{profiling_namelist}' does not exist.")
        raise FileNotFoundError(f"'{profiling_namelist}' does not exist.")