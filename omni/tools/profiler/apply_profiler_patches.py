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
                headers = self.headers_map.pop(request_id, {})
                if headers and headers.get('traceparent'):
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
    import vllm.v1.engine.processor
    vllm.v1.engine.processor.EngineCoreRequest = PatchedEngineCoreRequest
    vllm.v1.request.EngineCoreRequest = PatchedEngineCoreRequest
    print("<<< Monkey patch patch_enginecorerequest is applied")

def patch_enginecoreoutput():
    import vllm.v1.engine
    from typing import Optional, Mapping
    from msgspec import Struct
    class PatchedEngineCoreOutput(vllm.v1.engine.EngineCoreOutput, Struct, kw_only=True):
        trace_headers: Optional[Mapping[str, str]] = None
    vllm.v1.engine.EngineCoreOutput = PatchedEngineCoreOutput
    vllm.v1.engine.core.EngineCoreOutput = PatchedEngineCoreOutput
    vllm.v1.core.sched.scheduler.EngineCoreOutput = PatchedEngineCoreOutput
    import vllm.v1.engine.core_client
    vllm.v1.engine.core_client.EngineCoreOutput = PatchedEngineCoreOutput

    class PatchedEngineCoreOutputs(vllm.v1.engine.EngineCoreOutputs):
        outputs: list[PatchedEngineCoreOutput] = []
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

    orig_from_engine_core_request = Request.from_engine_core_request.__func__
    def patch_from_engine_core_request(cls, request):
        outputs = orig_from_engine_core_request(cls, request)
        outputs.trace_headers = request.trace_headers
        return outputs
    Request.from_engine_core_request = classmethod(patch_from_engine_core_request)

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
    import vllm.v1.engine.output_processor
    vllm.v1.engine.output_processor.RequestOutput.__init__ = new_init
    vllm.outputs.RequestOutput.__init__ = new_init

    import vllm.v1.engine.output_processor as otpro
    _origin_process_outputs = otpro.OutputProcessor.process_outputs
    def _patch_process_outputs(self, engine_core_outputs, engine_core_timestamp, iteration_stats):
        headers_map = {}
        req_state_map = {}
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            trace_headers = engine_core_output.trace_headers
            headers_map[req_id] = trace_headers
            req_state_map[req_id] = self.request_states.get(req_id)
        _outputs = _origin_process_outputs(self, engine_core_outputs, engine_core_timestamp, iteration_stats)

        # AsyncLLM: put into queue for handling by generate().
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            trace_headers = headers_map[req_id]
            req_state = req_state_map[req_id]
            if req_state is None:
                continue
            if req_state.queue is not None and trace_headers and req_state.queue.output is not None:
                req_state.queue.output.trace_headers = trace_headers

        for request_output in _outputs.request_outputs:
            req_id = request_output.request_id
            req_state = self.request_states.get(req_id)
            trace_headers = headers_map.pop(req_id, None) #headers_map[req_id]
            req_state = req_state_map.pop(req_id, None)
            if req_state is None:
                continue
            if req_state.queue is not None:
                pass
            else:
                # LLMEngine: return list of RequestOutputs.
                if trace_headers:
                    request_output.trace_headers = getattr(req_state, "trace_headers", None)
        return _outputs
    otpro.OutputProcessor.process_outputs = _patch_process_outputs
    print("<<< Monkey patch patch_requestoutput is applied")


def patch_scheduler():
    import vllm.v1.core.sched.scheduler as sch
    from typing import Optional, Mapping
    from collections import OrderedDict
    _orig_init = sch.Scheduler.__init__
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.traced_waiting_ids: 'OrderedDict[str, None]' = OrderedDict() #set[str] = set()
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

    _orig_from_request = otp.NewRequestData.from_request.__func__
    def patch_from_request(cls, request, block_ids,):
        outputs = _orig_from_request(cls, request, block_ids,)
        outputs.trace_headers = request.trace_headers
        return outputs
    otp.NewRequestData.from_request = classmethod(patch_from_request)

    _orig_update_from_output = sch.Scheduler.update_from_output
    def _patch_update_from_output(self, scheduler_output, model_output, num_steps):
        engine_core_outputs = _orig_update_from_output(self, scheduler_output, model_output, num_steps)
        headers_map = {}
        for req in scheduler_output.scheduled_new_reqs:
            if getattr(req, "trace_headers", None): #req.trace_headers:
                headers_map[req.req_id] = req.trace_headers
        
        for core_output in engine_core_outputs.outputs:
            core_output.trace_headers = headers_map.pop(core_output.request_id, None) #headers_map.get(core_output.request_id)
        return engine_core_outputs
    sch.Scheduler.update_from_output = _patch_update_from_output
    print("<<< Monkey patch patch_scheduler is applied")


def patch_create_chat_completion_api():
    from fastapi.responses import JSONResponse
    from fastapi import Request
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    from functools import wraps
    from vllm.entrypoints.openai import api_server
    import vllm.entrypoints.openai.api_server
    orig_create_chat_completion = vllm.entrypoints.openai.api_server.create_chat_completion
    def patch_response_headers(fn):
        @wraps(fn)
        async def wrapper(request: ChatCompletionRequest, raw_request: Request):
            response = await fn(request, raw_request)
            if isinstance(response, JSONResponse):
                try:
                    content = json.loads(response.body)
                except Exception as e:
                    print("Failed to load body:", e)
                    content = None
                if isinstance(content, dict):
                    kv_params = content.get("kv_transfer_params") or {}
                    trace_headers = kv_params.get("trace_headers", {})
                    if trace_headers:
                        response.headers["X-Request-Id"] = content.get("id")
                        response.headers["Traceparent"] = trace_headers.get("traceparent")
                        response.headers["Start_time_ns"] = trace_headers.get("start_time_ns")
            return response
        return wrapper

    new_create_chat_completion = patch_response_headers(orig_create_chat_completion)
    for route in api_server.router.routes:
        if getattr(route, "path", None) == "/v1/chat/completions":
            route.endpoint = new_create_chat_completion
    vllm.entrypoints.openai.api_server.create_chat_completion = new_create_chat_completion
    print("<<< Monkey patch patch_create_chat_completion_api is applied")


# following monkey patch is for passing tracer span_context inside process_input_socket function
def patched_process_inputs():
    from vllm.v1.engine.processor import Processor
    from typing import Any, Optional, Union, Mapping
    from vllm.inputs import PromptType
    from vllm.lora.request import LoRARequest
    from vllm.pooling_params import PoolingParams
    from vllm.prompt_adapter.request import PromptAdapterRequest
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    patch_enginecorerequest()

    _origin_process_inputs = Processor.process_inputs
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
        raw_trace_headers = trace_headers
        trace_headers = None
        prompt, outputs = _origin_process_inputs(
            self,
            request_id,
            prompt,
            params,
            arrival_time,
            lora_request,
            tokenization_kwargs,
            trace_headers,
            prompt_adapter_request,
            priority,
        )
        outputs.trace_headers = raw_trace_headers
        return prompt, outputs
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

def patch_ModelRunnerOutput():
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
    from functools import wraps
    import os
    original_init = ModelRunnerOutput.__init__
    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        finished_sending_headers = kwargs.pop('finished_sending_headers', None)
        finished_recving_headers = kwargs.pop('finished_recving_headers', None)
        original_init(self, *args, **kwargs)
        self.finished_sending_headers = finished_sending_headers if isinstance(finished_sending_headers, dict) else {}
        self.finished_recving_headers = finished_recving_headers if isinstance(finished_recving_headers, dict) else {}
    ModelRunnerOutput.__init__ = patched_init
    if hasattr(EMPTY_MODEL_RUNNER_OUTPUT, 'req_ids'):
        EMPTY_MODEL_RUNNER_OUTPUT.finished_sending_headers = {}
        EMPTY_MODEL_RUNNER_OUTPUT.finished_recving_headers = {}

    import vllm.v1.core.sched.scheduler as sch
    _orig_init = sch.Scheduler.__init__
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.finished_recving_kv_req_headers = {}
    sch.Scheduler.__init__ = _patched_init

    orig_update_from_kv_xfer_finished = sch.Scheduler._update_from_kv_xfer_finished
    def new__update_from_kv_xfer_finished(self, model_runner_output: ModelRunnerOutput):
        if model_runner_output.finished_recving_headers:
            for req_id in list(model_runner_output.finished_recving_headers.keys()):
                headers = model_runner_output.finished_recving_headers.pop(req_id)
                self.finished_recving_kv_req_headers[req_id] = headers
        orig_update_from_kv_xfer_finished(self, model_runner_output)
    sch.Scheduler._update_from_kv_xfer_finished = new__update_from_kv_xfer_finished
    print("<<< Monkey patch patch_ModelRunnerOutput is applied")

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
        patch_ModelRunnerOutput()
        apply_patches(profiling_namelist)
        monkey_patch_async_generator_io_logger()
    else:
        logger.error(f"'{profiling_namelist}' does not exist.")
        raise FileNotFoundError(f"'{profiling_namelist}' does not exist.")