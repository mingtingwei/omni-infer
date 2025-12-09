from contextvars import ContextVar
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.status import Status, StatusCode
from typing import List, Mapping, Optional
import os, time, socket
parent_ctx_var: ContextVar = ContextVar("parent_ctx", default=None)
span_start_time: ContextVar = ContextVar('span_start_time', default=None)
ttft_start_time: ContextVar = ContextVar('ttft_start_time', default=None)
ttft_trace_id: ContextVar = ContextVar('ttft_trace_id', default=None)
ttft_end_time: ContextVar = ContextVar('ttft_end_time', default=None)

from vllm.logger import init_logger
import logging
logging.basicConfig(level=logging.DEBUG)
logger = init_logger(__name__)


def init_tracer(service_name: str = "vllm-service"):
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    logger.info(f"<<<<<<<< service.name = {trace.get_tracer_provider().resource.attributes.get(SERVICE_NAME)}")

def clean_ctx():
    parent_ctx_var.set(None)
    span_start_time.set(None)

def create_span(
    start_time_ns: int,
    end_time: int,
    action_name: str, 
    request_id: str,
    ip_str: str,
    ctx_flag: bool,
    trace_headers: Optional[Mapping[str, str]] = None,
):
    tracer = trace.get_tracer("omni_tracing")
    if ctx_flag:
        ctx = parent_ctx_var.get()
        start_time = span_start_time.get()
    else:
        ctx = TraceContextTextMapPropagator().extract(trace_headers)
        start_time = start_time_ns

    new_span = tracer.start_span(action_name, start_time=start_time, context=ctx, kind=trace.SpanKind.SERVER)
    real_parent = trace.get_current_span(ctx)
    new_span.set_attribute("father_span id",format(real_parent.get_span_context().span_id, '016x'))

    current_ctx = trace.set_span_in_context(new_span)
    parent_ctx_var.set(current_ctx)
    span_start_time.set(end_time)

    new_span.set_attribute("span id", format(new_span.get_span_context().span_id, '016x'))
    new_span.set_attribute("request_id", request_id)
    new_span.set_attribute("role", os.getenv("ROLE"))
    new_span.set_attribute("ip", ip_str)
    new_span.set_attribute("start time", start_time)
    new_span.set_attribute("end time", end_time)  
    new_span.set_status(Status(StatusCode.OK))

    new_span.end(end_time=end_time)

    return current_ctx