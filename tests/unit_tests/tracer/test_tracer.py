from omni.tools.profiler.trace.tracing import (
    init_tracer,
    create_span,
    clean_ctx,
    parent_ctx_var,
    span_start_time,
    ttft_start_time,
    ttft_trace_id,
    ttft_end_time,
)

from unittest.mock import Mock, patch
import time


def test_tracer_task():
    pass
    init_tracer()
    ip = "127.0.0.1"
    dot_group = ip.split(".")
    assert len(dot_group) == 4
    time_stamp_1 = time.time()
    req1 = create_span(time_stamp_1, time_stamp_1, "test_01", "0", ip, False, {"name": "test", "type": "demo"})

    from opentelemetry.context.context import Context
    assert isinstance(req1, Context)
    time.sleep(2)
    time_stamp_2 = time.time()
    req2 = create_span(time_stamp_2, time_stamp_2, "test_02", "1", ip, True, {"name": "test", "type": "demo"})
    assert isinstance(req2, Context)
    clean_ctx()

