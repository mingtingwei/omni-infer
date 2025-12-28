import os
import json
import pytest
import uuid
from collections import defaultdict
from omni.tools.profiler.trace.log_to_jaeger import (
    normalize_reqid, parse_log, build_spans, build_jaeger_trace, main,
    ACTION_LIST, EXTRA_SPANS
)

TEST_LOG_DIR = os.path.join(os.path.dirname(__file__), "test_logs")
TEST_OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "test_output.json")


@pytest.fixture(scope="module")
def cleanup():
    if os.path.exists(TEST_OUTPUT_JSON):
        os.remove(TEST_OUTPUT_JSON)
    yield
    if os.path.exists(TEST_OUTPUT_JSON):
        os.remove(TEST_OUTPUT_JSON)


@pytest.fixture(scope="module")
def parsed_log_data():
    req_action_times, req_roles = parse_log(TEST_LOG_DIR)
    return req_action_times, req_roles


def test_normalize_reqid():
    # prefix reqid
    assert normalize_reqid("chatcmpl-testreq1") == "testreq1"
    # no-prefix reqid
    assert normalize_reqid("testreq2") == "testreq2"
    # null string
    assert normalize_reqid("") == ""


def test_parse_log(parsed_log_data):
    req_action_times, req_roles = parsed_log_data
    # test 2 req
    assert len(req_action_times) == 2
    assert "testreq1" in req_action_times
    assert "testreq2" in req_action_times

    # make sure testreq1 has all ACTION
    for action in ACTION_LIST:
        assert action in req_action_times["testreq1"]

    # role and ip correct
    assert req_roles["testreq1"]["Start to schedule"] == ("proxy", "192.168.1.1")
    assert req_roles["testreq1"]["Enter state P running"] == ("pnode", "192.168.1.4")
    assert req_roles["testreq1"]["Enter state D running"] == ("dnode", "192.168.1.5")


def test_build_spans(parsed_log_data):
    # test build span
    req_action_times, req_roles = parsed_log_data
    req_id = "testreq1"
    duration_dict = defaultdict(list)
    TP_x = (90, 95, 99)

    spans, _ = build_spans(req_id, req_action_times[req_id], req_roles[req_id], False, TP_x, duration_dict, False)
    assert len(spans) > 0
    # check all elements
    for span in spans:
        assert "traceID" in span
        assert "spanID" in span
        assert "operationName" in span
        assert "startTime" in span
        assert "duration" in span
        assert "tags" in span

    # Span avg and Tpx
    avg_spans, tp_x_dict = build_spans(req_id, req_action_times[req_id], req_roles[req_id], True, TP_x, duration_dict,
                                       True)
    assert len(avg_spans) > 0
    assert set(tp_x_dict.keys()) == {90, 95, 99}
    assert len(tp_x_dict[90]) > 0


def test_main_full_flow(cleanup):
    # main work flow
    main(TEST_LOG_DIR, TEST_OUTPUT_JSON, None, 0)  # num=0

    assert os.path.exists(TEST_OUTPUT_JSON)
    with open(TEST_OUTPUT_JSON, "r") as f:
        jaeger_data = json.load(f)

    assert "data" in jaeger_data
    traces = jaeger_data["data"]
    assert len(traces) >= 1

    # test_req1 exist
    test_req1_found = False
    for trace in traces:
        for span in trace["spans"]:
            for tag in span["tags"]:
                if tag["key"] == "RequestID" and tag["value"] == "testreq1":
                    test_req1_found = True
    assert test_req1_found


def test_build_jaeger_trace(parsed_log_data):
    req_action_times, req_roles = parsed_log_data
    reqid = "testreq1"
    duration_dict = defaultdict(list)
    TP_x = (90, 95, 99)

    spans, _ = build_spans(reqid, req_action_times[reqid], req_roles[reqid], False, TP_x, duration_dict, False)
    jaeger_trace = build_jaeger_trace(reqid, spans)

    # Jaeger struct
    assert "traceID" in jaeger_trace
    assert "spans" in jaeger_trace
    assert "processes" in jaeger_trace
    # traceID in all span
    trace_id = jaeger_trace["traceID"]
    for span in jaeger_trace["spans"]:
        assert span["traceID"] == trace_id
        for ref in span["references"]:
            assert ref["traceID"] == trace_id


def test_parse_log_empty_dir():
    # empty log file
    empty_log_dir = os.path.join(os.path.dirname(__file__), "empty_logs")
    os.makedirs(empty_log_dir, exist_ok=True)
    try:
        req_action_times, req_roles = parse_log(empty_log_dir)
        assert len(req_action_times) == 0
        assert len(req_roles) == 0
    finally:
        os.rmdir(empty_log_dir)


def test_main_no_matching_reqids(cleanup):
    # select-req_ids no matching
    if os.path.exists(TEST_OUTPUT_JSON):
        os.remove(TEST_OUTPUT_JSON)

    select_req_ids_set = {"nonexistent_req_id"}
    main(TEST_LOG_DIR, TEST_OUTPUT_JSON, select_req_ids_set, 1)

    if os.path.exists(TEST_OUTPUT_JSON):
        with open(TEST_OUTPUT_JSON, "r") as f:
            jaeger_data = json.load(f)

        traces = jaeger_data["data"]
        nonexistent_found = False
        for trace in traces:
            for span in trace["spans"]:
                for tag in span["tags"]:
                    if tag["key"] == "RequestID" and tag["value"] == "nonexistent_req_id":
                        nonexistent_found = True
        assert not nonexistent_found
    else:
        assert True


if __name__ == "__main__":
    pytest.main(["-v", __file__])