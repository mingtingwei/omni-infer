import pytest
import os
import requests
from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock import start_vllm_mock, cleanup_subprocess
from proxy_test_methods import chat_completions, chat_completions_stream, chat_completions_invalid_server, \
    chat_completions_with_proxy
import port_manager

# Configuration
PREFILL_NUM = 3
DECODE_NUM = 3
proxy_port = 7000
prefill_port_list = None
decode_port_list = None

@pytest.fixture(scope="module")
def setup_teardown(vllm_keep_alive):
    global proxy_port
    global prefill_port_list
    global decode_port_list

    if os.getenv("PROXY_VLLM_POOL") == "1":
        ports = port_manager.get_ports_from_file()
        proxy_port = ports["proxy_port"]
        prefill_port_list = ports["prefill"][:PREFILL_NUM]
        decode_port_list = ports["decode"][:DECODE_NUM]
        ret = setup_proxy(proxy_port, prefill_port_list, decode_port_list, pd_policy="aggregation")
        if ret == -1:
            pytest.fail(f"Start proxy fail")
        print(f"\n[DEBUG] Skipping setup/teardown, {proxy_port=}, {prefill_port_list=}, {decode_port_list=}")
        yield
        teardown_proxy()
        return

    ports = port_manager.load_ports(PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    ret = setup_proxy(proxy_port, prefill_port_list, decode_port_list, pd_policy="aggregation")
    if ret == -1:
        pytest.fail(f"Start proxy fail")

    processes = start_vllm_mock(PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail(f"Start vllm fail")

    yield

    # --- Teardown: Shut down all instances ---
    teardown_proxy()
    print(f"\n[TEARDOWN] Shutting down {PREFILL_NUM + DECODE_NUM} instances...")
    cleanup_subprocess(processes)


def test_chat_completions(setup_teardown):
    chat_completions(prefill_port_list)

def test_chat_completions_stream(setup_teardown):
    chat_completions_stream(prefill_port_list)

def test_chat_completions_invalid_server(setup_teardown):
    chat_completions_invalid_server()

def test_chat_completions_with_proxy(setup_teardown):
    chat_completions_with_proxy(proxy_port)