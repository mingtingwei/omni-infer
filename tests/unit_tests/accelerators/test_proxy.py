import pytest
import os
import subprocess
import time
import requests
import json
from pathlib import Path
from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock import strart_vllm_mock, cleanup_subprocess
import port_manager

# Configuration
PREFILL_NUM = 3
DECODE_NUM = 3
proxy_port = 7000
prefill_port_list = None
decode_port_list = None

@pytest.fixture(scope="module")
def setup_teardown():
    global proxy_port
    global prefill_port_list
    global decode_port_list

    if os.getenv("SKIP_FIXTURE") == "1":
        ports = port_manager.get_ports_from_file()
        proxy_port = ports["proxy_port"]
        prefill_port_list = ports["prefill"]
        decode_port_list = ports["decode"]
        print(f"\n[DEBUG] Skipping setup/teardown, {proxy_port=}, {prefill_port_list=}, {decode_port_list=}")
        yield
        return

    ports = port_manager.load_ports(PREFILL_NUM, DECODE_NUM)
    proxy_port = ports["proxy_port"]
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]

    ret = setup_proxy(proxy_port, prefill_port_list, decode_port_list)
    if not ret == 0:
        pytest.fail(f"Start proxy fail")

    processes = strart_vllm_mock(PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail(f"Start vllm fail")

    yield

    # --- Teardown: Shut down all instances ---
    teardown_proxy()
    print(f"\n[TEARDOWN] Shutting down {PREFILL_NUM + DECODE_NUM} instances...")
    cleanup_subprocess(processes)


def test_chat_completions(setup_teardown):
    url = f"http://127.0.0.1:{prefill_port_list[0]}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True
    }

    response = requests.post(url, headers=headers, json=data, timeout=10)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200

def test_chat_completions_stream(setup_teardown):
    url = f"http://127.0.0.1:{prefill_port_list[0]}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True
    }
    # Enable streaming response handling
    with requests.post(url, headers=headers, json=data, stream=True, timeout=30) as resp:
        resp.raise_for_status()  # Raise exception for HTTP error status codes
        token_cnt = 0
        for line in resp.iter_lines():
            if line:
                # Process Server-Sent Events (SSE) lines
                if line.startswith(b"data:"):
                    json_str = line[len(b"data:"):].strip()
                    if json_str == b"[DONE]":
                        print(f"Stream finished. get {token_cnt} output\n")
                        break
                    try:
                        chunk = json.loads(json_str)
                        # Extract content (assuming OpenAI-compatible format)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            token_cnt += 1
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        print(f"\nFailed to decode JSON: {json_str}")

def test_chat_completions_invalid_server(setup_teardown):
    invaild_port = port_manager.find_free_port_excluding_existing()
    url = f"http://127.0.0.1:{invaild_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True
    }

    with pytest.raises(requests.exceptions.ConnectionError):
        response = requests.post(url, headers=headers, json=data, timeout=3)

def test_chat_completions_with_proxy(setup_teardown):
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "deepseek",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True
    }

    response = requests.post(url, headers=headers, json=data, timeout=10)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200
