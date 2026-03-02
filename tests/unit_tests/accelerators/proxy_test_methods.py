import pytest
import requests
import json
import port_manager

def chat_completions(prefill_port_list):
    url = f"http://127.0.0.1:{prefill_port_list[0]}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "qwen",
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

def chat_completions_stream(prefill_port_list):
    url = f"http://127.0.0.1:{prefill_port_list[0]}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "qwen",
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

def chat_completions_invalid_server():
    invaild_port = port_manager.find_free_port_excluding_existing()
    url = f"http://127.0.0.1:{invaild_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True
    }

    with pytest.raises(requests.exceptions.ConnectionError):
        response = requests.post(url, headers=headers, json=data, timeout=3)

def chat_completions_with_proxy(proxy_port):
    url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": "12345"
    }
    data = {
        "model": "qwen",
        "temperature": 0,
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "stream": True
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)
    print("Status Code:", response.status_code)
    print("Response Headers:", response.headers)
    print("Response Body (preview):", response.text[:200] + "..." if len(response.text) > 200 else response.text)
    assert response.status_code == 200
