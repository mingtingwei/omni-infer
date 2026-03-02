"""
Proxy reload stability tests.

This file contains:
- Serial reload correctness test (deterministic)
- Concurrent traffic + multi-round reload stress test

Redundancy in test logic is intentional for observability and debuggability.
"""

import pytest
import os
import time
import requests

from run_proxy import setup_proxy, teardown_proxy
from run_vllm_mock import start_vllm_mock, cleanup_subprocess
import port_manager
from proxy_reload_test_methods import proxy_reload, proxy_reload_under_concurrent_traffic, wait_proxy_health

PREFILL_NUM = 3
DECODE_NUM = 3

@pytest.fixture(scope="module")
def reload_env(vllm_keep_alive):
    os.environ["no_proxy"] = "localhost,127.0.0.1"

    if os.getenv("PROXY_VLLM_POOL") == "1":
        ports = port_manager.get_ports_from_file()
        proxy_port = ports["proxy_port"]
        prefill_ports = ports["prefill"][:PREFILL_NUM]
        decode_ports = ports["decode"][:DECODE_NUM]
        ret = setup_proxy(proxy_port, prefill_ports, decode_ports, pd_policy="aggregation")
        if ret == -1:
            pytest.fail(f"Start proxy fail")
        print(f"\n[DEBUG] Skipping setup/teardown, {proxy_port=}, {prefill_ports=}, {decode_ports=}")
        yield {
        "proxy_port": proxy_port,
        "prefill_ports": prefill_ports,
        "decode_ports": decode_ports,
        }
        teardown_proxy()
        return

    ports = port_manager.load_ports(PREFILL_NUM, DECODE_NUM)

    proxy_port = ports["proxy_port"]
    prefill_ports = ports["prefill"]
    decode_ports = ports["decode"]

    ret = setup_proxy(proxy_port, prefill_ports, decode_ports, pd_policy="aggregation")
    if ret == -1:
        pytest.fail("Start proxy fail")

    wait_proxy_health(proxy_port)

    processes = start_vllm_mock(PREFILL_NUM, DECODE_NUM)
    if not processes:
        pytest.fail("Start vllm fail")
    time.sleep(1)

    yield {
        "proxy_port": proxy_port,
        "prefill_ports": prefill_ports,
        "decode_ports": decode_ports,
        "processes": processes,
    }

    cleanup_subprocess(processes)

    try:
        teardown_proxy()
    except Exception as e:
        print(f"[TEARDOWN] teardown_proxy ignored: {e}")



def test_proxy_reload(reload_env):
    proxy_reload(reload_env)

def test_proxy_reload_under_concurrent_traffic(reload_env):
    """
    Concurrent traffic + multi-round real reload stability test.

    Test objective:
    With continuous real request traffic in flight, repeatedly perform nginx reloads
    in the background. During each reload, the upstream configuration (Prefill / Decode
    backends) is actively changed (remove / add / rollback) to validate the proxy's
    stability under high churn conditions.

    Key guarantees covered by this test:
    1. Requests must continue to succeed during reload (no dropped requests, no 502s).
    2. Reload operations run in a background thread, concurrently with request traffic.
    3. Reload is executed repeatedly across multiple rounds.
    4. Each reload involves real upstream changes (Prefill / Decode add/remove/rollback).
    5. After each round, the configuration must be fully restored to the initial baseline
    (no state pollution).

    Policy constraints:
    - Client timeouts are relaxed to avoid false positives.
    - Any request timeout is treated as a failure.
    - All requests must return HTTP 200.
    """
    proxy_reload_under_concurrent_traffic(reload_env)