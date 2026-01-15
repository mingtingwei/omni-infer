import gc
import os

import pytest
import torch


def parse_ascend_devices():
    env_val = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")

    if not env_val.strip():
        return 0, [0, 1]
    try:
        visible_die_list = [int(x.strip()) for x in env_val.split(",") if x.strip()]
        device_no_list = sorted(list(set(x // 2 for x in visible_die_list)))
        first_device_no = device_no_list[0]
    except ValueError as exc:
        print(f"Error parsing ASCEND_RT_VISIBLE_DEVICES: {exc}, using default values.")
        return 0, [0, 1]

    return first_device_no, device_no_list


def _require_torch_npu():
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:
        pytest.skip(f"torch_npu unavailable: {exc}")
    return torch_npu


@pytest.fixture
def npu_device():
    torch_npu = _require_torch_npu()
    first_die, _ = parse_ascend_devices()
    device = torch.device(f"npu:{first_die}")
    torch.npu.set_device(device)
    try:
        yield device
    finally:
        torch_npu.npu.synchronize()
        torch_npu.npu.empty_cache()
        gc.collect()
