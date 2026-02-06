import gc
import os

import pytest
import torch


def parse_ascend_devices():
    return 0, [0, 1]


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
