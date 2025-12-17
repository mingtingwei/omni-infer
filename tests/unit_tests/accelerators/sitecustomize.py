import coverage

coverage.process_startup()

import os
import sys
from unittest.mock import MagicMock, patch

ENABLE_MOCK = int(os.getenv("NO_NPU_MOCK", "0"))

def apply_vllm_patch():
    try:
        from vllm.v1.worker import gpu_model_runner
    except ImportError:
        return

    original_init = gpu_model_runner.GPUModelRunner.__init__

    mock_props = MagicMock()
    mock_props.multi_processor_count = 2

    def patched_init(self, *args, **kwargs):
        import torch
        with patch.object(torch.cuda, 'get_device_properties', return_value=mock_props):
            original_init(self, *args, **kwargs)

    gpu_model_runner.GPUModelRunner.__init__ = patched_init

    print(f"apply dt patch")

try:
    if ENABLE_MOCK:
        apply_vllm_patch()
except Exception as e:
    print(f"Warning: Failed to apply vllm mock patch: {e}")

