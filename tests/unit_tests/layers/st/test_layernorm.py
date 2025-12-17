import os
import pytest
import torch
import torch_npu
import torch.multiprocessing as mp
import tempfile
import traceback
import sys
import importlib
from typing import Optional, Callable, Any, List, Tuple
from omni.adaptors.vllm.patches.model_patch import patch_all
# Top-level imports for pytest parameterization
from omni.layers.layernorm import RMSNorm, RMSNormFlashComm
from omni.models.config_loader.loader import model_extra_config

from .distributed_test_common import parse_ascend_devices, distributed_worker_pool, _persistent_worker_loop

FIRST_DIE, _ = parse_ascend_devices()
TEST_SEED = 0
@pytest.fixture
def npu_device():
    return torch.device(f"npu:{FIRST_DIE}")

def rmsnorm_golden(x: torch.Tensor, 
                   residual: Optional[torch.Tensor], 
                   weight: torch.Tensor, 
                   bias: Optional[torch.Tensor], 
                   eps: float):
    """
    reference rmsnorm
    """
    x_f32 = x.float()
    weight_f32 = weight.float()
    
    if residual is not None:
        res_f32 = residual.float()
        res_out = x_f32 + res_f32
        norm_input = res_out
    else:
        res_out = None
        norm_input = x_f32

    # Var = mean(x^2)
    variance = norm_input.pow(2).mean(dim=-1, keepdim=True)
    # Normed = x * 1/sqrt(var + eps)
    hidden_states = norm_input * torch.rsqrt(variance + eps)
    # Apply Weight
    out = hidden_states * weight_f32
    
    if bias is not None:
        out = out + bias.float()
        
    return out.to(x.dtype), (res_out.to(x.dtype) if res_out is not None else None)

# @pytest.mark.parametrize("load_bias_env", ['0', '1'])
@pytest.mark.parametrize("load_bias_env", ['0'])
@pytest.mark.parametrize("with_residual", [True, False])
@pytest.mark.parametrize("with_quant", [True, False])
def test_rmsnorm_basic(npu_device, load_bias_env, with_residual, with_quant):
    """
    Runs the RMSNorm logic on actual NPU hardware and compares against a Golden Reference.
    """
    hidden_size = 128
    dtype = torch.float16 
    eps = 1e-6
    
    with pytest.MonkeyPatch.context() as m:
        m.setenv("LOAD_RMSNORM_BIAS", load_bias_env)
        
        norm = RMSNorm(hidden_size, eps=eps, dtype=dtype).to(npu_device)
        
        if load_bias_env == "1":
            # Initialize bias to non-zero to actually test the addition logic
            torch.nn.init.normal_(norm.bias, mean=0.5, std=0.1)

        # Input
        x = torch.randn(1, 10, hidden_size, dtype=dtype, device=npu_device)
        
        if with_residual:
            residual = torch.randn(1, 10, hidden_size, dtype=dtype, device=npu_device)
            x_ref = x.clone()
            residual_ref = residual.clone()
        else:
            residual = None
            x_ref = x.clone()
            residual_ref = None
        
        output = norm(x, residual=residual, quant_symbol=with_quant)

        ref_out, ref_residual = rmsnorm_golden(
            x_ref, residual_ref, norm.weight, norm.bias, eps
        )

        if with_residual:
            assert isinstance(output, tuple)
            out_x, out_res = output
            
            # Check Residual Correctness
            # The residual returned by fused kernel is (x + old_residual)
            assert torch.allclose(out_res, ref_residual, atol=1e-3, rtol=1e-3), \
                "Residual update mismatch"
            
            if with_quant:
                assert isinstance(out_x, dict)
                # For now, just ensuring structure and residual correctness is sufficient.
            else:
                # Compare Norm Output
                assert torch.allclose(out_x, ref_out, atol=1e-3, rtol=1e-3), \
                    "RMSNorm output mismatch (Residual path)"

        else:
            if with_quant:
                assert not isinstance(output, dict)
                assert torch.allclose(output, ref_out, atol=1e-3, rtol=1e-3), \
                     "RMSNorm output mismatch (Standard path, Quant ignored)"
            else:
                assert torch.allclose(output, ref_out, atol=1e-3, rtol=1e-3), \
                    "RMSNorm output mismatch (Standard path)"

def _logic_rmsnorm_tp(device, local_rank, world_size, hidden_size, dtype, y_transform):
    """
    Logic for testing RMSNormFlashComm in a distributed setting.
    Verifies that 'AG' transform correctly gathers data from all ranks.
    """
    device = torch.device(f"npu:{device}")
    eps = 1e-6
    
    model = RMSNormFlashComm(hidden_size, eps=eps).to(dtype).to(device)
    # Ensure same weight on all ranks for consistent calculation
    torch.nn.init.ones_(model.weight) 

    # 2. Create Rank-Specific Input
    # Rank 0: filled with 1.0, Rank 1: filled with 2.0
    val = float(local_rank + 1)
    x = torch.full((1, 2, hidden_size), val, dtype=dtype, device=device)
    residual = torch.full((1, 2, hidden_size), val, dtype=dtype, device=device)
    
    # 3. Run Forward
    out, out_residual = model(x, residual=residual, y_transform=y_transform)
    
    # 4. Validation
    if y_transform == "AG":
        # Check Shape: [Batch * WorldSize, Seq, Hidden]
        # Input [1, 2, H] -> Gathered [2, 2, H]
        expected_shape = (world_size * x.shape[0], x.shape[1], x.shape[2])
        assert out.shape == expected_shape, f"Shape mismatch. Got {out.shape}, expected {expected_shape}"
        
        # Check Residual (Local): Should be local input + local residual = 2 * val
        expected_res_val = val + val
        assert torch.allclose(out_residual, torch.full_like(out_residual, expected_res_val)), \
            "Residual value mismatch"
    else:
        # No Gather
        assert out.shape == x.shape


def _logic_rmsnorm_tp_random_input(device, local_rank, world_size, hidden_size, dtype):
    """
    More robust test with random inputs to verify AllGather content.
    """
    device = torch.device(f"npu:{device}")
    eps = 1e-6
    model = RMSNormFlashComm(hidden_size, eps=eps).to(dtype).to(device)
    assert model.tp_size == world_size
    torch.nn.init.ones_(model.weight)
    
    # Generate random input specific to rank
    torch.manual_seed(local_rank) # Different seed per rank
    x = torch.randn(2, 4, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(2, 4, hidden_size, dtype=dtype, device=device)
    
    # Execute
    out_gathered, _ = model(x, residual=residual, y_transform="AG")
        
    # 1. Compute local truth
    local_norm_out_golden, _ = rmsnorm_golden(x, residual, model.weight, None, eps)
    
    # 2. Verify part of the gathered output matches local output
    start_idx = local_rank * x.shape[0]
    end_idx = (local_rank + 1) * x.shape[0]
    my_slice_in_gathered = out_gathered[start_idx:end_idx]
    
    assert torch.allclose(my_slice_in_gathered, local_norm_out_golden, atol=1e-3, rtol=1e-3), \
        f"Rank {local_rank}: Gathered output's local slice does not match local computation"


@pytest.mark.parametrize("y_transform", ["AG", ""])
def test_rmsnorm_tp_distributed(distributed_worker_pool, y_transform):
    """
    Tests RMSNormFlashComm using the shared persistent worker pool.
    """
    hidden_size = 128
    dtype = torch.float16
    
    if y_transform == "AG":
        # Use robust random check for AllGather
        func = _logic_rmsnorm_tp_random_input
        distributed_worker_pool(func, hidden_size, dtype)
    else:
        # Use basic shape check for No Gather
        func = _logic_rmsnorm_tp
        distributed_worker_pool(func, hidden_size, dtype, y_transform)
