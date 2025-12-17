# tests/test_fused_mlp.py
import pytest
import torch
import torch_npu
import copy
from typing import Optional, Tuple

from omni.layers.fused_mlp.layer import FusedMLP, W8A8DynamicFusedMLPMethod
from omni.models.config_loader.loader import model_extra_config
from ..distributed_test_common import distributed_worker_pool


def mlp_golden(x: torch.Tensor, 
               gate_up_weight: torch.Tensor, 
               down_weight: torch.Tensor, 
               gate_up_bias: Optional[torch.Tensor] = None, 
               down_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Standard FP16 Golden Reference: Down( SiLU(Gate) * Up )
    Expects standard nn.Linear weight shapes: [Out, In]
    """
    # 1. GateUp
    gate_up_out = torch.nn.functional.linear(x, gate_up_weight, gate_up_bias)
    
    # 2. Split & SwiGLU
    gate, up = torch.chunk(gate_up_out, 2, dim=-1)
    act_out = torch.nn.functional.silu(gate) * up
    
    # 3. Down
    output = torch.nn.functional.linear(act_out, down_weight, down_bias)
    return output

def quantize_weight_per_tensor(tensor: torch.Tensor, output_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes tensor and returns scale shaped for NPU kernels.
    Args:
        tensor: Weight tensor [In, Out]
        output_dim: The output dimension size (for scale broadcasting)
    Returns:
        quant: Int8 Tensor [In, Out]
        scale: Float32 Tensor [Output_Dim]
    """
    max_val = tensor.abs().max()
    scale = max_val / 127.0
    scale = torch.max(scale, torch.tensor(1e-6, device=tensor.device))
    scale = scale.to(torch.float32)
    
    quant = torch.clamp(torch.round(tensor.float() / scale), -127, 127).to(torch.int8)
    
    # NPU kernels often require scale to match the output channel count (per-channel)
    scale_expanded = scale.repeat(output_dim)
    
    return quant, scale_expanded

def _logic_fused_mlp_unquantized(device: int, rank: int, world_size: int, 
                                 hidden_size: int, intermediate_size: int, 
                                 x_transform: Optional[str]):
    device = torch.device(f"npu:{device}")
    dtype = torch.float16
    
    # 1. Initialize Model (FP16)
    model = FusedMLP(hidden_size, intermediate_size, hidden_act="silu", quant_config=None).to(device).to(dtype)
    
    # 2. Generate Global Weights (Deterministic Pattern)
    # Use arange to create unique, predictable values. 
    # Modulo 7 to keep values small and avoid FP16 overflow during accumulation.
    torch.manual_seed(42) # Still needed for randomness if any, but we overwrite below
    
    # Create pattern: 0.1, 0.2, 0.3... etc
    # Shape: [2 * Inter, Hidden]
    num_elements = 2 * intermediate_size * hidden_size
    global_gate_up = (torch.arange(num_elements, device=device) % 7).to(dtype).reshape(2 * intermediate_size, hidden_size) * 0.1
    
    # Shape: [Hidden, Inter]
    num_elements_down = hidden_size * intermediate_size
    global_down = (torch.arange(num_elements_down, device=device) % 5).to(dtype).reshape(hidden_size, intermediate_size) * 0.1
    
    # 3. Slice for Local Rank
    # GateUp (Column Parallel): Split Output Dim
    gate_chunk = intermediate_size // world_size
    g_start, g_end = rank * gate_chunk, (rank + 1) * gate_chunk
    
    gl_gate, gl_up = torch.chunk(global_gate_up, 2, dim=0)
    loc_gate = gl_gate[g_start:g_end, :]
    loc_up = gl_up[g_start:g_end, :]
    local_gate_up = torch.cat([loc_gate, loc_up], dim=0) # Shape: [Local_Out, Hidden]
    
    # Down (Row Parallel): Split Input Dim
    d_start, d_end = rank * gate_chunk, (rank + 1) * gate_chunk
    local_down = global_down[:, d_start:d_end] # Shape: [Hidden, Local_Inter]
    
    # 4. Assign Weights
    # Omni layers use matmul(x, W), so W must be [In, Out].
    # We must REPLACE the parameter because the shape changes from the default initialization.
    with torch.no_grad():
        # Transpose [Out, In] -> [In, Out]
        model.gate_up_proj.weight = torch.nn.Parameter(local_gate_up.t().contiguous())
        model.down_proj.weight = torch.nn.Parameter(local_down.t().contiguous())
        
        if model.gate_up_proj.bias is not None: model.gate_up_proj.bias.zero_()
        if model.down_proj.bias is not None: model.down_proj.bias.zero_()

    # 5. Input
    #  - For fixed debugging, we want simple inputs like 1.0, 2.0
    # Inputs: [Batch=2, Seq=4, Hidden]
    x_val = torch.arange(2 * 4 * hidden_size, device=device).reshape(2, 4, hidden_size).to(dtype)
    # Normalize to small values
    x_val = (x_val % 3) * 0.5 
    
    if x_transform == "AG":
        # For AllGather, different ranks get different slices.
        # Rank 0 gets first half, Rank 1 gets second half (roughly simulating data parallel split)
        # But to keep math verification easy, let's just use specific fixed tensors per rank.
        if rank == 0:
            x = torch.full_like(x_val, 1.0)
        else:
            x = torch.full_like(x_val, 2.0)
    else:
        # Standard TP: Identical Inputs on all ranks
        x = x_val.clone()
        
    # 6. Run & Verify
    output = model(x, x_transform=x_transform)
    
    # Golden Verification
    if x_transform == "AG":
        x_list = [torch.zeros_like(x) for _ in range(world_size)]
        torch.distributed.all_gather(x_list, x)
        x_global = torch.cat(x_list, dim=0) 
    else:
        x_global = x

    # Golden uses F.linear, so it expects [Out, In] (Non-transposed global weights)
    ref_output = mlp_golden(x_global, global_gate_up, global_down)
    
    # Stricter tolerance possible now with deterministic small values
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-3), \
        f"Rank {rank}: Unquantized Output mismatch"


def _logic_fused_mlp_w8a8(device: int, rank: int, world_size: int, 
                          hidden_size: int, intermediate_size: int, 
                          x_transform: Optional[str]):
    device = torch.device(f"npu:{device}")
    dtype = torch.float16
    
    # 1. Initialize Model
    model = FusedMLP(hidden_size, intermediate_size, hidden_act="silu", quant_config=None).to(device).to(dtype)
    
    # 2. Prepare Weights (Deterministic Pattern for W8A8)
    # Use very small integers so quantization error is minimized or easier to reason about.
    num_elements = 2 * intermediate_size * hidden_size
    # Pattern: -1, 0, 1 repeating
    global_gate_up = ((torch.arange(num_elements, device=device) % 3) - 1).to(dtype).reshape(2 * intermediate_size, hidden_size)
    
    num_elements_down = hidden_size * intermediate_size
    # Pattern: 1, 1, 1 (Identity-ish)
    global_down = torch.ones(hidden_size, intermediate_size, dtype=dtype, device=device)
    
    # Slice
    gate_chunk = intermediate_size // world_size
    g_start, g_end = rank * gate_chunk, (rank + 1) * gate_chunk
    
    gl_gate, gl_up = torch.chunk(global_gate_up, 2, dim=0)
    loc_gate = gl_gate[g_start:g_end, :]
    loc_up = gl_up[g_start:g_end, :]
    local_gate_up_fp16 = torch.cat([loc_gate, loc_up], dim=0) # [Local_Out, Hidden]
    
    d_start, d_end = rank * gate_chunk, (rank + 1) * gate_chunk
    local_down_fp16 = global_down[:, d_start:d_end] # [Hidden, Local_Inter]

    # 3. Quantize Weights
    # Transpose for W8A8 matmul [In, Out]
    local_gate_up_t = local_gate_up_fp16.t().contiguous() # [Hidden, Local_Out]
    local_down_t = local_down_fp16.t().contiguous()       # [Local_Inter, Hidden]
    
    q_gate_up, s_gate_up = quantize_weight_per_tensor(local_gate_up_t, output_dim=local_gate_up_t.shape[1])
    q_down, s_down = quantize_weight_per_tensor(local_down_t, output_dim=local_down_t.shape[1])
    
    # 4. Inject into Model
    with torch.no_grad():
        model.gate_up_proj.weight = torch.nn.Parameter(q_gate_up, requires_grad=False)
        if not hasattr(model.gate_up_proj, 'weight_scale'):
            model.gate_up_proj.register_parameter('weight_scale', torch.nn.Parameter(s_gate_up))
        else:
            model.gate_up_proj.weight_scale = torch.nn.Parameter(s_gate_up)

        model.down_proj.orig_dtype = dtype
        model.down_proj.weight = torch.nn.Parameter(q_down, requires_grad=False)
        if not hasattr(model.down_proj, 'weight_scale'):
            model.down_proj.register_parameter('weight_scale', torch.nn.Parameter(s_down))
        else:
             model.down_proj.weight_scale = torch.nn.Parameter(s_down)

    class MockConfig: pass
    model.quant_method = W8A8DynamicFusedMLPMethod(MockConfig())
    
    # 6. Input
    # Use simple ones for W8A8 to minimize quantization noise impact
    x = torch.ones(2, 4, hidden_size, dtype=dtype, device=device)
    
    # Flatten input to [Batch*Seq, Hidden] to avoid NPU kernel dimension ambiguity
    x_flattened = x.view(-1, hidden_size)
    
    # 7. Run W8A8
    output_w8a8 = model(x_flattened, x_transform=x_transform)
    
    # 8. Run Golden (FP16)
    if x_transform == "AG":
        x_list = [torch.zeros_like(x_flattened) for _ in range(world_size)]
        torch.distributed.all_gather(x_list, x_flattened)
        x_global = torch.cat(x_list, dim=0) 
    else:
        x_global = x_flattened

    ref_output = mlp_golden(x_global, global_gate_up, global_down)
    
    # 9. Verification
    assert output_w8a8.shape == ref_output.shape
    
    mae = (output_w8a8 - ref_output).abs().mean()
    ref_mean = ref_output.abs().mean()
    
    assert mae < 0.2 * ref_mean, \
        f"Rank {rank}: Quantization error too high. MAE={mae}, RefMean={ref_mean}"

@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("intermediate_size", [256])
@pytest.mark.parametrize("x_transform", [None, "AG"])
def test_fused_mlp_unquantized_distributed(distributed_worker_pool, hidden_size, intermediate_size, x_transform):
    distributed_worker_pool(_logic_fused_mlp_unquantized, hidden_size, intermediate_size, x_transform)

@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("intermediate_size", [256])
@pytest.mark.parametrize("x_transform", [None]) 
def test_fused_mlp_w8a8_distributed(distributed_worker_pool, hidden_size, intermediate_size, x_transform):
    distributed_worker_pool(_logic_fused_mlp_w8a8, hidden_size, intermediate_size, x_transform)