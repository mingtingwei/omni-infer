import pytest
import torch
import torch.nn.functional as F
import torch.distributed as dist
from unittest.mock import MagicMock, patch

# Adjust import path based on your project structure
from ..distributed_test_common import distributed_worker_pool

# --- Golden References ---

def moe_golden_constant(
    hidden_states: torch.Tensor,
    top_k: int,
    w13_val: float,
    w2_val: float,
    hidden_size: int,
    inter_size: int
) -> torch.Tensor:
    """
    Simplified Golden for Constant Inputs/Weights.
    Calculates the expected scalar value and broadcasts it.
    """
    # 1. Input Value
    input_val = hidden_states[0,0,0].item()
    
    # 2. W13 Projection (Gate + Up)
    # val = HiddenSize * input_val * w13_val
    dot_w13 = hidden_size * input_val * w13_val
    
    # 3. SwiGLU
    # Out = SiLU(Gate) * Up
    hidden_act = F.silu(torch.tensor(dot_w13)) * dot_w13
    hidden_act_val = hidden_act.item()
    
    # 4. W2 Projection
    # val = InterSize * hidden_act_val * w2_val
    dot_w2 = inter_size * hidden_act_val * w2_val
    
    # 5. Weighted Sum
    final_val = dot_w2
    
    return torch.full_like(hidden_states, final_val)

def mock_dist_all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None):
    """
    Simulates All-to-All by treating it as a loopback (Input -> Output).
    This ensures shapes are valid for subsequent operations without needing real distributed comms.
    """
    if output.shape == input.shape:
        output.copy_(input)
    else:
        # If shapes differ (due to split sizes), we just fill with zeros to let execution proceed
        # This is sufficient for verifying the 'new_empty' SymInt crash is gone.
        output.zero_()

def mock_dist_all_reduce(tensor, op=None, group=None):
    """Simulates All-Reduce (Sum) by keeping value as is (Simulating 1 active rank effectively)."""
    return tensor

def _logic_fused_moe_ep_constant(device_id, rank, world_size, hidden_size, num_experts_global, top_k):
    """
    EP Test with Deterministic Constants.
    """
    from omni.layers.moe.fused_moe.layer import FusedMoE
    
    # Setup Device
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    dtype = torch.float16
    intermediate_size = hidden_size * 2
    
    # 1. Mock Distributed Environment Groups
    # We mock the groups, BUT we must also patch the dist functions that use them
    ep_mock = MagicMock()
    ep_mock.world_size = world_size
    ep_mock.rank_in_group = rank
    ep_mock.device_group = MagicMock() # The internal group object

    tp_mock = MagicMock()
    tp_mock.world_size = 1
    tp_mock.rank_in_group = 0

    world_mock = MagicMock()
    world_mock.world_size = world_size
    world_mock.rank_in_group = rank

    # 2. Patch Context
    # We patch dist.all_to_all_single because passing a MagicMock group to the real C++ dist op causes a crash.
    with patch("omni.layers.moe.fused_moe.layer.get_ep_group", return_value=ep_mock), \
         patch("omni.layers.moe.fused_moe.layer.get_tp_group", return_value=tp_mock), \
         patch("omni.layers.moe.fused_moe.layer.get_world_group", return_value=world_mock), \
         patch("torch.distributed.all_to_all_single", side_effect=mock_dist_all_to_all_single), \
         patch("torch.distributed.all_reduce", side_effect=mock_dist_all_reduce):

        model = FusedMoE(
            num_experts=num_experts_global,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=dtype,
            reduce_results=True, # Enable reduction to test the full flow
            tp_size=1
        ).to(device).to(dtype)
        
        # 3. Constant Init
        w_val = 0.1
        with torch.no_grad():
            if hasattr(model.quant_method, "process_weights_after_loading"):
                torch.nn.init.constant_(model.w13_weight, w_val)
                torch.nn.init.constant_(model.w2_weight, w_val)
                model.quant_method.process_weights_after_loading(model)

        # 4. Inputs
        batch_size = 2
        seq_len = 8 
        x_val = 0.5
        x_3d = torch.full((batch_size, seq_len, hidden_size), x_val, device=device, dtype=dtype)
        
        # 5. Golden Calculation
        golden_out_3d = moe_golden_constant(
            x_3d, top_k, w_val, w_val, hidden_size, intermediate_size
        )

        # 6. Forward Setup
        topk_weights = torch.full((batch_size * seq_len, top_k), 1.0/top_k, device=device, dtype=dtype)
        topk_ids = torch.zeros((batch_size * seq_len, top_k), device=device, dtype=torch.int32)
        
        # Routing: Local (0) and Remote (2)
        # Note: Since we mocked all_to_all as loopback, the "Remote" tokens will just return to us.
        # This is fine for testing the 'new_empty' stability.
        topk_ids[:, 0] = 0
        topk_ids[:, 1] = 2

        x_2d = x_3d.view(-1, hidden_size)

        mock_metadata = MagicMock()
        mock_metadata.prefill = None 
        mock_metadata.num_prefills = 0
        mock_metadata.num_decode_tokens = x_2d.shape[0]
        
        with patch("omni.layers.moe.fused_moe.layer.get_forward_context") as mock_ctx:
            mock_ctx.return_value.attn_metadata = mock_metadata

            raw_output = model(
                hidden_states=x_2d,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                pertoken_scale=None,
                attn_metadata=mock_metadata
            )

            # 1. Unpack the Tuple: Get Index [1] (gathered_tokens)
            if isinstance(raw_output, tuple):
                gathered_tokens = raw_output[1]
            else:
                gathered_tokens = raw_output

            # 2. Perform Weighted Sum Reduction (Required because the layer returns expanded tokens)
            #    Reshape: [Batch*Seq, TopK, Hidden]
            gathered_reshaped = gathered_tokens.view(x_2d.shape[0], top_k, hidden_size)
            
            #    Weight:  [Batch*Seq, TopK, 1]
            weighted_tokens = gathered_reshaped * topk_weights.unsqueeze(-1)
            
            #    Sum:     [Batch*Seq, Hidden]
            model_out_2d = weighted_tokens.sum(dim=1)
            
            # 3. View as 3D for validation
            model_out_3d = model_out_2d.view(batch_size, seq_len, hidden_size)

            # Debug print
            print(f"Rank {rank} Sample: Golden={golden_out_3d[0,0,0].item():.4f}, Model={model_out_3d[0,0,0].item():.4f}")

            assert torch.allclose(model_out_3d, golden_out_3d, atol=5e-3, rtol=5e-3), \
                f"Rank {rank}: Output mismatch"

def _logic_fused_moe_ep_prefill(device_id, rank, world_size, hidden_size, num_experts_global, top_k):
    """
    Verifies FusedMoE correctness in **Prefill Phase** (EP mode).
    """
    from omni.layers.moe.fused_moe.layer import FusedMoE
    
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    dtype = torch.float16
    intermediate_size = hidden_size * 2
    
    # Mock Groups
    ep_mock = MagicMock(); ep_mock.world_size = world_size; ep_mock.rank_in_group = rank
    tp_mock = MagicMock(); tp_mock.world_size = 1; tp_mock.rank_in_group = 0
    world_mock = MagicMock(); world_mock.world_size = world_size; world_mock.rank_in_group = rank

    with patch("omni.layers.moe.fused_moe.layer.get_ep_group", return_value=ep_mock), \
         patch("omni.layers.moe.fused_moe.layer.get_tp_group", return_value=tp_mock), \
         patch("omni.layers.moe.fused_moe.layer.get_world_group", return_value=world_mock), \
         patch("torch.distributed.all_to_all_single", side_effect=mock_dist_all_to_all_single), \
         patch("torch.distributed.all_reduce", side_effect=mock_dist_all_reduce):

        model = FusedMoE(
            num_experts=num_experts_global,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=dtype,
            reduce_results=True,
            tp_size=1
        ).to(device).to(dtype)
        
        # Constant Init
        w_val = 0.1
        with torch.no_grad():
            if hasattr(model.quant_method, "process_weights_after_loading"):
                torch.nn.init.constant_(model.w13_weight, w_val)
                torch.nn.init.constant_(model.w2_weight, w_val)
                model.quant_method.process_weights_after_loading(model)

        # Inputs
        batch_size = 2
        seq_len = 8 
        x_val = 0.5
        x_3d = torch.full((batch_size, seq_len, hidden_size), x_val, device=device, dtype=dtype)
        
        golden_out_3d = moe_golden_constant(x_3d, top_k, w_val, w_val, hidden_size, intermediate_size)

        topk_weights = torch.full((batch_size * seq_len, top_k), 1.0/top_k, device=device, dtype=dtype)
        topk_ids = torch.zeros((batch_size * seq_len, top_k), device=device, dtype=torch.int32)
        topk_ids[:, 0] = 0
        topk_ids[:, 1] = 2

        x_2d = x_3d.view(-1, hidden_size)

        mock_metadata = MagicMock()
        mock_metadata.prefill = MagicMock() # Triggers is_prefill=True
        mock_metadata.num_prefills = 1
        mock_metadata.num_decode_tokens = 0
        
        with patch("omni.layers.moe.fused_moe.layer.get_forward_context") as mock_ctx:
            mock_ctx.return_value.attn_metadata = mock_metadata
            
            raw_output = model(
                hidden_states=x_2d,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                pertoken_scale=None,
                attn_metadata=mock_metadata
            )

            # 1. Unpack the Tuple: Get Index [1] (gathered_tokens)
            if isinstance(raw_output, tuple):
                gathered_tokens = raw_output[1]
            else:
                gathered_tokens = raw_output

            # 2. Perform Weighted Sum Reduction (Required because the layer returns expanded tokens)
            #    Reshape: [Batch*Seq, TopK, Hidden]
            gathered_reshaped = gathered_tokens.view(x_2d.shape[0], top_k, hidden_size)
            
            #    Weight:  [Batch*Seq, TopK, 1]
            weighted_tokens = gathered_reshaped * topk_weights.unsqueeze(-1)
            
            #    Sum:     [Batch*Seq, Hidden]
            model_out_2d = weighted_tokens.sum(dim=1)
            
            # 3. View as 3D for validation
            model_out_3d = model_out_2d.view(batch_size, seq_len, hidden_size)

            # Debug print
            print(f"Rank {rank} Sample: Golden={golden_out_3d[0,0,0].item():.4f}, Model={model_out_3d[0,0,0].item():.4f}")

            assert torch.allclose(model_out_3d, golden_out_3d, atol=5e-3, rtol=5e-3), \
                f"Rank {rank}: Output mismatch"

@pytest.mark.parametrize("num_experts", [4])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("hidden_size", [128])
def test_fused_moe_ep_correctness(distributed_worker_pool, num_experts, top_k, hidden_size):
    distributed_worker_pool(_logic_fused_moe_ep_constant, hidden_size, num_experts, top_k)

@pytest.mark.parametrize("num_experts", [4])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("hidden_size", [128])
def test_fused_moe_ep_prefill(distributed_worker_pool, num_experts, top_k, hidden_size):
    distributed_worker_pool(_logic_fused_moe_ep_prefill, hidden_size, num_experts, top_k)