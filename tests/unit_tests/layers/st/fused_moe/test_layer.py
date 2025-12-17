import pytest
import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

from ..distributed_test_common import distributed_worker_pool

# --- Golden Reference ---

def swiglu_golden(x: torch.Tensor) -> torch.Tensor:
    x0, x1 = x.chunk(2, dim=-1)
    return F.silu(x0) * x1

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
    # hidden_states is all 0.5
    input_val = hidden_states[0,0,0].item() # 0.5
    
    # 2. W13 Projection (Gate + Up)
    # Matrix [Hidden, 2*Inter]. All values = w13_val (0.1)
    # Dot Product = Sum(Input * Weight) over Hidden dim
    # val = HiddenSize * input_val * w13_val
    dot_w13 = hidden_size * input_val * w13_val
    
    # 3. SwiGLU
    # Gate = dot_w13, Up = dot_w13
    # Out = SiLU(Gate) * Up
    hidden_act = F.silu(torch.tensor(dot_w13)) * dot_w13
    hidden_act_val = hidden_act.item()
    
    # 4. W2 Projection
    # Matrix [Inter, Hidden]. All values = w2_val (0.1)
    # Dot Product = Sum(Act * Weight) over Inter dim
    # val = InterSize * hidden_act_val * w2_val
    dot_w2 = inter_size * hidden_act_val * w2_val
    
    # 5. Weighted Sum
    # Logits are 0 -> Weights are uniform (1/TopK)
    # Experts are identical (constant weights)
    # Sum = (ExpertOut * 1/TopK) + (ExpertOut * 1/TopK) + ...
    # Sum = ExpertOut * (Sum(1/TopK)) = ExpertOut * 1.0
    final_val = dot_w2
    
    return torch.full_like(hidden_states, final_val)

# --- Test Logic ---

def _logic_fused_moe_ep_constant(device, rank, world_size, hidden_size, num_experts_global, top_k):
    """
    EP Test with Deterministic Constants.
    Input: 0.5
    Weights: 0.1
    """
    from omni.layers.moe.fused_moe.layer import FusedMoE
    
    device = torch.device(f"npu:{device}")
    dtype = torch.float16
    intermediate_size = hidden_size * 2
    
    # 1. Mock Distributed Environment (EP=2, TP=1)
    ep_mock = MagicMock(); ep_mock.world_size = world_size; ep_mock.rank_in_group = rank
    tp_mock = MagicMock(); tp_mock.world_size = 1; tp_mock.rank_in_group = 0
    world_mock = MagicMock(); world_mock.world_size = world_size; world_mock.rank_in_group = rank

    with patch("omni.layers.moe.fused_moe.layer.get_ep_group", return_value=ep_mock), \
         patch("omni.layers.moe.fused_moe.layer.get_tp_group", return_value=tp_mock), \
         patch("omni.layers.moe.fused_moe.layer.get_world_group", return_value=world_mock):
         
        model = FusedMoE(
            num_experts=num_experts_global,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=dtype,
            reduce_results=False, 
            tp_size=1
        ).to(device).to(dtype)
        
        # 2. Constant Init
        w_val = 0.1
        if hasattr(model.quant_method, "process_weights_after_loading"):
            torch.nn.init.constant_(model.w13_weight, w_val)
            torch.nn.init.constant_(model.w2_weight, w_val)
            model.quant_method.process_weights_after_loading(model)

        # 3. Inputs (Constant 0.5)
        batch_size = 2
        seq_len = 8 
        x_val = 0.5
        x_3d = torch.full((batch_size, seq_len, hidden_size), x_val, device=device, dtype=dtype)
        
        # 4. Golden Calculation (Math-based)
        golden_out_3d = moe_golden_constant(
            x_3d, top_k, w_val, w_val, hidden_size, intermediate_size
        )

        # 5. Forward
        # Uniform logits -> Uniform weights
        topk_weights = torch.full((batch_size * seq_len, top_k), 1.0/top_k, device=device, dtype=dtype)
        topk_ids = torch.zeros((batch_size * seq_len, top_k), device=device, dtype=torch.int32)
        # Point to experts 0 and 1 (Rank 0 has 0,1. Rank 1 has 2,3. But globally 0,1 should work if EP handles it)
        # To ensure valid routing in EP test:
        # Rank 0 should process some, Rank 1 should process some.
        # Let's route to Expert 0 (Rank 0) and Expert 2 (Rank 1).
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
            
            # Unpack: (hidden_states, gathered_tokens, ...)
            gathered_tokens = raw_output[1] if isinstance(raw_output, tuple) else raw_output

            # --- REDUCTION ---
            # Reshape [TotalTokens, TopK, Hidden]
            gathered_reshaped = gathered_tokens.view(x_2d.shape[0], top_k, hidden_size)
            # Weighted Sum
            weighted_tokens = gathered_reshaped * topk_weights.unsqueeze(-1)
            model_out_2d = weighted_tokens.sum(dim=1)

        # 6. Verify
        model_out_3d = model_out_2d.view(batch_size, seq_len, hidden_size)
        
        # Check Values
        # Print a sample to debug if it fails
        print(f"Rank {rank} Sample: Golden={golden_out_3d[0,0,0].item():.4f}, Model={model_out_3d[0,0,0].item():.4f}")
        
        assert torch.allclose(model_out_3d, golden_out_3d, atol=2e-3, rtol=2e-3), \
            f"Rank {rank}: Constant Output mismatch"

def _logic_expert_selection(device, rank, world_size):
    from omni.layers.moe.fused_moe.layer import FusedMoE
    device = torch.device(f"npu:{device}")
    logits = torch.zeros(4, 8, device=device)
    mock_metadata = MagicMock()
    mock_metadata.prefill = None
    with patch("omni.layers.moe.fused_moe.layer.get_forward_context") as mock_ctx:
        mock_ctx.return_value.attn_metadata = mock_metadata
        topk_weights, topk_ids, row_idx = FusedMoE.select_experts(
            None, logits, 2, False, True, "softmax"
        )
    assert 0 in topk_ids[0]

@pytest.mark.parametrize("num_experts", [4])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("hidden_size", [128])
def test_fused_moe_ep_correctness(distributed_worker_pool, num_experts, top_k, hidden_size):
    distributed_worker_pool(_logic_fused_moe_ep_constant, hidden_size, num_experts, top_k)

def test_moe_routing_logic(distributed_worker_pool):
    distributed_worker_pool(_logic_expert_selection)

def _logic_fused_moe_ep_prefill(device, rank, world_size, hidden_size, num_experts_global, top_k):
    """
    Verifies FusedMoE correctness in **Prefill Phase** (EP mode).
    Triggers is_prefill=True path in moe_infer_fusion.
    """
    from omni.layers.moe.fused_moe.layer import FusedMoE
    
    device = torch.device(f"npu:{device}")
    dtype = torch.float16
    intermediate_size = hidden_size * 2
    
    # 1. Mock Distributed Environment (EP=2, TP=1)
    ep_mock = MagicMock(); ep_mock.world_size = world_size; ep_mock.rank_in_group = rank
    tp_mock = MagicMock(); tp_mock.world_size = 1; tp_mock.rank_in_group = 0
    world_mock = MagicMock(); world_mock.world_size = world_size; world_mock.rank_in_group = rank

    with patch("omni.layers.moe.fused_moe.layer.get_ep_group", return_value=ep_mock), \
         patch("omni.layers.moe.fused_moe.layer.get_tp_group", return_value=tp_mock), \
         patch("omni.layers.moe.fused_moe.layer.get_world_group", return_value=world_mock):
         
        model = FusedMoE(
            num_experts=num_experts_global,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=dtype,
            reduce_results=False, 
            tp_size=1
        ).to(device).to(dtype)
        
        # 2. Constant Init (Stable Math: Input=0.5, Weight=0.1)
        w_val = 0.1
        if hasattr(model.quant_method, "process_weights_after_loading"):
            torch.nn.init.constant_(model.w13_weight, w_val)
            torch.nn.init.constant_(model.w2_weight, w_val)
            model.quant_method.process_weights_after_loading(model)

        # 3. Inputs (Constant 0.5)
        # Prefill typically uses larger sequence lengths, but 8 is sufficient to trigger logic
        batch_size = 2
        seq_len = 8 
        x_val = 0.5
        x_3d = torch.full((batch_size, seq_len, hidden_size), x_val, device=device, dtype=dtype)
        
        # 4. Golden Calculation
        golden_out_3d = moe_golden_constant(
            x_3d, top_k, w_val, w_val, hidden_size, intermediate_size
        )

        # 5. Forward setup
        # Uniform weights for deterministic reduction
        topk_weights = torch.full((batch_size * seq_len, top_k), 1.0/top_k, device=device, dtype=dtype)
        topk_ids = torch.zeros((batch_size * seq_len, top_k), device=device, dtype=torch.int32)
        # Route to Expert 0 (Rank 0) and Expert 2 (Rank 1) to force cross-rank comms check
        topk_ids[:, 0] = 0
        topk_ids[:, 1] = 2

        x_2d = x_3d.view(-1, hidden_size)

        # --- KEY CHANGE: Mock Metadata for PREFILL ---
        mock_metadata = MagicMock()
        # Setting 'prefill' to NOT None triggers is_prefill=True
        mock_metadata.prefill = MagicMock() 
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
            
            # Unpack Gathered Tokens (Index 1)
            gathered_tokens = raw_output[1] if isinstance(raw_output, tuple) else raw_output

            # --- REDUCTION ---
            gathered_reshaped = gathered_tokens.view(x_2d.shape[0], top_k, hidden_size)
            weighted_tokens = gathered_reshaped * topk_weights.unsqueeze(-1)
            model_out_2d = weighted_tokens.sum(dim=1)

        # 6. Validation
        model_out_3d = model_out_2d.view(batch_size, seq_len, hidden_size)
        
        assert torch.allclose(model_out_3d, golden_out_3d, atol=1e-3, rtol=1e-3), \
            f"Rank {rank}: Output mismatch in Prefill Mode"


@pytest.mark.parametrize("num_experts", [4])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("hidden_size", [128])
def test_fused_moe_ep_prefill(distributed_worker_pool, num_experts, top_k, hidden_size):
    """
    Tests FusedMoE in **Prefill Phase** (EP mode).
    """
    distributed_worker_pool(_logic_fused_moe_ep_prefill, hidden_size, num_experts, top_k)