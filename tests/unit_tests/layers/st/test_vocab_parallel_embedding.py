import pytest
import torch
import torch.nn.functional as F
from omni.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
    get_masked_input_and_mask
)
from .distributed_test_common import distributed_worker_pool

pytestmark = [
    pytest.mark.npu_2cards_4dies,
    pytest.mark.npu,
]

def _logic_masking_helper(device, rank, world_size):
    """
    Tests the standalone helper function get_masked_input_and_mask.
    This logic is stateless and can be run identically on any rank.
    """
    device = torch.device(f"npu:{device}")
    
    # Define a simplified vocab range logic
    # Assume total vocab is [0, 20), padding to 20
    # Rank 0 covers [0, 10), Rank 1 covers [10, 20)
    
    # Simulating Rank 0 parameters
    org_vocab_start = 0
    org_vocab_end = 10
    num_padding = 0
    added_start = 20 # No added vocab for simplicity
    added_end = 20
    
    # Input tensor: [0 (valid), 5 (valid), 10 (invalid for rank 0), 15 (invalid for rank 0)]
    input_ids = torch.tensor([0, 5, 10, 15], dtype=torch.int32, device=device)
    
    masked_input, mask = get_masked_input_and_mask(
        input_ids, 
        org_vocab_start, org_vocab_end, num_padding,
        added_start, added_end
    )
    
    # Expectation for Rank 0:
    # Indices < 10 are valid. Indices >= 10 are masked.
    # mask: True means INVALID (masked out). The function returns ~vocab_mask, 
    # but let's check the implementation details. 
    # Implementation returns: input_, ~vocab_mask. 
    # vocab_mask is 1 where index is VALID. So output mask is 0 where valid (Keep), 1 where invalid (Mask).
    # Wait, strict reading of code: `return input_, ~vocab_mask`.
    # vocab_mask = (input inside range). ~vocab_mask has 1s where input is OUTSIDE range.
    
    expected_mask = torch.tensor([0, 0, 1, 1], dtype=torch.bool, device=device) # 0=Valid, 1=Invalid
    
    # The function shifts input by offset, then multiplies by valid mask.
    # For indices outside range, it should be 0.
    expected_input = torch.tensor([0, 5, 0, 0], dtype=torch.int32, device=device)
    
    assert torch.equal(mask, expected_mask), f"Rank 0 Mask mismatch. Got {mask}, expected {expected_mask}"
    assert torch.equal(masked_input, expected_input), f"Rank 0 Input value mismatch. Got {masked_input}"

def _logic_vocab_embedding_correctness(device, rank, world_size, vocab_size, hidden_size, dtype):
    """
    Verifies mathematical correctness of VocabParallelEmbedding against a local Golden reference.
    """
    device = torch.device(f"npu:{device}")
    torch.manual_seed(42) # Ensure deterministic weights across ranks if generated
    
    # 1. Setup Golden Reference (Full Model)
    # Using fixed tensors for stability
    # Weight: Simple arange to easily track values
    # Shape: [vocab_size, hidden_size]
    full_weight_cpu = torch.arange(vocab_size * hidden_size, dtype=dtype).reshape(vocab_size, hidden_size)
    full_weight = full_weight_cpu.to(device)
    
    # Input: Fixed indices covering both ranks
    # Rank 0 handles [0, vocab_size/2), Rank 1 handles [vocab_size/2, vocab_size)
    mid_point = vocab_size // 2
    # Input: [0, 1, mid_point, mid_point+1]
    input_ids = torch.tensor([0, 1, mid_point, mid_point + 1], device=device).long()
    
    # Calculate Golden Output
    # F.embedding handles lookup on the full weight matrix
    golden_output = F.embedding(input_ids, full_weight)
    
    # 2. Setup Distributed Model
    # We initialize the model, then overwrite weights to match the golden slice
    model = VocabParallelEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        params_dtype=dtype,
        prefix="test"
    ).to(device)
    
    # Ensure TP size is correct
    assert model.tp_size == world_size, "Model TP size does not match world size"
    
    # Manually load the correct slice of weights onto this rank
    # VocabParallelEmbedding partitions along dim 0 (vocab dimension)
    start_idx = model.shard_indices.org_vocab_start_index
    end_idx = model.shard_indices.org_vocab_end_index
    
    # Extract slice from full_weight corresponding to this rank
    my_weight_slice = full_weight[start_idx:end_idx, :]
    
    # Overwrite model weights
    with torch.no_grad():
        model.weight.copy_(my_weight_slice)
        
    # 3. Run Distributed Forward
    # forward_vocab performs masking, lookup, and all-reduce
    dist_output = model.forward_vocab(input_ids)
    
    # 4. Compare
    assert torch.allclose(dist_output, golden_output, atol=1e-3, rtol=1e-3), \
        f"Rank {rank}: Output mismatch.\nGolden: {golden_output}\nDist: {dist_output}"


def _logic_lmhead_correctness(device, rank, world_size, vocab_size, hidden_size, dtype):
    """
    Verifies ParallelLMHead (Linear projection to logits).
    Tests the standard Tensor Parallel behavior (Column Parallel for Linear, gathering outputs).
    """
    device = torch.device(f"npu:{device}")
    
    # 1. Setup Golden Reference
    # LM Head operation: Y = X * W^T
    # W shape: [vocab_size, hidden_size]
    # We construct W such that W^T is [hidden_size, vocab_size]
    
    # Create fixed weights: Value = row_idx + col_idx * 0.1
    # This ensures every element is distinct
    W_full = torch.zeros(vocab_size, hidden_size, dtype=dtype, device=device)
    for i in range(vocab_size):
        for j in range(hidden_size):
            W_full[i, j] = i + j * 0.1
            
    # Input X: [Batch=2, Hidden=hidden_size]
    # Fixed input
    X = torch.ones(2, hidden_size, dtype=dtype, device=device) * 0.5
    
    # Golden Calculation: Linear
    # F.linear(input, weight) -> input @ weight.T
    golden_logits = F.linear(X, W_full)
    
    # 2. Setup Distributed Model
    # Explicitly set parallel_lmhead=False to use standard tensor_model_parallel_all_gather
    # logic which is safer to test without specific group topology configuration.
    model = ParallelLMHead(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        params_dtype=dtype,
        parallel_lmhead=False 
    ).to(device)
    
    # 3. Load Weights
    # ParallelLMHead partitions the output dimension (vocab size) across ranks.
    # This is Column Parallelism w.r.t the Linear layer (Weight is [Out, In]).
    start_idx = model.shard_indices.org_vocab_start_index
    end_idx = model.shard_indices.org_vocab_end_index
    
    # W_full is [vocab, hidden]. We slice the first dimension.
    my_weight_slice = W_full[start_idx:end_idx, :]
    
    with torch.no_grad():
        model.weight.copy_(my_weight_slice)
        
    # 4. Run Forward
    # The forward pass expects (hidden_states, embedding_bias). 
    # We pass None for bias to isolate weight multiplication logic.
    dist_logits = model.forward(X, embedding_bias=None)
    
    # 5. Compare
    # Output should be gathered and match full vocab size [Batch, Vocab]
    assert dist_logits.shape == golden_logits.shape, \
        f"Shape mismatch: Got {dist_logits.shape}, Expected {golden_logits.shape}"
        
    assert torch.allclose(dist_logits, golden_logits, atol=1e-3, rtol=1e-3), \
        f"Rank {rank}: Logits mismatch."

def test_vocab_embedding_masking_logic(distributed_worker_pool):
    """
    Unit test for the helper function masking logic running on NPU.
    """
    distributed_worker_pool(_logic_masking_helper)

def test_vocab_embedding_distributed_fwd(distributed_worker_pool):
    """
    Tests the VocabParallelEmbedding forward pass:
    Input -> Masking -> Local Lookup -> AllReduce -> Output
    """
    vocab_size = 1024 # Divisible by 2 for easy splitting
    hidden_size = 64
    dtype = torch.float16
    
    distributed_worker_pool(
        _logic_vocab_embedding_correctness, 
        vocab_size, hidden_size, dtype
    )

def test_lmhead_distributed_fwd(distributed_worker_pool):
    """
    Tests the ParallelLMHead forward pass:
    Hidden -> Local Linear -> AllGather -> Logits
    """
    vocab_size = 1024
    hidden_size = 64
    dtype = torch.float16
    
    distributed_worker_pool(
        _logic_lmhead_correctness,
        vocab_size, hidden_size, dtype
    )