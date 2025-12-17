# Overview

We choose a Persistent Worker Pool model (which takes ~15 seconds) for high performance distributed tests.

Instead of initializing the NPU/HCCL environment for every single test case, we spin up two worker processes once at the start of the module. These workers stay alive, accept tasks, and execute them in an already-initialized distributed environment.

# How to Add a New Test
Adding a distributed test requires separating your code into two parts:

The Logic Function: Runs on the Worker (Rank 0, Rank 1, etc.).

The Test Driver: Runs on the Host (Pytest), dispatching tasks to workers.

Step 1: Define the Logic Function
This function contains the actual PyTorch/Omni code.

The Golden Rules:
Must be Top-Level: The function must be defined at the global scope (so it can be pickled).
Local Imports: You must import omni.layers inside this function.
Deterministic Seeds: You must reset the seed at the start.

logic_functions.py or inside test_linear_optimized.py
```python
def _logic_my_new_feature(local_rank, world_size, input_size, dtype):
    """
    This function runs INSIDE the persistent worker process.
    """
    # [CRITICAL] 1. Reset Seed: Workers are long-lived; prevent RNG drift.
    torch.manual_seed(0) 

    # [CRITICAL] 2. Local Import: Ensures the layer picks up the latest config.
    # If you rely on top-level imports, the worker might use a cached 
    # version of the class with the wrong TP size.
    from omni.layers.linear import AscendRowParallelLinear

    # 3. Standard Test Logic
    device = torch.device("npu")
    
    # Create layer
    layer = AscendRowParallelLinear(input_size, ...).to(device)
    
    # Create input
    x = torch.randn(input_size, dtype=dtype, device=device)
    
    # Run
    out, _ = layer(x)
    
    # Verify (e.g., against a local Golden calculation)
    expected = ... 
    assert torch.allclose(out, expected)
```
Step 2: Write the Pytest Driver
This function runs in the main process. It generates test data (if needed) and dispatches the logic to the pool.

```python
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_my_new_feature_distributed(distributed_worker_pool, dtype):
    """
    The 'distributed_worker_pool' fixture manages the workers.
    """
    input_size = 64
    world_size = 2 # Implied by the pool
    
    # You can call the pool like a function.
    # Arguments: (Logic Function, *Args...)
    distributed_worker_pool(
        _logic_my_new_feature,  # The function defined above
        input_size,             # Arg 1
        dtype                   # Arg 2
    )
```

# Deep Dive: Common Pitfalls
1. The "Local Import" Requirement
Symptom: Your test fails with Size Mismatch. You expect a gathered tensor (size 60), but get a local shard (size 30). Cause: The Ascend layers read global configuration (like dense_mlp_tp_size) when the module is first imported. Why it breaks: The worker process imports omni at startup. If you change the config in your test, the worker still holds the old class definition. Fix: By importing inside _logic_..., you ensure the worker uses the reloaded module with the correct configuration.

2. Pickling Errors
Symptom: AttributeError: Can't pickle local object... Cause: You defined your logic function inside another function or as a lambda. Fix: Move _logic_... to the global scope of the file.

3. Randomness Divergence
Symptom: Test passes on Rank 0 but fails on Rank 1, or fails intermittently. Cause: Workers do not restart between tests. The RNG state advances continuously. Rank 0 might be at RNG step 100, while Rank 1 is at step 500. Fix: Always call torch.manual_seed(TEST_SEED) as the first line of your logic function.

# Debugging
If a worker crashes, the main process will receive a RuntimeError containing the traceback from the specific rank that failed.

Example Error:

```
RuntimeError: Rank 1 failed:
Traceback (most recent call last):
  File ...
    assert out.shape == expected.shape
AssertionError: ...
```
# Tips:

You cannot use ipdb or pdb breakpoints inside the logic function (it's in a separate process).

Use print(f"Rank {local_rank}: ...", flush=True) for debugging.

If the pool hangs indefinitely, it usually means a collective operation (like all_gather) was called on one rank but not the other (e.g., inside an if local_rank == 0: block).