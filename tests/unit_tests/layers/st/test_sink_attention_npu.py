import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationLevel

import omni.layers.attention.backend.attention as attn_mod

torch_npu = pytest.importorskip("torch_npu")

GOLDEN_PATH_ENV = "OMNI_SINK_ATTENTION_GOLDEN_PATH"
GOLDEN_MODE_ENV = "OMNI_SINK_ATTENTION_GOLDEN_MODE"
GOLDEN_MODE_RECORD = "record"
GOLDEN_TOL = 1e-5


def _load_golden(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def _save_golden(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


class _DummyLayer:
    _k_scale_float = 1.0
    _v_scale_float = 1.0


def _dummy_vllm_config(enable_chunked_prefill: bool) -> SimpleNamespace:
    return SimpleNamespace(
        npu_compilation_config=SimpleNamespace(level=CompilationLevel.NO_COMPILATION),
        kv_transfer_config=None,
        additional_config={},
        scheduler_config=SimpleNamespace(enable_chunked_prefill=enable_chunked_prefill),
    )


def test_forward_sink_attention_npu(npu_device, monkeypatch):
    monkeypatch.setattr(attn_mod, "get_current_vllm_config", lambda: _dummy_vllm_config(False))

    torch.manual_seed(2)
    torch_npu.npu.manual_seed(2)

    golden_path_value = os.environ.get(GOLDEN_PATH_ENV)
    golden_mode = os.environ.get(GOLDEN_MODE_ENV, "").lower()
    default_golden_path = Path(__file__).with_name("sink_attention_npu_golden.pt")
    golden_path = Path(golden_path_value) if golden_path_value else default_golden_path
    if golden_mode == GOLDEN_MODE_RECORD and not golden_path_value:
        raise AssertionError(
            f"{GOLDEN_MODE_ENV}=record requires {GOLDEN_PATH_ENV} to be set explicitly."
        )
    golden_payload = None
    if golden_mode != GOLDEN_MODE_RECORD:
        if not golden_path.exists():
            raise AssertionError(
                f"Golden file not found at {golden_path}; set {GOLDEN_PATH_ENV} to a valid path."
            )
        golden_payload = _load_golden(golden_path)

    num_heads = 1
    num_kv_heads = 1
    num_tokens = 1
    block_size = 128
    num_blocks = 1
    nz_dim = attn_mod.get_nz_dim()
    head_size = 64
    assert head_size % nz_dim == 0

    backend = attn_mod.AscendAttentionBackendImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0 / math.sqrt(head_size),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="float16",
    )

    input_dtype = torch.float16
    expected_config = {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "num_tokens": num_tokens,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "head_size": head_size,
        "dtype": str(input_dtype),
    }
    if golden_payload:
        golden_config = golden_payload.get("config")
        if golden_config and golden_config != expected_config:
            raise AssertionError(
                f"Golden config mismatch: expected {expected_config}, got {golden_config}"
            )

    def _load_or_rand(name, shape):
        if golden_payload:
            return golden_payload["inputs"][name].to(device=npu_device, dtype=input_dtype)
        return torch.randn(shape, device=npu_device, dtype=input_dtype)

    query = _load_or_rand("query", (num_tokens, num_heads * head_size))
    key = _load_or_rand("key", (num_tokens, num_kv_heads * head_size))
    value = _load_or_rand("value", (num_tokens, num_kv_heads * head_size))

    key_cache = torch.zeros(
        (num_blocks, num_kv_heads * head_size // nz_dim, block_size, nz_dim),
        device=npu_device,
        dtype=torch.float16,
    )
    value_cache = torch.zeros_like(key_cache)

    block_tables = torch.zeros((1, num_blocks), dtype=torch.int32, device=npu_device)
    query_lens = torch.tensor([num_tokens], dtype=torch.int32, device=npu_device)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=npu_device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device=npu_device)
    slot_indices = torch.zeros((num_tokens, 2), dtype=torch.int32, device=npu_device)

    metadata = attn_mod.AscendMetadata(
        num_actual_tokens=num_tokens,
        block_tables=block_tables,
        query_lens=query_lens,
        query_lens_list=query_lens.tolist(),
        seq_lens=seq_lens,
        seq_lens_list=seq_lens.tolist(),
        max_query_len=num_tokens,
        slot_mapping=slot_mapping,
        slot_indices=slot_indices,
        attn_state=attn_mod.AscendAttentionState.PrefillNoCache,
    )

    sink_key = _load_or_rand("sink_key", (block_size, num_kv_heads, head_size))
    sink_value = _load_or_rand("sink_value", (block_size, num_kv_heads, head_size))

    output = backend.forward_sink_attention(
        _DummyLayer(),
        query,
        key,
        value,
        (key_cache, value_cache),
        metadata,
        sink_key=sink_key,
        sink_value=sink_value,
    )

    assert output.shape == (num_tokens, num_heads * head_size)
    assert output.device == npu_device

    torch_npu.npu.synchronize()
    output_cpu = output.detach().cpu()

    if golden_mode == GOLDEN_MODE_RECORD:
        payload = {
            "seed": {"torch": 2, "torch_npu": 2},
            "config": expected_config,
            "inputs": {
                "query": query.detach().cpu(),
                "key": key.detach().cpu(),
                "value": value.detach().cpu(),
                "sink_key": sink_key.detach().cpu(),
                "sink_value": sink_value.detach().cpu(),
            },
            "output": output_cpu,
            "tol": GOLDEN_TOL,
        }
        _save_golden(golden_path, payload)
        print(f"SINK_ATTENTION_GOLDEN_SAVED={golden_path}")
    else:
        golden_output = golden_payload["output"]
        if not torch.allclose(output_cpu, golden_output, atol=GOLDEN_TOL, rtol=GOLDEN_TOL):
            max_diff = (output_cpu - golden_output).abs().max().item()
            raise AssertionError(
                f"SINK_ATTENTION_GOLDEN_MISMATCH max_abs_diff={max_diff} tol={GOLDEN_TOL}"
            )
        print("SINK_ATTENTION_GOLDEN_OK")

    print(f"NPU_DEVICE={npu_device}")
    print("SINK_ATTENTION_NPU_OK")
