from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

import omni.models.pangu.pangu_pro_moe_v2.pangu_moe_v2 as pangu_mod


def _metadata(attn_state):
    return SimpleNamespace(attn_state=attn_state)


def test_pangu_moe_paths_on_npu(npu_device):
    torch.manual_seed(4)
    torch_npu.npu.manual_seed(4)

    block = pangu_mod.PanguProMoEV2MoEBlock.__new__(pangu_mod.PanguProMoEV2MoEBlock)
    block.is_init_gate = False
    block.gate = SimpleNamespace(
        weight=torch.nn.Parameter(
            torch.randn((1, 1), device=npu_device, dtype=torch.float16)
        )
    )

    def _prefill(hidden_states, residual, attn_metadata, layer_id, *args, **kwargs):
        assert hidden_states.device == npu_device
        return torch.zeros_like(hidden_states), residual

    def _decode(hidden_states, residual, attn_metadata, layer_id, next_attention_weights=None, *args, **kwargs):
        assert hidden_states.device == npu_device
        return torch.ones_like(hidden_states), residual

    block._forward_prefill_norm = Mock(side_effect=_prefill)
    block._forward_decode_norm = Mock(side_effect=_decode)

    hidden_states = torch.randn((2, 4), device=npu_device, dtype=torch.float16)
    residual = torch.zeros_like(hidden_states)

    prefill_meta = _metadata(pangu_mod.AscendAttentionState.PrefillNoCache)
    decode_meta = _metadata(pangu_mod.AscendAttentionState.DecodeOnly)

    out_prefill, res_prefill = block.forward(
        hidden_states, residual, prefill_meta, layer_id=0, is_hybrid_chunked_prefill_graph_mode=False
    )
    assert torch.all(out_prefill == 0)
    assert res_prefill is residual or torch.all(res_prefill == residual)
    assert block.is_init_gate

    out_decode, _ = block.forward(
        hidden_states, residual, decode_meta, layer_id=0, is_hybrid_chunked_prefill_graph_mode=False
    )
    assert torch.all(out_decode == 1)

    out_hybrid, _ = block.forward(
        hidden_states, residual, prefill_meta, layer_id=0, is_hybrid_chunked_prefill_graph_mode=True
    )
    assert torch.all(out_hybrid == 1)

    print(f"NPU_DEVICE={npu_device}")
    print("PANGU_MOE_NPU_OK")
