import sys
import math
import pytest
import torch
import torch_npu
from unittest.mock import MagicMock

from omni.layers.rotary_embedding import (
    RotaryEmbeddingTorchNpu, 
    LinearScalingRotaryEmbedding, 
    YaRNScalingRotaryEmbedding
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding as GPURotaryEmbedding
from vllm.model_executor.layers.rotary_embedding import YaRNScalingRotaryEmbedding as GPUYaRNScalingRotaryEmbedding

from .distributed_test_common import parse_ascend_devices

def apply_rotary_ref(x, cos, sin):
    # NeoX style: split last dim into two halves
    x1, x2 = x.chunk(2, dim=-1)
    x_new = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (x_new * sin)

class TestRotaryEmbeddingNPU:
    
    def setup_method(self):
        first_die, _ = parse_ascend_devices()
        self.device = torch.device(f"npu:{first_die}")
        self.dtype = torch.float32 
        self.base = 10000
        self.head_size = 128
        self.rotary_dim = 128

    def test_fused_ops_identity_pos0(self):
        """
        Test Fused NPU Ops at Position 0.
        """
        layer = RotaryEmbeddingTorchNpu(
            head_size=self.head_size,
            rotary_dim=self.rotary_dim,
            max_position_embeddings=2048,
            base=self.base,
            is_neox_style=True,
            dtype=self.dtype
        ).to(self.device)

        # Use 2D Input (Batch*Seq=1, Hidden=128)
        q = torch.ones((1, self.head_size), dtype=self.dtype, device=self.device)
        k = torch.ones((1, self.head_size), dtype=self.dtype, device=self.device)
        positions = torch.tensor([0], device=self.device, dtype=torch.long)

        q_out, k_out = layer(positions, q, k)

        assert torch.allclose(q_out, q, atol=1e-5)
        assert torch.allclose(k_out, k, atol=1e-5)

    def test_fused_ops_specific_rotation(self):
        """
        Test Fused NPU Ops at Position 1 with specific values.
        """
        layer = RotaryEmbeddingTorchNpu(
            head_size=self.head_size,
            rotary_dim=self.rotary_dim,
            base=self.base,
            is_neox_style=True,
            dtype=self.dtype
        ).to(self.device)

        q = torch.zeros((1, self.head_size), dtype=self.dtype, device=self.device)
        # x1=1.0 at index 0, x2=0.0 at index 64
        q[0, 0] = 1.0 
        q[0, 64] = 0.0 
        
        k = torch.zeros_like(q)
        positions = torch.tensor([1], device=self.device, dtype=torch.long)

        expected_cos = math.cos(1.0)
        expected_sin = math.sin(1.0)

        q_out, k_out = layer(positions, q, k)

        res_0 = q_out[0, 0].item()
        res_64 = q_out[0, 64].item()

        assert math.isclose(res_0, expected_cos, abs_tol=1e-4)
        assert math.isclose(res_64, expected_sin, abs_tol=1e-4)

    def test_small_ops_path_correctness(self):
        """
        Test 'Small Ops' path (Dim 64).
        """
        dim = 64
        layer = RotaryEmbeddingTorchNpu(
            head_size=dim,
            rotary_dim=dim,
            base=self.base,
            is_neox_style=True,
            dtype=self.dtype
        ).to(self.device)

        # Use 2D Input (Batch*Seq=2, Hidden=64)
        q = torch.randn((2, dim), dtype=self.dtype, device=self.device)
        k = torch.randn((2, dim), dtype=self.dtype, device=self.device)
        positions = torch.tensor([0, 1], device=self.device, dtype=torch.long)

        # 1. Run Layer
        q_npu, k_npu = layer(positions, q, k)

        # 2. Run Reference
        # Use layer.cos / layer.sin which are expanded to full dim
        # layer.cos shape: [MaxPos, Dim]
        # We index to get [2, 64]
        cos_ref = layer.cos[positions]
        sin_ref = layer.sin[positions]
        
        q_ref = apply_rotary_ref(q, cos_ref, sin_ref)
        
        assert torch.allclose(q_npu, q_ref, atol=1e-5)

# Setup mocked base methods to behave like standard rotary so we can isolate the user code
def mock_compute_inv_freq(base):
    # Standard 1 / base^(i/dim)
    return 1.0 / (base ** (torch.arange(0, 128, 2).float() / 128))

GPURotaryEmbedding._compute_inv_freq = mock_compute_inv_freq
GPUYaRNScalingRotaryEmbedding._compute_inv_freq = mock_compute_inv_freq

class TestRotaryVariantsNPU:
    
    def setup_method(self):
        first_die, _ = parse_ascend_devices()
        self.device = torch.device(f"npu:{first_die}")
        self.dtype = torch.float32 
        self.base = 10000
        self.head_size = 128
        self.rotary_dim = 128

    # --- Linear Scaling Tests ---
    def test_linear_scaling_freqs(self):
        """
        Verify Linear Scaling logic: t = t / scaling_factor.
        Effectively, the frequency should decrease (period increases).
        """
        scaling_factor = 2.0
        layer = LinearScalingRotaryEmbedding(
            head_size=self.head_size,
            rotary_dim=self.rotary_dim,
            max_position_embeddings=2048,
            base=self.base,
            is_neox_style=True,
            scaling_factor=scaling_factor,
            dtype=self.dtype
        ).to(self.device)

        # Logic check:
        # Standard: angle = pos * freq
        # Linear:   angle = (pos / scale) * freq
        
        # Test at Position 2. With scale 2.0, this should behave like Position 1 unscaled.
        pos_idx = 2
        positions = torch.tensor([pos_idx], device=self.device, dtype=torch.long)
        
        # Get cos/sin from scaled layer
        cos = layer.cos[pos_idx].cpu()
        
        # Calculate expected "Standard" cos at pos = 2 / 2 = 1
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        expected_angle = 1.0 * inv_freq # pos=1
        
        # NeoX style duplication
        expected_cos = torch.cos(expected_angle)
        expected_cos = torch.cat([expected_cos, expected_cos], dim=-1) # Expand to [128]
        
        assert torch.allclose(cos, expected_cos, atol=1e-5), \
            "Linear scaling did not scale position effectively (Pos 2 scaled by 2 should match Pos 1)"

    def test_linear_scaling_max_len(self):
        """
        Verify that max_len is expanded by scaling factor.
        """
        max_pos = 100
        scale = 4.0
        layer = LinearScalingRotaryEmbedding(
            head_size=self.head_size, rotary_dim=self.rotary_dim,
            max_position_embeddings=max_pos, base=self.base, is_neox_style=True,
            scaling_factor=scale, dtype=self.dtype
        ).to(self.device)
        
        # Cache size should be max_pos * scale
        expected_len = int(max_pos * scale)
        assert layer.cos.shape[0] == expected_len
        assert layer.sin.shape[0] == expected_len

    # --- YaRN Scaling Tests ---
    def test_yarn_init_structure(self):
        """
        Verify YaRN initializes correctly and sets mscale.
        """

        layer_yarn = YaRNScalingRotaryEmbedding(
            head_size=self.head_size, rotary_dim=self.rotary_dim,
            max_position_embeddings=2048, base=self.base, is_neox_style=True,
            scaling_factor=2.0, dtype=self.dtype
        ).to(self.device)
        
        layer_std = RotaryEmbeddingTorchNpu(
            head_size=self.head_size, rotary_dim=self.rotary_dim,
            max_position_embeddings=2048, base=self.base, is_neox_style=True,
            dtype=self.dtype
        ).to(self.device)
        
        # YaRN should have a larger cache due to scaling
        assert layer_yarn.cos.shape[0] == 2048 * 2
        
        # YaRN modifies values via mscale and interpolation.
        # Check that cache at pos 1 is different from standard cache at pos 1
        # (even if close, they shouldn't be identical bytes)
        assert not torch.equal(layer_yarn.cos[1], layer_std.cos[1]), \
            "YaRN layer should produce different embeddings than standard layer"