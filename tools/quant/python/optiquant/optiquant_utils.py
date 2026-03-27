"""Shared quantization utilities for INT4/INT8 weight quantization.

This module contains common functions used across different quantization
implementations including sszs quantization, packing/unpacking, and utilities.
"""

import re
from copy import deepcopy
import torch

# Constants
SCALE_CLAMP_MIN = 1e-5
LOSS_MODE = 2
MSE_EPSILON = 1e-10
MSE_EPSILON_RELATIVE = 1e-4
INT8_QMAX = 127.0

# QType regex pattern
QTYPE_PATTERN = r"sszs([0-9]+)g([0-9]+)a([0-9]+)b([0-9]+)sym([0-9]+)$"


class QType:
    """Quantization type configuration for sszs scheme.

    Attributes:
        desc: Description string.
        num_step: Number of optimization iterations.
        blk_size: Block/group size.
        is_act_integer: Whether activations are integer.
        numbits: Number of quantization bits.
        ssz_sym: Whether symmetric quantization.
        q_dim: Quantization dimension (set via dim() method).
    """

    def __init__(self, desc):
        """Initialize QType from description string.

        Args:
            desc: String like 'sszs50g0a0b4sym1'.

        Raises:
            ValueError: If description format is invalid.
        """
        self.desc = desc
        self.q_dim = None

        if desc.lower()[:3] == "ssz":
            res = re.match(QTYPE_PATTERN, desc)
            if res is None:
                raise ValueError("Invalid QType description: %s" % desc)

            self.num_step = int(res.group(1))
            self.blk_size = int(res.group(2))
            self.is_act_integer = bool(int(res.group(3)) > 0)
            self.numbits = int(res.group(4))
            self.ssz_sym = bool(int(res.group(5)) > 0)
        else:
            raise ValueError("Unsupported quantization type: %s" % desc)

    def dim(self, dim):
        """Create copy with specified quantization dimension.

        Args:
            dim: Dimension value.

        Returns:
            New QType instance with q_dim set.
        """
        out = deepcopy(self)
        out.q_dim = dim
        return out

    def __repr__(self):
        return "QType: %s   Dim: %s   ExpOffset: %s" % (
            self.desc, self.q_dim, self.exp_offset
        )


def get_qbits_minmax(numbits, is_sym):
    """Calculate min/max quantized values for given bit width.

    Args:
        numbits: Number of bits for quantization.
        is_sym: Whether symmetric quantization.

    Returns:
        Tuple of (qmin, qmax).
    """
    bit_max = 2 ** (numbits - 1) - 1

    if is_sym:
        bit_min = -bit_max
    else:
        bit_min = -bit_max - 1

    return bit_min, bit_max


def get_scale_offset(x, qW_min, qW_max, is_sym, is_act_integer):
    """Calculate scale and offset for quantization.

    Args:
        x: Input tensor.
        qW_min: Minimum quantized value.
        qW_max: Maximum quantized value.
        is_sym: Whether symmetric quantization.
        is_act_integer: Whether activations are integer.

    Returns:
        Tuple of (scale, offset).
    """
    scale = None
    offset = None

    if is_sym:
        xmax = torch.abs(x).max(dim=-1, keepdim=True)[0]

        if is_act_integer:
            scale = torch.round(xmax / qW_max).clamp(min=1)
        else:
            scale = (xmax / qW_max).clamp(min=SCALE_CLAMP_MIN)
    else:
        xmax = x.max(dim=-1, keepdim=True)[0]
        xmin = x.min(dim=-1, keepdim=True)[0]

        compare = ((xmax - xmin) < 1e-5).to(torch.int32)
        xmax = xmax * (1 - compare) + torch.max(
            torch.abs(xmax), torch.abs(xmin)
        ) * compare
        xmin = xmin * (1 - compare)

        scale = (xmax - xmin).clamp(min=SCALE_CLAMP_MIN) / (qW_max - qW_min)

        if is_act_integer:
            scale = torch.round(scale).clamp(min=1)
        else:
            scale = scale.clamp(min=SCALE_CLAMP_MIN)

        offset = torch.round(-xmin / scale) + qW_min

    return scale, offset


def get_quant(x, qW_min, qW_max, scale, offset=None):
    """Perform quantization.

    Args:
        x: Input tensor.
        qW_min: Minimum quantized value.
        qW_max: Maximum quantized value.
        scale: Scale factor.
        offset: Offset for asymmetric quantization.

    Returns:
        Quantized tensor.
    """
    if offset is not None:
        return torch.round(x / scale + offset).clamp(min=qW_min, max=qW_max)
    else:
        return torch.round(x / scale).clamp(min=qW_min, max=qW_max)


def get_dequant(x_quant, qW_min, qW_max, scale, offset=None):
    """Perform dequantization.

    Args:
        x_quant: Quantized tensor.
        qW_min: Minimum quantized value.
        qW_max: Maximum quantized value.
        scale: Scale factor.
        offset: Offset for asymmetric quantization.

    Returns:
        Dequantized tensor.
    """
    if offset is not None:
        return (x_quant - offset) * scale
    else:
        return x_quant * scale


def get_mseloss(x, qW_min, qW_max, scale=None, offset=None, quant=None, dequant=None):
    """Calculate MSE loss for quantization error.

    Args:
        x: Original tensor.
        qW_min: Minimum quantized value.
        qW_max: Maximum quantized value.
        scale: Scale factor.
        offset: Offset for asymmetric quantization.
        quant: Pre-computed quantized tensor.
        dequant: Pre-computed dequantized tensor.

    Returns:
        Mean absolute error along last dimension.
    """
    if quant is None and dequant is None:
        quant = get_quant(x, qW_min, qW_max, scale, offset)

    if dequant is None:
        dequant = get_dequant(quant, qW_min, qW_max, scale, offset)

    return torch.mean(
        torch.pow(torch.abs(x - dequant), LOSS_MODE), dim=-1, keepdim=True
    )


def quant_ssz(x, Q, qdim, init_scale=None, init_offset=None, init_quant=None,
              w8=False, w4=False):
    """SSZS quantization with iterative optimization.

    Args:
        x: Input tensor.
        Q: QType configuration.
        qdim: Quantization dimension.
        init_scale: Initial scale (optional).
        init_offset: Initial offset (optional).
        init_quant: Initial quantized tensor (optional).
        w8: Return INT8 format.
        w4: Return INT4 format.

    Returns:
        Quantized tensor and scale, or (weight, scale, bias) for w4.
    """
    num_step = Q.num_step
    groupsize = Q.blk_size
    is_act_integer = Q.is_act_integer
    numbits = Q.numbits
    is_ssz_sym = Q.ssz_sym
    shape = x.shape

    if groupsize != 0:
        shaped_x = x.view(-1, groupsize)
    else:
        shaped_x = x
        groupsize = shaped_x.shape[-1]

    qW_min, qW_max = get_qbits_minmax(numbits, is_sym=is_ssz_sym)

    if init_offset is not None and init_scale is not None and init_quant is not None:
        scale, offset, quant = init_scale, init_offset, init_quant
    else:
        scale, offset = get_scale_offset(
            shaped_x, qW_min, qW_max, is_sym=is_ssz_sym, is_act_integer=is_act_integer
        )
        scale = scale.clamp(min=SCALE_CLAMP_MIN)
        quant = get_quant(shaped_x, qW_min, qW_max, scale, offset=offset)

    dequant = get_dequant(quant, qW_min, qW_max, scale, offset=offset)
    bestScale = scale

    if is_ssz_sym:
        offset = 0

    bestOffset = offset
    bestQuant = quant
    bestMse = get_mseloss(
        shaped_x, qW_min, qW_max, scale=scale, offset=offset,
        quant=quant, dequant=dequant
    )

    for i in range(num_step):
        a = quant - offset

        if is_act_integer:
            scale = torch.round(
                torch.sum(a * shaped_x, dim=-1, keepdim=True) /
                torch.sum(a * a, dim=-1, keepdim=True)
            ).clamp(min=1)
        else:
            scale = (
                torch.sum(a * shaped_x, dim=-1, keepdim=True) /
                torch.sum(a * a, dim=-1, keepdim=True)
            ).clamp(min=SCALE_CLAMP_MIN)

        offset = torch.round(
            torch.sum(quant * scale - shaped_x, dim=-1, keepdim=True) /
            groupsize / scale
        )

        if is_ssz_sym:
            offset = 0

        quant = get_quant(shaped_x, qW_min, qW_max, scale, offset=offset)
        dequant = get_dequant(quant, qW_min, qW_max, scale, offset=offset)

        currentMse = get_mseloss(
            shaped_x, qW_min, qW_max, scale=scale, offset=offset,
            quant=quant, dequant=dequant
        )

        mask1 = (bestMse - currentMse) / bestMse.clamp(min=MSE_EPSILON_RELATIVE) < MSE_EPSILON
        mask2 = torch.abs(bestMse - currentMse) < MSE_EPSILON

        if torch.sum(
            torch.logical_and(torch.logical_not(mask1), torch.logical_not(mask2))
        ) == 0:
            break

        mask = (currentMse < bestMse).to(torch.int32)
        bestMse = currentMse * mask + bestMse * (1 - mask)
        bestScale = scale * mask + bestScale * (1 - mask)
        bestOffset = offset * mask + bestOffset * (1 - mask)
        bestQuant = quant * mask + bestQuant * (1 - mask)

    recovered = get_dequant(bestQuant, qW_min, qW_max, bestScale, bestOffset)
    recovered = recovered.view(shape)

    if w8:
        return bestQuant.to(torch.int8), bestScale

    if w4:
        bestQuantInt4 = bestQuant.view(shape).to(torch.int8)
        bestScale = bestScale.view(shape[0], -1).T.to(torch.float32).view(torch.int32)

        bestScaleInt64 = (
            torch.zeros(bestScale.shape, dtype=torch.int64)
            .view(torch.int32)
            .reshape(-1, 2)
        )
        bestScaleInt64[:, 0] = bestScale.reshape(-1)
        bestScaleInt64 = bestScaleInt64.view(torch.int64).reshape(bestScale.shape)

        bias = (8 * recovered.to(torch.float32)).sum(dim=-1)
        return bestQuantInt4, bestScaleInt64, bias

    return recovered


def weight_quant(tensor):
    """Quantize 2D weight tensor to INT8 symmetric per-channel.

    Args:
        tensor: Input torch.Tensor with 2 dimensions.

    Returns:
        Tuple of (quantized_int8_tensor, scale_tensor).
    """
    assert tensor.dim() == 2, "Input tensor must be 2D"

    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    scale = abs_max / INT8_QMAX
    assert scale.shape == (tensor.shape[0], 1)

    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -INT8_QMAX, INT8_QMAX)

    return quantized.to(torch.int8), scale.to(torch.float32)


def pack_4bit(x):
    """Pack int4 weights into bytes.

    Packs 2 int4 values (each 4 bits) into 1 byte.

    Args:
        x: Input tensor with int4 values.

    Returns:
        Packed tensor.
    """
    x = x.T.contiguous()
    shape = x.shape
    x = x.view(-1, 2)

    x1 = x[:, 0]
    x2 = x[:, 1]

    y_x2 = torch.bitwise_left_shift(x2, 4)
    y_x1 = x1 & 0b00001111
    y = torch.bitwise_or(y_x1, y_x2)

    y = y.view(shape[0], shape[1] // 2)
    return y.T.contiguous()


def unpack_from_int32(value, num_bits, shape=None, packed_dim=1):
    """Unpack int32 tensor into individual values.

    Args:
        value: Packed int32 tensor.
        num_bits: Number of bits per value.
        shape: Original shape to unpack to (for removing padding).
        packed_dim: Dimension that was used for packing.

    Returns:
        Unpacked int8 tensor.

    Raises:
        ValueError: If dtype is not int32 or num_bits > 8.
    """
    if value.dtype != torch.int32:
        raise ValueError(
            "Expected %s but got %s, aborting unpack"
            % (torch.int32, value.dtype)
        )

    if num_bits > 8:
        raise ValueError("Unpacking is only supported for less than 8 bits")

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1

    if packed_dim == 1:
        unpacked = torch.zeros(
            (value.shape[0], value.shape[1] * pack_factor),
            device=value.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask

        if shape is not None:
            original_row_size = int(shape[1])
            unpacked = unpacked[:, :original_row_size]
    else:
        unpacked = torch.zeros(
            (value.shape[0] * pack_factor, value.shape[1]),
            device=value.device,
            dtype=torch.int32,
        )
        for i in range(pack_factor):
            unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask

        if shape is not None:
            original_row_size = int(shape[0])
            unpacked = unpacked[:original_row_size, :]

    # Convert unsigned to signed
    offset = pow(2, num_bits) // 2
    unpacked = (unpacked - offset).to(torch.int8)

    return unpacked
