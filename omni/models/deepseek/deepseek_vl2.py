# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://github.com/deepseek-ai/DeepSeek-VL2/blob/faf18023f24b962b32d9f0a2d89e402a8d383a78/deepseek_vl2/models/modeling_deepseek_vl_v2.py
"""Inference-only Deepseek-VL2 model compatible with HuggingFace weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.transformers_utils.configs.deepseek_vl2 import MlpProjectorConfig


class MlpProjector(nn.Module):
    def __init__(self, cfg: MlpProjectorConfig):
        super().__init__()

        self.cfg = cfg
        self.projector_type = cfg.projector_type
        assert not cfg.token_pooling, "Token pooling is not supported currently."

        if self.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.depth
            mlp_ratio = cfg.mlp_ratio
            modules = [
                nn.Linear(
                    cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio,
                    cfg.n_embed * mlp_ratio,
                )
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio)
                )
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)
        elif self.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)
        else:
            raise NotImplementedError(
                f"Unsupported projector type: {cfg.projector_type}"
            )

        self.layers = modules

    def forward(self, x):
        bs, hw, input_dim = x.shape
        if self.projector_type == "downsample_mlp_gelu":
            h = w = int((hw) ** 0.5)
            """compute padding"""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)
            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(
                x,
                kernel_size=self.cfg.downsample_ratio,
                stride=self.cfg.downsample_ratio,
                padding=0,
            )  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)
