# src/models/utils.py

from typing import Optional
import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,) or (B, 1) like diffusion step index
        return: (B, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 8,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    Residual block with FiLM conditioning:
      - x: (B, C_in, T)
      - cond: (B, cond_dim)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    n_groups=n_groups,
                ),
                Conv1dBlock(
                    out_channels,
                    out_channels,
                    kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        # FiLM modulation: per-channel scale and bias from cond
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1)),  # (B, 2*C) -> (B, 2*C, 1)
        )

        # residual projection if channel size changes
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)  # (B, C_out, T)

        embed = self.cond_encoder(cond)  # (B, 2*C_out, 1)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]  # (B, C_out, 1)
        bias = embed[:, 1, ...]   # (B, C_out, 1)
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


__all__ = [
    "SinusoidalPosEmb",
    "Downsample1d",
    "Upsample1d",
    "Conv1dBlock",
    "ConditionalResidualBlock1D",
]
