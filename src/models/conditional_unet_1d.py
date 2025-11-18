# src/models/conditional_unet_1d.py

from typing import Union

import torch
import torch.nn as nn

from .utils import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    ConditionalResidualBlock1D,
)


class ConditionalUnet1D(nn.Module):
    """
    1D UNet with FiLM conditioning on (diffusion step embedding + global_cond).

    Args:
        input_dim: action dimension (C_in at input)
        global_cond_dim: dim of global conditioning (e.g., obs_horizon * obs_dim)
        diffusion_step_embed_dim: size of sinusoidal time embedding
        down_dims: channel sizes for each U-Net level
        kernel_size: Conv1d kernel size
        n_groups: GroupNorm group size
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims=None,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()

        if down_dims is None:
            down_dims = [256, 512, 1024]

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # diffusion step encoder
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        # middle blocks
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        # down path
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.down_modules = down_modules

        # up path
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.up_modules = up_modules

        # final conv
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )

        print(
            "number of parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        sample: (B, T, input_dim)
        timestep: (B,) or scalar, diffusion step
        global_cond: (B, global_cond_dim)
        return: (B, T, input_dim)
        """
        # (B, T, C) -> (B, C, T)
        x = sample.moveaxis(-1, -2)

        # handle timesteps
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=x.device
            )
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(x.device)
        timesteps = timesteps.expand(x.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat(
                [global_feature, global_cond], dim=-1
            )

        h = []
        # down path
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # middle
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # up path
        for resnet1, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        # (B, C, T) -> (B, T, C)
        x = x.moveaxis(-1, -2)
        return x


__all__ = ["ConditionalUnet1D"]
