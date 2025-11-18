# src/models/pusht_network.py

from typing import Tuple

import torch
import torch.nn as nn
from diffusers import DDPMScheduler

from .vision import get_resnet, replace_bn_with_gn
from .conditional_unet_1d import ConditionalUnet1D


def build_vision_encoder(
    backbone: str = "resnet18",
    weights=None,
    use_groupnorm: bool = True,
) -> Tuple[nn.Module, int]:
    """
    Create vision encoder (ResNet backbone) and return (encoder, feature_dim).

    backbone:   "resnet18", "resnet34", "resnet50", ...
    weights:    torchvision weights (or None)
    use_groupnorm:
        True이면 BatchNorm2d → GroupNorm으로 교체 (EMA와 궁합 위해).
    """
    vision_encoder = get_resnet(backbone, weights=weights)

    if use_groupnorm:
        vision_encoder = replace_bn_with_gn(vision_encoder)

    # ResNet18/34: 512, ResNet50+: 2048
    if backbone in ["resnet18", "resnet34"]:
        vision_feature_dim = 512
    else:
        vision_feature_dim = 2048

    return vision_encoder, vision_feature_dim


def build_pusht_nets(
    obs_horizon: int,
    action_dim: int = 2,
    lowdim_obs_dim: int = 2,
    backbone: str = "resnet18",
    weights=None,
    use_groupnorm: bool = True,
) -> Tuple[nn.ModuleDict, int]:
    """
    Construct the full network dict for PushT:

        nets = {
            "vision_encoder": ResNet backbone,
            "noise_pred_net": ConditionalUnet1D
        }

    Args:
        obs_horizon: 관측 시퀀스 길이 (예: 2)
        action_dim:  액션 차원 (Push-T에서 2)
        lowdim_obs_dim: 저차원 관측 차원 (agent_pos = 2)
        backbone:  비전 백본 이름 ("resnet18" 등)
        weights:   ResNet 가중치 (None or torchvision weights)
        use_groupnorm: BN → GN 변환 여부

    Returns:
        nets: nn.ModuleDict
        obs_dim: 단일 타임스텝의 관측 feature 차원 (vision + lowdim)
    """
    vision_encoder, vision_feature_dim = build_vision_encoder(
        backbone=backbone,
        weights=weights,
        use_groupnorm=use_groupnorm,
    )

    obs_dim = vision_feature_dim + lowdim_obs_dim

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
    )

    nets = nn.ModuleDict(
        {
            "vision_encoder": vision_encoder,
            "noise_pred_net": noise_pred_net,
        }
    )

    return nets, obs_dim


def build_noise_scheduler(
    num_diffusion_iters: int = 100,
) -> DDPMScheduler:
    """
    Construct the DDPM noise scheduler used in the demo.
    """
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",  # network predicts noise
    )
    return noise_scheduler


def move_nets_to_device(
    nets: nn.ModuleDict,
    device: torch.device,
) -> nn.ModuleDict:
    """
    Simple helper to move nets to a specific device.
    """
    return nets.to(device)


__all__ = [
    "build_vision_encoder",
    "build_pusht_nets",
    "build_noise_scheduler",
    "move_nets_to_device",
]
