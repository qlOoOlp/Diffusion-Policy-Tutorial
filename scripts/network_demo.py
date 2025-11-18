#!/usr/bin/env python3
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
import torch
from models.pusht_network import build_pusht_nets, build_noise_scheduler  # noqa: E402


# -----------------------------------------------------
# Config
# -----------------------------------------------------
OBS_HORIZON = 2
PRED_HORIZON = 16
ACTION_DIM = 2
LOWDIM_OBS_DIM = 2  # agent_pos dim


def main():
    # 1. build nets and scheduler
    nets, obs_dim = build_pusht_nets(
        obs_horizon=OBS_HORIZON,
        action_dim=ACTION_DIM,
        lowdim_obs_dim=LOWDIM_OBS_DIM,
        backbone="resnet18",
        weights=None,
        use_groupnorm=True,
    )

    noise_scheduler = build_noise_scheduler(num_diffusion_iters=100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets.to(device)

    print("[INFO] obs_dim per step:", obs_dim)
    print("[INFO] nets keys:", list(nets.keys()))
    print("[INFO] noise_scheduler:", type(noise_scheduler).__name__)

    # 2. demo forward pass (원래 Network Demo 내용)
    with torch.no_grad():
        # example inputs
        image = torch.zeros((1, OBS_HORIZON, 3, 96, 96), device=device)
        agent_pos = torch.zeros((1, OBS_HORIZON, LOWDIM_OBS_DIM), device=device)

        # vision encoder
        image_features = nets["vision_encoder"](image.flatten(end_dim=1))
        # (B * obs_horizon, 512) -> (B, obs_horizon, 512)
        image_features = image_features.reshape(*image.shape[:2], -1)

        obs = torch.cat([image_features, agent_pos], dim=-1)
        # (B, obs_horizon, obs_dim)

        noised_action = torch.randn(
            (1, PRED_HORIZON, ACTION_DIM), device=device
        )
        diffusion_iter = torch.zeros((1,), device=device, dtype=torch.long)

        noise = nets["noise_pred_net"](
            sample=noised_action,
            timestep=diffusion_iter,
            global_cond=obs.flatten(start_dim=1),
        )

        denoised_action = noised_action - noise

    print("[INFO] image_features.shape:", image_features.shape)
    print("[INFO] obs.shape:", obs.shape)
    print("[INFO] noised_action.shape:", noised_action.shape)
    print("[INFO] noise.shape:", noise.shape)
    print("[INFO] denoised_action.shape:", denoised_action.shape)


if __name__ == "__main__":
    main()
