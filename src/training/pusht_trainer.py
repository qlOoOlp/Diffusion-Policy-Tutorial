# src/training/pusht_trainer.py

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

# 여기서 datasets 모듈의 builder를 불러온다
from datasets.pusht_builder import build_pusht_dataset, build_pusht_dataloader

# models에서 네트워크와 스케줄러 빌더만 사용
from models.pusht_network import build_pusht_nets, build_noise_scheduler


def train_pusht(
    data_zip_path: str,
    device: torch.device,
    num_epochs: int = 100,
    batch_size: int = 64,
    num_workers: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    num_diffusion_iters: int = 100,
    pred_horizon: int = 16,
    obs_horizon: int = 2,
    action_horizon: int = 8,
    ckpt_dir: str = None,
) -> str:
    """
    PushT diffusion policy trainer.
    """

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading dataset: {data_zip_path}")

    # ---------------------------------------------------------
    # 1. Dataset & DataLoader (이미 datasets 모듈에서 제공함)
    # ---------------------------------------------------------
    dataset = build_pusht_dataset(
        data_zip_path=data_zip_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )
    dataloader = build_pusht_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    print(f"[INFO] Loaded dataset: {len(dataset)} samples")

    # ---------------------------------------------------------
    # 2. Networks
    # ---------------------------------------------------------
    nets, obs_dim = build_pusht_nets(
        obs_horizon=obs_horizon,
        action_dim=2,      # PushT action dim = 2
        lowdim_obs_dim=2,  # agent_pos dim = 2
        backbone="resnet18",
        weights=None,
        use_groupnorm=True,
    )
    nets.to(device)

    noise_pred_net = nets["noise_pred_net"]

    # ---------------------------------------------------------
    # 3. Noise scheduler
    # ---------------------------------------------------------
    noise_scheduler = build_noise_scheduler(
        num_diffusion_iters=num_diffusion_iters
    )

    # ---------------------------------------------------------
    # 4. Optimizer, EMA, LR scheduler
    # ---------------------------------------------------------
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75,
    )

    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    num_training_steps = len(dataloader) * num_epochs
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps,
    )

    print(f"[INFO] obs_dim: {obs_dim}")
    print(f"[INFO] Training steps: {num_training_steps}")

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    with tqdm(range(num_epochs), desc="Epoch") as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []

            with tqdm(dataloader, desc="Batch", leave=False) as tepoch:
                for batch in tepoch:
                    nimage = batch["image"][:, :obs_horizon].to(device)
                    nagent_pos = batch["agent_pos"][:, :obs_horizon].to(device)
                    naction = batch["action"].to(device)
                    B = nagent_pos.shape[0]

                    # 1) Vision encoder
                    image_features = nets["vision_encoder"](
                        nimage.flatten(end_dim=1)
                    )
                    image_features = image_features.reshape(
                        *nimage.shape[:2], -1
                    )

                    # 2) Concatenate obs
                    obs_features = torch.cat(
                        [image_features, nagent_pos], dim=-1
                    )
                    obs_cond = obs_features.flatten(start_dim=1)

                    # 3) Forward diffusion
                    noise = torch.randn_like(naction)
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (B,),
                        device=device,
                    ).long()

                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps
                    )

                    # 4) Predict noise
                    noise_pred = noise_pred_net(
                        sample=noisy_actions,
                        timestep=timesteps,
                        global_cond=obs_cond,
                    )

                    # 5) Loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    ema.step(nets.parameters())

                    loss_val = loss.item()
                    epoch_loss.append(loss_val)
                    tepoch.set_postfix(loss=loss_val)

            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # ---------------------------------------------------------
    # 6. EMA → nets로 복사
    # ---------------------------------------------------------
    ema.copy_to(nets.parameters())

    # ---------------------------------------------------------
    # 7. Checkpoint 저장
    # ---------------------------------------------------------
    if ckpt_dir is None:
        ckpt_dir = os.path.join(os.getcwd(), "checkpoints")

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "pusht_dp_ema.ckpt")

    torch.save(
        {
            "nets": nets.state_dict(),
            "noise_scheduler": noise_scheduler.config,
            "obs_horizon": obs_horizon,
            "pred_horizon": pred_horizon,
            "action_horizon": action_horizon,
        },
        ckpt_path,
    )

    print(f"[INFO] Saved checkpoint: {ckpt_path}")

    return ckpt_path
