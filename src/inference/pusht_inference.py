# src/utils/pusht_inference.py

import os
import collections
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import imageio

from envs.pusht_env import PushTImageEnv
from datasets import build_pusht_dataset
from datasets.pusht_dataset import normalize_data, unnormalize_data
from models.pusht_network import build_pusht_nets, build_noise_scheduler


def load_nets_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    obs_horizon: int = 2,
    action_dim: int = 2,
    lowdim_obs_dim: int = 2,
):
    """
    ckpt_path에서 네트워크 로드.
    - ckpt 포맷 1: state_dict (ema_nets.state_dict()) 만 저장된 경우
    - ckpt 포맷 2: {"nets": nets.state_dict(), ...} 형태인 경우
    """

    # 네트워크 구성 (노트북 설정과 동일)
    nets, obs_dim = build_pusht_nets(
        obs_horizon=obs_horizon,
        action_dim=action_dim,
        lowdim_obs_dim=lowdim_obs_dim,
        backbone="resnet18",
        weights=None,
        use_groupnorm=True,
    )
    nets.to(device)

    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    # 포맷 2: {"nets": ...}
    if isinstance(state, dict) and "nets" in state:
        nets.load_state_dict(state["nets"])
    else:
        # 포맷 1: 바로 state_dict
        nets.load_state_dict(state)

    print("[INFO] Checkpoint loaded into nets.")
    return nets, obs_dim


def run_pusht_inference(
    ckpt_path: str,
    data_zip_path: str,
    device: Optional[torch.device] = None,
    max_steps: int = 200,
    env_seed: int = 100000,
    output_video_path: str = "vis.mp4",
    pred_horizon: int = 16,
    obs_horizon: int = 2,
    action_horizon: int = 8,
    action_dim: int = 2,
    lowdim_obs_dim: int = 2,
    num_diffusion_iters: int = 100,
):
    """
    PushT Diffusion Policy 인퍼런스를 수행하고 roll-out 비디오를 저장.

    Args:
        ckpt_path: 학습된 또는 pretrained ckpt 경로
        data_zip_path: pusht_cchi_v7_replay.zarr.zip 경로 (stats 로딩용)
        device: torch.device (None이면 자동 선택)
        max_steps: 환경에서 최대 상호작용 스텝 수
        env_seed: env.seed() 값 (>200 으로 train과 다른 초기 상태)
        output_video_path: 저장할 mp4 파일 경로
        pred_horizon, obs_horizon, action_horizon: 노트북과 동일한 하이퍼파라미터
        action_dim: PushT 액션 차원 (2)
        lowdim_obs_dim: agent_pos 차원 (2)
        num_diffusion_iters: DDPM iteration 수
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[ERROR] ckpt not found: {ckpt_path}")

    if not os.path.isfile(data_zip_path):
        raise FileNotFoundError(
            f"[ERROR] dataset zip not found: {data_zip_path}\n"
            "Run `python scripts/load_dataset.py` first."
        )

    # ------------------------------------------------------------------
    # 1. Dataset 로딩해서 stats 가져오기 (agent_pos / action 정규화용)
    # ------------------------------------------------------------------
    print(f"[INFO] Loading dataset stats from: {data_zip_path}")
    dataset = build_pusht_dataset(
        data_zip_path=data_zip_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )
    stats = dataset.stats  # {'agent_pos': {...}, 'action': {...}, ...}

    # ------------------------------------------------------------------
    # 2. 네트워크 + noise scheduler 준비
    # ------------------------------------------------------------------
    nets, obs_dim = load_nets_from_ckpt(
        ckpt_path=ckpt_path,
        device=device,
        obs_horizon=obs_horizon,
        action_dim=action_dim,
        lowdim_obs_dim=lowdim_obs_dim,
    )
    noise_scheduler = build_noise_scheduler(num_diffusion_iters=num_diffusion_iters)

    # ------------------------------------------------------------------
    # 3. 환경 세팅
    # ------------------------------------------------------------------
    env = PushTImageEnv()
    env.seed(env_seed)
    obs, info = env.reset()

    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    imgs = [env.render(mode="rgb_array")]
    rewards = []
    done = False
    step_idx = 0

    print("[INFO] Start inference rollout...")
    from tqdm.auto import tqdm as tq

    with tq(total=max_steps, desc="Eval PushTImageEnv") as pbar:
        while not done:
            B = 1

            # --------------------------------------------------
            # 관측 스택 구성
            # --------------------------------------------------
            images = np.stack([x["image"] for x in obs_deque])        # (obs_horizon, 3, 96, 96)
            agent_poses = np.stack([x["agent_pos"] for x in obs_deque])  # (obs_horizon, 2)

            # 정규화 (agent_pos만)
            nagent_poses = normalize_data(agent_poses, stats=stats["agent_pos"])
            nimages = images  # 이미 [0,1]

            # Tensor 변환 + device
            nimages_t = torch.from_numpy(nimages).to(device, dtype=torch.float32)
            nagent_poses_t = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

            # --------------------------------------------------
            # Diffusion Policy로 action 시퀀스 샘플링
            # --------------------------------------------------
            with torch.no_grad():
                # 1) vision feature 추출
                image_features = nets["vision_encoder"](nimages_t)  # (obs_horizon, 512)

                # 2) 저차원 상태와 concat
                obs_features = torch.cat([image_features, nagent_poses_t], dim=-1)
                # (obs_horizon, obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
                # (B, obs_horizon * obs_dim)

                # 3) action 초기화 (Gaussian noise)
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device
                )
                naction = noisy_action

                # 4) scheduler timesteps 설정
                noise_scheduler.set_timesteps(num_diffusion_iters)

                # 5) reverse diffusion loop
                for k in noise_scheduler.timesteps:
                    noise_pred = nets["noise_pred_net"](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond,
                    )
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction,
                    ).prev_sample

            # --------------------------------------------------
            # 6. unnormalize + action_horizon 만큼만 실행
            # --------------------------------------------------
            naction_np = naction.detach().cpu().numpy()[0]  # (pred_horizon, action_dim)
            action_pred = unnormalize_data(naction_np, stats=stats["action"])

            start = obs_horizon - 1
            end = start + action_horizon
            action_seq = action_pred[start:end, :]  # (action_horizon, action_dim)

            # --------------------------------------------------
            # 7. env에 rollout
            # --------------------------------------------------
            for u in action_seq:
                obs, reward, terminated, truncated, info = env.step(u)
                done = terminated or truncated

                obs_deque.append(obs)
                rewards.append(reward)
                imgs.append(env.render(mode="rgb_array"))

                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=float(reward))

                if step_idx >= max_steps:
                    done = True
                if done:
                    break

    print("Score:", max(rewards) if len(rewards) > 0 else 0.0)

    # ------------------------------------------------------------------
    # 8. 비디오 저장
    # ------------------------------------------------------------------
    imageio.mimwrite(output_video_path, imgs, fps=10)
    print(f"[INFO] Saved rollout video to: {output_video_path}")

    env.close()
