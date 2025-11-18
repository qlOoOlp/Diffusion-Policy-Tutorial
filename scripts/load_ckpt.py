#!/usr/bin/env python3
import os, sys
# add project_root/src to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
# add project_root/src to sys.path (so "datasets", "envs" can be imported)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
import torch
import gdown

from models.pusht_network import build_pusht_nets              # noqa: E402
from models.pusht_network import build_noise_scheduler         # noqa: E402


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
CKPT_NAME = "pusht_vision_100ep.ckpt"
GDRIVE_ID = "1XKpfNSlwYMGaF5CncoFaLKCDTWoLAHf1&confirm=t"

CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, CKPT_NAME)

# 노트북과 동일한 설정
OBS_HORIZON = 2
ACTION_DIM = 2         # 2D position action
LOWDIM_OBS_DIM = 2     # agent_pos dim


def ensure_ckpt_dir():
    if not os.path.exists(CKPT_DIR):
        print(f"[INFO] Creating data directory: {CKPT_DIR}")
        os.makedirs(CKPT_DIR, exist_ok=True)


def download_checkpoint():
    """Pretrained ckpt 없으면 Google Drive에서 다운로드."""
    if os.path.isfile(CKPT_PATH):
        print(f"[INFO] Checkpoint already exists: {CKPT_PATH}")
        return

    print("[INFO] Downloading pretrained checkpoint from Google Drive...")
    gdown.download(id=GDRIVE_ID, output=CKPT_PATH, quiet=False)
    print(f"[INFO] Download complete: {CKPT_PATH}")


def load_checkpoint():
    """네트워크를 만들고, pretrained ckpt를 로드."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---------------------------------------------------------
    # 1) 노트북과 동일한 설정으로 네트워크 구성
    # ---------------------------------------------------------
    nets, obs_dim = build_pusht_nets(
        obs_horizon=OBS_HORIZON,
        action_dim=ACTION_DIM,
        lowdim_obs_dim=LOWDIM_OBS_DIM,
        backbone="resnet18",
        weights=None,
        use_groupnorm=True,
    )
    nets.to(device)
    print(f"[INFO] Built nets with obs_dim={obs_dim}")

    # ---------------------------------------------------------
    # 2) 체크포인트 로드
    #    노트북에서는:
    #      state_dict = torch.load(ckpt_path, map_location='cuda')
    #      ema_nets = nets
    #      ema_nets.load_state_dict(state_dict)
    #    이 패턴이었으니 여기서도 동일하게 처리
    # ---------------------------------------------------------
    print(f"[INFO] Loading checkpoint state_dict from: {CKPT_PATH}")
    state_dict = torch.load(CKPT_PATH, map_location=device)

    # ckpt가 nets.state_dict() 그대로 저장된 형태라고 가정
    nets.load_state_dict(state_dict)
    print("[INFO] Pretrained weights loaded into nets.")

    # 간단한 구조 출력
    print("[INFO] Nets modules:", list(nets.keys()))
    print("  - vision_encoder:", nets["vision_encoder"].__class__.__name__)
    print("  - noise_pred_net:", nets["noise_pred_net"].__class__.__name__)

    return nets


def main():
    ensure_ckpt_dir()
    download_checkpoint()
    nets = load_checkpoint()
    print("[INFO] Done. You can now reuse `nets` for inference / rollout scripts.")


if __name__ == "__main__":
    main()