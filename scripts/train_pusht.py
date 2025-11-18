#!/usr/bin/env python3
import argparse
import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from training import train_pusht  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train PushT Diffusion Policy.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_diffusion_iters", type=int, default=100)
    parser.add_argument("--pred_horizon", type=int, default=16)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--action_horizon", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = os.path.join(PROJECT_ROOT, "data", "pusht_cchi_v7_replay.zarr.zip")
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pusht(
        data_zip_path=data_path,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_diffusion_iters=args.num_diffusion_iters,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon,
        ckpt_dir=ckpt_dir,
    )


if __name__ == "__main__":
    main()
