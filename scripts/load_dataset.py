#!/usr/bin/env python3
import os, sys
# add project_root/src to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
# add project_root/src to sys.path (so "datasets", "envs" can be imported)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
import zipfile

import torch
import gdown

from datasets import PushTImageDataset  # noqa: E402 (import after sys.path setup)

# data directory inside project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Dataset paths
DATA_ZIP = os.path.join(DATA_DIR, "pusht_cchi_v7_replay.zarr.zip")
DATA_ZARR = os.path.join(DATA_DIR, "pusht_cchi_v7_replay.zarr")

# Google Drive ID
GDRIVE_ID = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"

# Dataset config
PRED_HORIZON = 16
OBS_HORIZON = 2
ACTION_HORIZON = 8


# -----------------------------------------------------
# Download dataset
# -----------------------------------------------------
def download_dataset():
    """Download dataset zip into project_root/data/ if missing."""
    if os.path.isfile(DATA_ZIP):
        print(f"[INFO] Dataset zip already exists: {DATA_ZIP}")
        return

    print("[INFO] Downloading dataset from Google Drive...")
    gdown.download(id=GDRIVE_ID, output=DATA_ZIP, quiet=False)
    print("[INFO] Download complete.")


# -----------------------------------------------------
# Load dataset
# -----------------------------------------------------
def load_dataset():
    """Create dataset object using ZIP (.zarr.zip) path."""
    dataset = PushTImageDataset(
        dataset_path=DATA_ZIP,  # zip 파일 경로를 그대로 사용
        pred_horizon=PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_HORIZON,
    )

    print("[INFO] Dataset loaded.")
    print(f"[INFO] Number of samples: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataset, dataloader


# -----------------------------------------------------
# Visualization
# -----------------------------------------------------
def visualize_batch(dataloader):
    """Print shapes of batch data for sanity check."""
    batch = next(iter(dataloader))
    print("batch['image'].shape      :", batch["image"].shape)
    print("batch['agent_pos'].shape  :", batch["agent_pos"].shape)
    print("batch['action'].shape     :", batch["action"].shape)


# -----------------------------------------------------
# Main
# -----------------------------------------------------
if __name__ == "__main__":
    print(f"[INFO] Project root:   {PROJECT_ROOT}")
    print(f"[INFO] Data directory: {DATA_DIR}")

    download_dataset()

    dataset, dataloader = load_dataset()

    print("[INFO] Normalization stats:")
    for key, stat in dataset.stats.items():
        print(f"  {key} min={stat['min']}, max={stat['max']}")

    visualize_batch(dataloader)