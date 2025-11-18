import os
import torch
from .pusht_dataset import PushTImageDataset

def build_pusht_dataset(
    data_zip_path: str,
    pred_horizon: int,
    obs_horizon: int,
    action_horizon: int,
):
    if not os.path.isfile(data_zip_path):
        raise FileNotFoundError(...)
    dataset = PushTImageDataset(
        dataset_path=data_zip_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )
    return dataset

def build_pusht_dataloader(dataset, batch_size, num_workers, device):
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    return dl