# Diffusion Policy Tutorial

## Overview  
This repository provides a simple tutorial and demo implementation of Diffusion Policy.
(This tutorial references the original paper and code from [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy).)

---
## Lectures
### Lecture videos
- [Lecture1](https://youtu.be/tHYzpUsa0nU)
- [Lecture2](https://youtu.be/ClgAxyvYDQc)
- [Lecture3](https://youtu.be/O3mPsjAPtQI)

### Lecture Notes
- [Lecture1](https://docs.google.com/presentation/d/1Gqo6hmNFLDertuDNCNTsSKS5B5bLqyRZ/edit?usp=sharing&ouid=114520448462165300564&rtpof=true&sd=true)
- [Lecture2](https://docs.google.com/presentation/d/1jleqnQkOyYzhnW599SQsgwHl1DAfcFEk/edit?usp=sharing&ouid=114520448462165300564&rtpof=true&sd=true)
- [Lecture3](https://docs.google.com/presentation/d/1Qf2XDt_D4p5Bg2_KjenvsRiedE6_NCrJ/edit?usp=sharing&ouid=114520448462165300564&rtpof=true&sd=true)
---

## Project Structure
A brief overview of the directory layout used in this tutorial:
```bash
Diffusion-Policy-Tutorial/
│
├── src/
│   ├── envs/                        # PushT environment implementation
│   │   ├── pusht_env.py             # Gym-style PushTImageEnv
│   │   └── utils.py                 # Rendering helpers (DrawOptions, etc.)
│   │
│   ├── datasets/                    # Dataset loading, normalization, builders
│   │   ├── pusht_dataset.py         # PushTImageDataset (zarr → tensor dataset)
│   │   ├── pusht_builder.py         # Dataset & DataLoader builder functions
│   │   └── utils.py                 # Normalization & indexing utilities
│   │
│   ├── models/                      # Vision encoder + Conditional UNet1D + builders
│   │   ├── vision.py                # ResNet encoder + BN→GN conversion utilities
│   │   ├── conditional_unet_1d.py   # 1D Conditional U-Net (noise prediction model)
│   │   ├── pusht_network.py         # High-level builders that assemble full nets
│   │   └── utils.py                 # Misc model-related helpers
│   │
│   ├── inference/                   # Inference utilities (env rollout logic)
│   │   └── pusht_inference.py       # run_pusht_inference() implementation
│   │
│   └── training/                    # Full training loop for diffusion policy
│       └── pusht_trainer.py         # train_pusht() + EMA, scheduler, loss logic
│
├── scripts/
│   ├── load_dataset.py              # Download PushT dataset to /data
│   ├── env_demo.py                  # Quick test: env reset/step + observation sanity check
│   ├── network_demo.py              # Quick test: vision encoder + UNet forward pass
│   ├── load_ckpt.py                 # Download pretrained DP checkpoint to /checkpoints
│   ├── train_pusht.py               # Launch training (CLI entrypoint)
│   └── infer_pusht.py               # Run policy rollout → saves video to /rollout
│
├── data/                            # Downloaded dataset (.zarr) and zip files (auto-created)
├── checkpoints/                     # Saved training/pretrained ckpts (auto-created)
└── rollout/                         # Rollout videos (auto-created by infer_pusht.py)

```

---

## Environment Setup  
```bash
git clone https://github.com/qlOoOlp/Diffusion-Policy-Tutorial.git
cd Diffusion-Policy-Tutorial

conda create -y -n dp_tutorial python=3.9
conda activate dp_tutorial

# Install torch (CUDA 버전에 맞게)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
pip install -r requirements.txt
```

---

## Load Datasets
```bash
python scripts/load_dataset.py
```
This script will:
- download the PushT replay dataset from Google Drive, and
- extract it under `./data/` as `pusht_cchi_v7_replay.zarr.zip`.
- The original .zarr dataset is the same as the one used in the official Diffusion Policy repository.

---

## Optional: Demo Scripts
Before training, you can run simple demo scripts to verify that the environment and the neural network are correctly configured.
### 1. Environment Demo
```bash
python scripts/env_demo.py
```
This script:
- creates the PushTImageEnv
- samples actions and steps the environment
- prints shapes of observations (image, agent_pos) and actions
- renders one frame to ensure the environment works properly

---

### 2. Network Deno
```bash
python scripts/network_demo.py
```
This script:
- constructs the ResNet18 vision encoder
- builds the Conditional UNet (noise prediction network)
- runs a forward pass using dummy observations & actions
- initializes a DDPMScheduler
- It is useful for confirming that the model architecture is functioning before training.

---

## Training
### Training your own policy
```bash
python scripts/train_pusht.py
```
This script trains a diffusion policy on the PushT image dataset using:
- a ResNet18 vision encoder, and
- a 1D Conditional U-Net as the noise prediction network.

By default:
- training is run for 100 epochs,
- Exponential Moving Average (EMA) is applied to the model parameters, and
- checkpoints are saved under `./checkpoints/`.
### OR download pretrained ckpt
- The pretrained checkpoint used in this tutorial is directly downloaded from the official Diffusion Policy GitHub repository.
- Checkpoint is saved under `./checkpoints/` as `pusht_vision_100ep.ckpt`.
```bash
python scripts/load_ckpt.py
```

---

## Rollout
### Inference Policy
```bash
python scripts/infer_pusht.py
```
This script:
- loads the trained (or pretrained) diffusion policy from a checkpoint,
- runs a rollout in the PushTImageEnv for up to 200 environment steps, and
- normalizes / unnormalizes observations and actions using the dataset statistics.
- Rollout videos are automatically saved under: `./rollout/viz_YYMMDD_HHMMSS.mp4`

Requirements:
- dataset zip from `scripts/load_dataset.py` (`./data/pusht_cchi_v7_replay.zarr.zip`)
- checkpoint from either:
    - `scripts/train_pusht.py` (your own model), or
    - `scripts/load_ckpt.py` (official pretrained model)
