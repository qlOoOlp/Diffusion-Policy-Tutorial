# Diffusion Policy Tutorial

## Overview  
This repository provides a simple tutorial and demo implementation of Diffusion Policy.
(This tutorial references the original paper and code from [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy).)

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

## Training
### Training your own policy
```bash
python scripts/train_pusht.py
```
### OR download pretrained ckpt
- The pretrained checkpoint used in this tutorial is directly downloaded from the official Diffusion Policy GitHub repository [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy).
```bash
python scripts/load_ckpt.py
```

## Rollout / Demo
### Inference Policy
```bash
python scripts/infer_pusht.py
```
