#!/usr/bin/env python3
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
import numpy as np
from envs import PushTImageEnv  # noqa: E402


def main():
    # --------------------------------------------------------------
    # 0. Create environment
    # --------------------------------------------------------------
    env = PushTImageEnv()
    print("[INFO] Environment created:", env)

    # --------------------------------------------------------------
    # 1. Seed environment
    # --------------------------------------------------------------
    env.seed(1000)
    print("[INFO] Environment seeded with 1000")

    # --------------------------------------------------------------
    # 2. Reset environment
    # --------------------------------------------------------------
    obs, info = env.reset()
    print("[INFO] Environment reset")

    # --------------------------------------------------------------
    # 3. Sample action
    # --------------------------------------------------------------
    action = env.action_space.sample()

    # --------------------------------------------------------------
    # 4. Step
    # --------------------------------------------------------------
    obs, reward, terminated, truncated, info = env.step(action)

    # --------------------------------------------------------------
    # 5. Print shapes and info
    # --------------------------------------------------------------
    with np.printoptions(precision=4, suppress=True, threshold=5):
        print("\n=== Env Demo Output ===")
        print("obs['image'].shape :", obs["image"].shape, "float32, [0,1]")
        print("obs['agent_pos'].shape :", obs["agent_pos"].shape, "float32, [0,512]")
        print("action.shape :", action.shape, "float32, [0,512]")
        print("reward:", reward)
        print("terminated:", terminated, "truncated:", truncated)
        print("info keys:", list(info.keys()))


if __name__ == "__main__":
    main()
