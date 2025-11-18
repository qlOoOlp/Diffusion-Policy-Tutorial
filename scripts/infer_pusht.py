#!/usr/bin/env python3
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
import argparse
import torch
from inference import run_pusht_inference  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script for PushT Diffusion Policy."
    )

    # ckpt: 기본은 data/pusht_vision_100ep.ckpt (공식 pretrained)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "pusht_vision_100ep.ckpt"),
        help="Path to checkpoint (.ckpt or .pt).",
    )

    # dataset zip: stats 로딩용
    parser.add_argument(
        "--data_zip",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "pusht_cchi_v7_replay.zarr.zip"),
        help="Path to pusht_cchi_v7_replay.zarr.zip.",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum number of environment steps.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=100000,
        help="Environment seed (>200 to avoid training initial states).",
    )

    parser.add_argument(
        "--video_out",
        type=str,
        default=None,   # 기본값 None → main()에서 자동 생성
        help="Output video path (mp4). If not provided, a timestamped file will be created in PROJECT_ROOT/rollout/",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # -------------------------------------------------------
    # 1. rollout 디렉토리 준비
    # -------------------------------------------------------
    rollout_dir = os.path.join(PROJECT_ROOT, "rollout")
    if not os.path.exists(rollout_dir):
        print(f"[INFO] Creating rollout directory: {rollout_dir}")
        os.makedirs(rollout_dir, exist_ok=True)

    # -------------------------------------------------------
    # 2. video_out 기본값 처리(현재 시각 기반 자동 생성)
    # -------------------------------------------------------
    if args.video_out is None:
        import datetime

        now = datetime.datetime.now()
        timestamp = now.strftime("%y%m%d_%H%M%S")     # 예: 251118_132413
        filename = f"viz_{timestamp}.mp4"
        video_out = os.path.join(rollout_dir, filename)
    else:
        # 사용자가 경로를 직접 지정한 경우
        video_out = args.video_out

    # -------------------------------------------------------
    # 3. 나머지 실행 출력
    # -------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Project root:", PROJECT_ROOT)
    print("[INFO] CKPT:", args.ckpt)
    print("[INFO] DATA ZIP:", args.data_zip)
    print("[INFO] Video out:", video_out)
    print("[INFO] Device:", device)

    # -------------------------------------------------------
    # 4. 인퍼런스 실행
    # -------------------------------------------------------
    run_pusht_inference(
        ckpt_path=args.ckpt,
        data_zip_path=args.data_zip,
        device=device,
        max_steps=args.max_steps,
        env_seed=args.seed,
        output_video_path=video_out,
    )



if __name__ == "__main__":
    main()
