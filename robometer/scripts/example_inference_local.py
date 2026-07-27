#!/usr/bin/env python3
"""
Run RBM inference locally: load a checkpoint from HuggingFace and compute per-frame progress
and success for a video (or .npy/.npz frames) and task instruction. Writes rewards .npy,
success-probs .npy, and a progress/success plot. Requires the robometer package.

Example:
  python scripts/example_inference_local.py \\
    --model-path aliangdw/qwen4b_pref_prog_succ_8_frames_all_part2 \\
    --video /path/to/video.mp4 \\
    --task "Pick up the red block and place it in the bin"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.evals.eval_viz_utils import create_combined_progress_success_plot, extract_frames
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


def load_frames_input(
    video_or_array_path: str,
    *,
    fps: float = 1.0,
    max_frames: int = 512,
) -> np.ndarray:
    """Load frames from a video path/URL or .npy/.npz file. Returns uint8 (T, H, W, C)."""
    if video_or_array_path.endswith(".npy"):
        frames_array = np.load(video_or_array_path)
    elif video_or_array_path.endswith(".npz"):
        with np.load(video_or_array_path, allow_pickle=False) as npz:
            if "frames" in npz:
                frames_array = npz["frames"].copy()
            elif "arr_0" in npz:
                frames_array = npz["arr_0"].copy()
            else:
                frames_array = next(iter(npz.values())).copy()
    else:
        frames_array = extract_frames(video_or_array_path, fps=fps, max_frames=max_frames)
        if frames_array is None or frames_array.size == 0:
            raise RuntimeError("Could not extract frames from video.")

    if frames_array.dtype != np.uint8:
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)
    if frames_array.ndim == 4 and frames_array.shape[1] in (1, 3) and frames_array.shape[-1] not in (1, 3):
        frames_array = frames_array.transpose(0, 2, 3, 1)
    return frames_array


def compute_rewards_per_frame_local(
    model_path: str,
    video_frames: np.ndarray,
    task: str,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load RBM from HuggingFace and run inference; return per-frame progress and success arrays."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=model_path,
        device=device,
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    T = int(video_frames.shape[0])
    traj = Trajectory(
        frames=video_frames,
        frames_shape=tuple(video_frames.shape),
        task=task,
        id="0",
        metadata={"subsequence_length": T},
        video_embeddings=None,
    )
    progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = batch_collator([progress_sample])

    progress_inputs = batch["progress_inputs"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(device)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    results = compute_batch_outputs(
        reward_model,
        tokenizer,
        progress_inputs,
        sample_type="progress",
        is_discrete_mode=is_discrete,
        num_bins=num_bins,
    )

    progress_pred = results.get("progress_pred", [])
    progress_array = (
        np.array(progress_pred[0], dtype=np.float32)
        if progress_pred and len(progress_pred) > 0
        else np.array([], dtype=np.float32)
    )

    outputs_success = results.get("outputs_success", {})
    success_probs = outputs_success.get("success_probs", []) if outputs_success else []
    success_array = (
        np.array(success_probs[0], dtype=np.float32)
        if success_probs and len(success_probs) > 0
        else np.array([], dtype=np.float32)
    )

    return progress_array, success_array


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RBM inference locally: load model from HuggingFace and compute per-frame progress and success.",
        epilog="Outputs: <out>.npy (rewards), <out>_success_probs.npy, <out>_progress_success.png",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="HuggingFace model id or local checkpoint path")
    parser.add_argument("--video", required=True, help="Video path/URL or .npy/.npz with frames (T,H,W,C)")
    parser.add_argument("--task", required=True, help="Task instruction for the trajectory")
    parser.add_argument("--fps", type=float, default=1.0, help="FPS when sampling from video (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=512, help="Max frames to extract from video (default: 512)")
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.5,
        help="Threshold for binary success in plot (default: 0.5)",
    )
    parser.add_argument("--out", default=None, help="Output path for rewards .npy (default: <video_stem>_rewards.npy)")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out) if args.out is not None else video_path.with_name(video_path.stem + "_rewards.npy")

    frames = load_frames_input(
        str(args.video),
        fps=float(args.fps),
        max_frames=int(args.max_frames),
    )

    rewards, success_probs = compute_rewards_per_frame_local(
        model_path=args.model_path,
        video_frames=frames,
        task=args.task,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), rewards)
    success_path = out_path.with_name(out_path.stem + "_success_probs.npy")
    np.save(str(success_path), success_probs)

    show_success = success_probs.size > 0 and success_probs.size == rewards.size
    success_binary = (success_probs > float(args.success_threshold)).astype(np.int32) if show_success else None
    fig = create_combined_progress_success_plot(
        progress_pred=rewards,
        num_frames=int(frames.shape[0]),
        success_binary=success_binary,
        success_probs=success_probs if show_success else None,
        success_labels=None,
        title=f"Progress/Success — {video_path.name}",
    )
    plot_path = out_path.with_name(out_path.stem + "_progress_success.png")
    fig.savefig(str(plot_path), dpi=200)
    plt.close(fig)

    summary = {
        "video": str(video_path),
        "num_frames": int(frames.shape[0]),
        "model_path": args.model_path,
        "out_rewards": str(out_path),
        "out_success_probs": str(success_path),
        "out_plot": str(plot_path),
        "reward_min": float(np.min(rewards)) if rewards.size else None,
        "reward_max": float(np.max(rewards)) if rewards.size else None,
        "reward_mean": float(np.mean(rewards)) if rewards.size else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
