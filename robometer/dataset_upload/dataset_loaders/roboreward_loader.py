"""
Loader for RoboReward dataset - a dataset for training vision-language reward models for robotics.

Paper: https://arxiv.org/abs/2601.00675
Dataset: https://huggingface.co/datasets/teetone/RoboReward
"""

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset_upload.helpers import generate_unique_id


class RoboRewardVideoLoader:
    """Pickle-able loader that reads frames from an existing MP4 video file."""

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path

    def __call__(self) -> np.ndarray:
        """Load all frames from the video file."""
        cap = cv2.VideoCapture(self.video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        if not frames:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)

        return np.asarray(frames, dtype=np.uint8)


def _reward_to_partial_success(reward: int) -> float:
    """Convert RoboReward score (1-5) to partial_success (0.0-1.0).

    Reward scale:
        1: No success -> 0.0
        2: Minimal progress -> 0.25
        3: Partial completion -> 0.5
        4: Near completion -> 0.75
        5: Perfect completion -> 1.0
    """
    return (reward - 1) / 4.0


def _make_traj(video_path: str, task: str, reward: int, dataset_name: str) -> dict[str, Any]:
    """Create a trajectory dictionary from RoboReward metadata."""
    partial_success = _reward_to_partial_success(reward)

    traj: dict[str, Any] = {}
    traj["id"] = generate_unique_id()
    traj["task"] = task
    traj["frames"] = RoboRewardVideoLoader(video_path)  # Lazy loader for existing MP4
    traj["is_robot"] = True
    traj["quality_label"] = "successful" if partial_success == 1.0 else "failure"
    traj["partial_success"] = partial_success
    traj["data_source"] = f"roboreward_{dataset_name}"
    traj["preference_group_id"] = None
    traj["preference_rank"] = None
    return traj


def load_roboreward_dataset(dataset_path: str, dataset_name: str) -> dict[str, list[dict]]:
    """Load RoboReward dataset from local folders.

    Args:
        dataset_path: Root directory containing train/, val/, test/ folders
        dataset_name: Dataset name to determine split (e.g., 'roboreward_train', 'roboreward_val', 'roboreward_test')

    Structure:
        dataset_path/
            train/
                metadata.jsonl
                [video folders with MP4s]
            val/
                metadata.jsonl
                [video folders with MP4s]
            test/
                metadata.jsonl
                [video folders with MP4s]

    Returns:
        Mapping: task instruction -> list of trajectory dicts
    """
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"RoboReward dataset path not found: {root}")

    # Determine split from dataset_name
    if "train" in dataset_name.lower():
        split = "train"
    elif "val" in dataset_name.lower():
        split = "val"
    elif "test" in dataset_name.lower():
        split = "test"
    else:
        raise ValueError(f"Dataset name must specify split (train/val/test): {dataset_name}")

    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    metadata_file = split_dir / "metadata.jsonl"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    task_data: dict[str, list[dict]] = {}

    # Read metadata.jsonl
    print(f"Loading RoboReward {split} split from {metadata_file}")
    with open(metadata_file, "r") as f:
        for line_idx, line in enumerate(f):
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_idx} in metadata.jsonl")
                continue

            file_name = entry.get("file_name")
            task = entry.get("task")
            reward = entry.get("reward")

            if not file_name or not task or reward is None:
                print(f"Warning: Missing required fields in line {line_idx}")
                continue

            # Construct full video path
            video_path = split_dir / file_name
            if not video_path.exists():
                print(f"Warning: Video file not found: {video_path}")
                continue

            dataset_name = file_name.split("/")[0]

            if dataset_name == "robo_arena":
                dataset_name = "roboarena"

            # Create trajectory
            traj = _make_traj(str(video_path), task, reward, dataset_name)
            task_data.setdefault(task, []).append(traj)

    print(f"Loaded {len(task_data)} unique tasks from RoboReward {split} split")

    # Print reward distribution
    all_trajs = [t for trajs in task_data.values() for t in trajs]
    reward_counts = {i: 0 for i in range(1, 6)}
    for traj in all_trajs:
        # Reverse conversion to get original reward
        reward = int(traj["partial_success"] * 4 + 1)
        reward_counts[reward] += 1

    print(f"Reward distribution:")
    for reward, count in sorted(reward_counts.items()):
        print(f"  Reward {reward}: {count} trajectories")
    print(f"Total trajectories: {len(all_trajs)}")

    return task_data
