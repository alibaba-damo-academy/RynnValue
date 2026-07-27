#!/usr/bin/env python3
"""
Metaworld dataset loader for the generic dataset converter for Robometer model training.
This module contains logic for loading metaworld data organized by task and quality.

uv run python dataset_upload/generate_hf_dataset.py \
    --config_path=dataset_upload/configs/data_gen_configs/metaworld.yaml
"""

import collections
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataset_upload.video_helpers import load_video_frames
from dataset_upload.dataset_loaders.mw_task_annotations import TRAIN_GT_ANN, EVAL_GT_ANN


def apply_center_crop_to_frames(frames: np.ndarray) -> np.ndarray:
    """Apply center crop (224, 224) to video frames using torchvision transforms.

    Args:
        frames: numpy array of shape (T, H, W, 3) in RGB order

    Returns:
        numpy array of shape (T, 224, 224, 3) with center cropped frames
    """
    # Define the center crop transform
    center_crop = transforms.CenterCrop(224)

    cropped_frames = []
    for frame in frames:
        # Convert numpy array to PIL Image
        pil_frame = Image.fromarray(frame.astype(np.uint8))

        # Apply center crop
        cropped_pil = center_crop(pil_frame)

        # Convert back to numpy array
        cropped_frame = np.array(cropped_pil)
        cropped_frames.append(cropped_frame)

    return np.array(cropped_frames)


def map_quality_label(original_label: str) -> str:
    """Map original quality labels to standardized Robometer labels."""
    label_mapping = {"GT": "successful", "success": "successful", "all_fail": "failure", "close_succ": "suboptimal"}
    return label_mapping.get(original_label, original_label)


def load_metaworld_dataset(base_path: str, dataset_name: str) -> dict[str, list[dict]]:
    """Load metaworld dataset and organize by task.

    Args:
        base_path: Path to the metaworld dataset directory

    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """

    print(f"Loading metaworld dataset from: {base_path}")

    task_data = collections.defaultdict(list)
    base_path = Path(base_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Metaworld dataset path not found: {base_path}")

    if "train" in dataset_name:
        tasks = TRAIN_GT_ANN.keys()
        anns = TRAIN_GT_ANN
    elif "eval" in dataset_name:
        tasks = EVAL_GT_ANN.keys()
        anns = EVAL_GT_ANN

    print("number of tasks: ", len(tasks))

    scucessful_trajs_file = "downloaded_data/metaworld_video.h5"

    with h5py.File(scucessful_trajs_file, "r") as f:
        print("Available tasks: ", f.keys())
        print("number of tasks: ", len(f.keys()))
        task_names = list(f.keys())

        for task_name in tqdm(task_names, desc="Loading successful trajectories"):
            if task_name not in tasks:
                continue

            for traj_name in ["0", "1", "10", "11", "12"]:  # not sure why we use these, but we use these 5
                traj = f[task_name][traj_name]
                frames = np.array(traj)

                cropped_frames = apply_center_crop_to_frames(frames)

                trajectory = {
                    "frames": cropped_frames,
                    "task": anns.get(task_name),
                    "quality_label": "successful",
                    "is_robot": True,
                    "partial_success": 0,
                }
                task_data[task_name].append(trajectory)

    if "eval" in dataset_name:
        task_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        for task_dir in tqdm(task_dirs, desc="Processing tasks"):
            task_name = task_dir.name

            if task_name in [".DS_Store"]:
                continue

            # Find all quality label directories within this task
            quality_dirs = [d for d in task_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

            for quality_dir in quality_dirs:
                original_quality_label = quality_dir.name

                # Map quality label to standardized format
                quality_label = map_quality_label(original_quality_label)

                # Find all video files in this quality directory
                video_files = list(quality_dir.glob("*.mp4")) + list(quality_dir.glob("*.gif"))

                # if len(video_files) != 2:
                #     import ipdb; ipdb.set_trace()

                for video_file in video_files:
                    # # Extract index from filename (e.g., "1.mp4" -> 1)
                    # try:
                    #     int(video_file.stem)
                    # except ValueError:
                    #     print(f"Warning: Could not parse index from filename: {video_file.name}")

                    # Load frames and apply center crop
                    original_frames = load_video_frames(video_file)
                    cropped_frames = apply_center_crop_to_frames(original_frames)

                    nl_name = anns.get(task_name)

                    # Create trajectory entry
                    trajectory = {
                        "frames": cropped_frames,
                        "task": nl_name,  # Natural language task
                        "quality_label": quality_label,  # Mapped quality label
                        "is_robot": True,
                        "partial_success": 0,
                    }

                    task_data[task_name].append(trajectory)

    for task_name, trajectories in task_data.items():
        print(f"Task {task_name}: {len(trajectories)} trajectories")

    total_trajectories = sum(len(trajectories) for trajectories in task_data.values())
    print(f"\nLoaded {total_trajectories} trajectories from {len(task_data)} tasks")

    return task_data
