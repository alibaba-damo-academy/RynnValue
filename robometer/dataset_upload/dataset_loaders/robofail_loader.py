#!/usr/bin/env python3
"""
RoboFail dataset loader for the generic dataset converter for Robometer model training.
This module contains RoboFail-specific logic for loading and processing data files.
"""

from pathlib import Path

import numpy as np
from dataset_upload.helpers import generate_unique_id
from robometer.data.video_helpers import load_video_frames
from tqdm import tqdm


class RoboFailFrameLoader:
    """Pickle-able loader that reads RoboFail frames from files on demand.

    Stores only simple fields so it can be safely passed across processes.
    Supports both HDF5 and video file formats.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def __call__(self) -> np.ndarray:
        """Load frames from file when called.

        Returns:
            np.ndarray of shape (T, H, W, 3), dtype uint8
        """
        return load_video_frames(Path(self.file_path))


# Task mapping from folder names to task descriptions
FOLDER_TO_TASK_NAME = {
    "putAppleBowl1": "put apple in bowl",
    "putAppleBowl2": "put apple in bowl",
    "putAppleBowl3": "put apple in bowl",
    "boilWater1": "boil water in a pot",
    "boilWater2": "boil water in a pot",
    "boilWater3": "boil water in a pot",
    "boilWater4": "boil water in a pot",
    "cutCarrot1": "cut carrot",
    "cutCarrot2": "cut carrot",
    "putPearDrawer1": "put pear in drawer",
    "putPearDrawer2": "put pear in drawer",
    "putPearDrawer3": "put pear in drawer",
    "sauteeCarrot1": "cook carrot",
    "sauteeCarrot2": "cook carrot",
    "sauteeCarrot3": "cook carrot",
    "sauteeCarrot4": "cook carrot",
    "secureObjects1": "secure objects",
    "putFruitsBowl1": "put all visible fruits in bowl",
    "putFruitsBowl2": "put all fruits in bowl",
    "makeCoffee1": "make coffee",
    "makeCoffee2": "make coffee",
    "makeCoffee3": "make coffee",
    "heatPot1": "pre-heat pot",
    "heatPot2": "pre-heat pot",
    "appleInFridge1": "store apple in a bowl and put in a fridge",
    "appleInFridge2": "store apple in a bowl and put in a fridge",
    "appleInFridge3": "store apple in a bowl and put in a fridge",
    "appleInFridge4": "store apple in a bowl and put in a fridge",
    "heatPotato1": "serve a bowl of potato on table that was heated using a microwave",
    "heatPotato2": "serve a bowl of potato on table that was heated using a microwave",
}


def _get_task_name_from_folder(folder_name: str) -> str:
    """Convert folder name to task name using the mapping."""
    # First try to find exact match
    if folder_name in FOLDER_TO_TASK_NAME:
        return FOLDER_TO_TASK_NAME[folder_name]

    # If no exact match, try partial matching
    for folder_key, task_name in FOLDER_TO_TASK_NAME.items():
        if folder_key in folder_name or folder_name in folder_key:
            return task_name

    # If no mapping found, convert folder name to readable task
    task = folder_name.replace("_", " ").replace("-", " ")
    return task.strip()


def _discover_robofail_files(dataset_path: Path) -> list[tuple[Path, str]]:
    """Discover all video files in the RoboFail dataset structure.

    Expected structure:
    robofail/real_data/
        folder_name_1/
            videos/
                color.mp4
        folder_name_2/
            videos/
                color.mp4
        ...

    Returns:
        List of tuples: (video_file_path, task_name)
    """
    trajectory_files: list[tuple[Path, str]] = []

    # Look for folders under robofail/real_data
    real_data_path = dataset_path / "real_data"
    if not real_data_path.exists():
        print(f"Warning: real_data directory not found at {real_data_path}")
        return trajectory_files

    for folder in real_data_path.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name
        task_name = _get_task_name_from_folder(folder_name)

        # Look for videos/color.mp4
        video_path = folder / "videos" / "color.mp4"
        if video_path.exists():
            trajectory_files.append((video_path, task_name))
        else:
            print(f"Warning: No color.mp4 found in {folder}/videos/")

    return trajectory_files


def load_robofail_dataset(dataset_path: str, max_trajectories: int | None = None) -> dict[str, list[dict]]:
    """Load RoboFail dataset and organize by task.

    Args:
        dataset_path: Path to the RoboFail dataset directory (should contain real_data/)
        max_trajectories: Maximum number of trajectories to load (None for all)

    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """

    print(f"Loading RoboFail dataset from: {dataset_path}")
    print("=" * 100)
    print("LOADING ROBOFAIL DATASET")
    print("=" * 100)

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"RoboFail dataset path not found: {dataset_path}")

    traj_files = _discover_robofail_files(dataset_path)
    print(f"Found {len(traj_files)} trajectory files")

    task_data: dict[str, list[dict]] = {}
    loaded_count = 0

    for video_file, task_name in tqdm(traj_files, desc="Processing RoboFail trajectories"):
        if max_trajectories is not None and loaded_count >= max_trajectories and max_trajectories != -1:
            break

        try:
            # Set up frame loader for video file
            frame_loader = RoboFailFrameLoader(
                file_path=str(video_file),
            )

            # For now, we don't have action data from the video files
            # This could be extended if action data is stored elsewhere
            actions = None

            # Use task name from folder mapping
            task_description = task_name

            # Create trajectory info
            trajectory = {
                "frames": frame_loader,
                "actions": actions,
                "is_robot": True,  # RoboFail is typically robot data
                "quality_label": "failure",  # Default assumption
                "preference_group_id": None,
                "preference_rank": None,
                "task": task_description,
                "id": generate_unique_id(),
            }

            task_data.setdefault(task_name, []).append(trajectory)
            loaded_count += 1

        except Exception as e:
            print(f"Error loading trajectory {video_file}: {e}")
            continue

    total_trajectories = sum(len(v) for v in task_data.values())
    print(f"Loaded {total_trajectories} trajectories from {len(task_data)} tasks")

    return task_data
