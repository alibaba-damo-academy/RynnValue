"""
EgoDex dataset loader for the generic dataset converter for Robometer model training.

This module provides a simple, readable loader inspired by the LIBERO loader:
- Discovers (HDF5, MP4) pairs per task directory
- Lazily loads frames via a small frame-loader callable
- Extracts a task description and pose actions from HDF5
- Returns a dictionary mapping task names to lists of trajectory dicts
"""

import os
from pathlib import Path
from re import A

import h5py
import numpy as np
from dataset_upload.helpers import generate_unique_id
from dataset_upload.video_helpers import load_video_frames
from tqdm import tqdm


class EgoDexFrameLoader:
    """Pickle-able frame loader for EgoDex videos."""

    def __init__(self, mp4_path: str):
        self.mp4_path = mp4_path

    def __call__(self) -> np.ndarray:
        """Load frames from the MP4 file when called."""
        return load_video_frames(Path(self.mp4_path), max_frames=1800)  # 30hz * 60s = 1800 frames


def _discover_trajectory_files(dataset_path: Path) -> list[tuple[Path, Path, str]]:
    """Discover all (HDF5, MP4) pairs grouped by task directory."""
    trajectory_files: list[tuple[Path, Path, str]] = []
    for task_dir in dataset_path.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        for hdf5_file in task_dir.glob("*.hdf5"):
            mp4_file = hdf5_file.with_suffix(".mp4")
            if mp4_file.exists():
                trajectory_files.append((hdf5_file, mp4_file, task_name))
            else:
                print(f"Warning: Missing MP4 file for {hdf5_file}")
    return trajectory_files


def _load_hdf5_data(hdf5_path: Path) -> tuple[np.ndarray, str]:
    """Load pose data and task description from EgoDex HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        task_description = ""
        try:
            if "llm_description" in f.attrs:
                if "which_llm_description" in f.attrs:
                    which_desc = f.attrs["which_llm_description"]
                    if int(which_desc) == 2 and "llm_description2" in f.attrs:
                        desc = f.attrs["llm_description2"]
                    else:
                        desc = f.attrs["llm_description"]
                else:
                    desc = f.attrs["llm_description"]
                if isinstance(desc, bytes):
                    task_description = desc.decode("utf-8")
                else:
                    task_description = str(desc)
        except Exception as e:
            print(f"Warning: Could not load task description from {hdf5_path}: {e}")
        pose_data = _extract_pose_actions(f)
    return pose_data, task_description


def _extract_pose_actions(hdf5_file) -> np.ndarray:
    """Extract pose actions (positions) from EgoDex HDF5."""
    actions: list[np.ndarray] = []
    pose_keys = [
        "transforms/leftHand",
        "transforms/rightHand",
        "transforms/leftIndexFingerTip",
        "transforms/rightIndexFingerTip",
        "transforms/camera",
    ]
    for key in pose_keys:
        if key in hdf5_file:
            transform_data = hdf5_file[key][:]  # (N, 4, 4)
            positions = transform_data[:, :3, 3]
            actions.append(positions)
    if not actions:
        if "transforms/camera" in hdf5_file:
            camera_transforms = hdf5_file["transforms/camera"][:]
            camera_positions = camera_transforms[:, :3, 3]
            actions.append(camera_positions)
        else:
            print("Warning: No pose data found, creating dummy actions")
            actions.append(np.zeros((100, 3)))
    return np.concatenate(actions, axis=1)


def load_egodex_dataset(dataset_path: str, max_trajectories: int = 100) -> dict[str, list[dict]]:
    """Load EgoDex dataset and organize by task, without a separate iterator class."""
    print(f"Loading EgoDex dataset from: {dataset_path}")
    print("=" * 100)
    print("LOADING EGODEX DATASET")
    print("=" * 100)

    dataset_path = Path(os.path.expanduser(dataset_path))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    traj_files = _discover_trajectory_files(dataset_path)
    print(f"Found {len(traj_files)} trajectory pairs")

    task_data: dict[str, list[dict]] = {}
    loaded_count = 0

    for hdf5_path, mp4_path, task_name in tqdm(traj_files, desc="Processing trajectories"):
        if max_trajectories is not None and loaded_count >= max_trajectories and max_trajectories != -1:
            break
        pose_data, task_description = _load_hdf5_data(hdf5_path)

        if "description unavailable" in task_description.lower():
            print(f"Skipping task {hdf5_path} because description is: {task_description}")
            continue
        frame_loader = EgoDexFrameLoader(str(mp4_path))

        assert task_description is not None

        trajectory = {
            "frames": frame_loader,
            # "actions": pose_data,
            "is_robot": False,
            "task": task_description,
            "quality_label": "successful",
            "preference_group_id": None,
            "preference_rank": None,
            "task_name": task_name,
            "id": generate_unique_id(),
        }

        task_data.setdefault(task_name, []).append(trajectory)
        loaded_count += 1

    total_trajectories = sum(len(v) for v in task_data.values())
    print(f"Loaded {total_trajectories} trajectories from {len(task_data)} tasks")

    return task_data
