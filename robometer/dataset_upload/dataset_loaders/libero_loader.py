#!/usr/bin/env python3
"""
LIBERO dataset loader for the generic dataset converter for Robometer model training.
This module contains LIBERO-specific logic for loading and processing HDF5 files.
"""

import os
from pathlib import Path

import h5py
import numpy as np
from dataset_upload.helpers import generate_unique_id
from tqdm import tqdm


class LiberoFrameLoader:
    """Pickle-able loader that reads LIBERO frames from an HDF5 dataset on demand.

    Stores only simple fields so it can be safely passed across processes.
    """

    def __init__(self, hdf5_path: str, dataset_path: str, rotate_180: bool = True):
        self.hdf5_path = hdf5_path
        self.dataset_path = dataset_path  # e.g., "data/<trajectory_key>/obs/agentview_rgb"
        self.rotate_180 = rotate_180

    def __call__(self) -> np.ndarray:
        """Load frames from HDF5 when called.

        Returns:
            np.ndarray of shape (T, H, W, 3), dtype uint8
        """
        with h5py.File(self.hdf5_path, "r") as f:
            if self.dataset_path not in f:
                raise KeyError(f"Dataset path '{self.dataset_path}' not found in {self.hdf5_path}")

            frames = f[self.dataset_path][:]

        # Ensure shape and dtype sanity
        if not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"Unexpected frames shape for {self.dataset_path} in {self.hdf5_path}: {getattr(frames, 'shape', None)}"
            )

        # Match existing behavior: flip vertically (previous code called this 180-degree rotate)
        if self.rotate_180:
            frames = frames[:, ::-1, :, :].copy()

        # Ensure uint8
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8, copy=False)

        return frames


def load_libero_dataset(base_path: str) -> dict[str, list[dict]]:
    """Load LIBERO dataset from HDF5 files and organize by task.

    Args:
        base_path: Path to the LIBERO dataset directory containing HDF5 files

    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """

    print(f"Loading LIBERO dataset from: {base_path}")

    task_data = {}

    # Find all HDF5 files in the base path
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"LIBERO dataset path not found: {base_path}")

    hdf5_files = list(base_path.glob("*.hdf5"))
    print("=" * 100)
    print("LOADING LIBERO DATASET")
    print("=" * 100)

    print(f"Found {len(hdf5_files)} HDF5 files")

    for file_path in tqdm(hdf5_files, desc=f"Processing LIBERO dataset, {len(hdf5_files)} files"):
        task_name = file_path.stem  # Remove .hdf5 extension
        # print(f"Loading task: {task_name}")

        with h5py.File(file_path, "r") as f:
            if "data" not in f:
                print(f"No 'data' group in {task_name}")
                continue

            data_group = f["data"]
            trajectories = []

            for trajectory_key in data_group.keys():
                trajectory = data_group[trajectory_key]
                if isinstance(trajectory, h5py.Group):
                    # Extract trajectory data
                    trajectory_info = {"frames": [], "actions": []}

                    # Set up lazy frame loader to avoid loading frames into memory up front
                    if "obs" in trajectory and "agentview_rgb" in trajectory["obs"]:
                        dataset_path = f"data/{trajectory_key}/obs/agentview_rgb"
                        trajectory_info["frames"] = LiberoFrameLoader(
                            hdf5_path=str(file_path),
                            dataset_path=dataset_path,
                            rotate_180=True,
                        )

                    # Get actions if available
                    if "actions" in trajectory:
                        trajectory_info["actions"] = trajectory["actions"][:]

                    # Core attributes
                    trajectory_info["is_robot"] = True
                    trajectory_info["quality_label"] = "successful"
                    trajectory_info["preference_group_id"] = None
                    trajectory_info["preference_rank"] = None

                    # Parse the original file path to extract scene and task info
                    file_name = os.path.basename(file_path).replace(".hdf5", "")

                    # Extract scene and task from the file name
                    # Example: LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray
                    parts = file_name.split("_")

                    # Find the scene part (contains "SCENE")
                    scene_part = None
                    task_parts = []

                    for i, part in enumerate(parts):
                        if "SCENE" in part:
                            scene_part = part
                            # Everything after the scene is the task
                            task_parts = parts[i + 1 :]
                            break

                    # If no scene found, then don't use a scene
                    if scene_part is None:
                        scene_part = "UNKNOWN_SCENE"
                        task_parts = parts

                    # Convert task parts to readable string
                    task_string = " ".join(task_parts).replace("_", " ")
                    task_string = task_string.replace("demo", "")

                    # Add parsed information to trajectory
                    trajectory_info["task"] = task_string.strip()
                    # Assign unique UUID id
                    trajectory_info["id"] = generate_unique_id()
                    trajectories.append(trajectory_info)

            task_data[task_name] = trajectories
            # print(f"  Loaded {len(trajectories)} trajectories for {task_name}")

    print(
        f"Loaded {sum(len(trajectories) for trajectories in task_data.values())} trajectories from {len(task_data)} tasks"
    )
    return task_data
