"""
Loader for HAND_paired_data dataset containing paired robot and human demonstrations.
"""

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset_upload.helpers import generate_unique_id


CAMERA_VIEWS = ["external_imgs", "over_shoulder_imgs"]


class HandPairedFrameLoader:
    """Pickle-able loader that reads a list of JPG image paths on demand (RGB, uint8)."""

    def __init__(self, image_paths: list[str]) -> None:
        if not image_paths:
            raise ValueError("image_paths must be non-empty")
        self.image_paths = image_paths

    def __call__(self) -> np.ndarray:
        frames: list[np.ndarray] = []
        for p in self.image_paths:
            img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)
        if not frames:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        frames_np = np.asarray(frames, dtype=np.uint8)
        return frames_np


def _sorted_jpgs(dir_path: Path) -> list[str]:
    """Return sorted list of JPG file paths from a directory."""
    paths = [p for p in dir_path.glob("*.jpg")]

    def _key(p: Path):
        # Extract number from filenames like "im_0.jpg", "im_1.jpg", etc.
        name = p.stem
        try:
            # Handle "im_X" format
            if "_" in name:
                return int(name.split("_")[-1])
            return int(name)
        except Exception:
            return 0

    paths.sort(key=_key)
    return [str(p) for p in paths]


def _parse_task_name(folder_name: str) -> str:
    """Convert folder name to human-readable task instruction.

    Examples:
        blend_carrot -> blend carrot
        close_microwave_hand -> close microwave
    """
    # Remove '_hand' suffix if present
    task = folder_name.replace("_hand", "")
    # Replace underscores with spaces
    task = task.replace("_", " ")
    return task


def _is_human_task(folder_name: str) -> bool:
    """Check if this is a human demonstration task."""
    return folder_name.endswith("_hand")


def _make_traj(image_paths: list[str], task_text: str, is_robot: bool) -> dict[str, Any]:
    """Create a trajectory dictionary."""
    traj: dict[str, Any] = {}
    traj["id"] = generate_unique_id()
    traj["task"] = task_text
    traj["frames"] = HandPairedFrameLoader(image_paths)
    traj["is_robot"] = is_robot
    traj["quality_label"] = "successful"  # Assuming all demonstrations are successful
    traj["data_source"] = "hand_paired"
    traj["preference_group_id"] = None
    traj["preference_rank"] = None
    return traj


def load_hand_paired_dataset(dataset_path: str, dataset_name: str) -> dict[str, list[dict]]:
    """Load HAND_paired_data dataset from local folders.

    Args:
        dataset_path: Root directory containing task folders (e.g., blend_carrot, blend_carrot_hand, etc.)

    Structure:
        dataset_path/
            blend_carrot/
                traj0/
                    external_imgs/
                        im_0.jpg, im_1.jpg, ...
                    over_shoulder_imgs/
                        im_0.jpg, im_1.jpg, ...
                traj1/
                    ...
            blend_carrot_hand/
                traj0/
                    ...
            close_microwave/
                ...
            close_microwave_hand/
                ...

    Returns:
        Mapping: task instruction -> list of trajectory dicts
    """
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"HAND_paired dataset path not found: {root}")

    # Get all task directories
    task_dirs = [p for p in root.iterdir() if p.is_dir()]

    task_data: dict[str, list[dict]] = {}

    dataset_name = dataset_name.replace("hand_paired_", "")

    for task_dir in task_dirs:
        print(f"Processing task: {task_dir.name}")
        task_name = _parse_task_name(task_dir.name)
        is_robot = not _is_human_task(task_dir.name)
        if dataset_name == "robot":
            if not is_robot:
                continue
        elif dataset_name == "human":
            if is_robot:
                continue

        # Get all trajectory directories (traj0, traj1, traj2, etc.)
        traj_dirs = [p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("traj")]
        print(f"Found {len(traj_dirs)} trajectory directories")
        for traj_dir in traj_dirs:
            print(f"Processing trajectory: {traj_dir.name}")
            # Process each camera view
            for camera_view in CAMERA_VIEWS:
                print(f"Processing camera view: {camera_view}")
                camera_dir = traj_dir / camera_view
                if not camera_dir.exists():
                    continue

                # Get sorted list of JPG images
                image_paths = _sorted_jpgs(camera_dir)
                if not image_paths:
                    continue

                # Create trajectory
                traj = _make_traj(image_paths, task_name, is_robot)
                task_data.setdefault(task_name, []).append(traj)

    print(f"Loaded {len(task_data)} unique tasks from HAND paired {dataset_name} dataset")
    for task, trajs in task_data.items():
        robot_count = sum(1 for t in trajs if t["is_robot"])
        human_count = sum(1 for t in trajs if not t["is_robot"])
        print(f"  {task}: {robot_count} robot, {human_count} human trajectories")

    return task_data
