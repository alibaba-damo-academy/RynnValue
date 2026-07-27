#!/usr/bin/env python3
from __future__ import annotations
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import cv2

from dataset_upload.helpers import generate_unique_id

RGB_CAMERA_PRIORITY = (
    "f1421749_rgb",
    "f1421370_rgb",
    "front_rgb",
    "hand_rgb",
    "rgb",
)

TASK_DESCRIPTION_MAP = {
    "close_drawer": "Close the drawer",
    "fold_towel": "Fold the towel in half",
    "pick_blue_cup": "Put blue cup in sink",
    "put_banana_on_red_plate": "Put the banana on the red plate",
    "put_eggplant_black_bowl": "Put the eggplant in the black bowl",
    "stack_green_on_red": "Stack the green shape on the red cylinder",
}

QUALITY_LABEL_MAP = {
    "success": "successful",
    "successful": "successful",
    "subopt": "suboptimal",
    "suboptimal": "suboptimal",
    "fail": "failure",
    "failure": "failure",
}


class USCXArmFrameLoader:
    """Lazy loader that stitches RGB frames from timestep pickle files."""

    def __init__(self, pickle_paths: Iterable[str], rgb_keys: tuple[str, ...] = RGB_CAMERA_PRIORITY) -> None:
        self.pickle_paths = list(pickle_paths)
        self.rgb_keys = rgb_keys

    def _extract_rgb_frame(self, step_data: dict):
        for key in self.rgb_keys:
            if key in step_data and step_data[key] is not None:
                return step_data[key]
        return None

    def _prepare_frame(self, frame_array: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame_array)
        if frame.ndim != 3:
            raise ValueError(f"Expected 3D RGB frame, got shape {frame.shape}")

        # Convert channel-first -> channel-last if needed
        if frame.shape[0] in (3, 4) and frame.shape[-1] not in (3, 4):
            frame = frame[:3, ...]
            frame = np.transpose(frame, (1, 2, 0))

        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                # Assume [0, 1] range and scale, otherwise clip to [0, 255]
                frame = np.clip(frame, 0.0, 1.0) if frame.max() <= 1.0 else np.clip(frame, 0.0, 255.0)
                scale = 255.0 if frame.max() <= 1.0 else 1.0
                frame = (frame * scale).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8, copy=False)

        # Convert BGR to RGB (frames from pickle files are stored as BGR)
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ensure the array is contiguous for efficient processing
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        return frame

    def __call__(self) -> np.ndarray:
        frames: list[np.ndarray] = []
        for path in self.pickle_paths:
            with open(path, "rb") as f:
                step_data = pickle.load(f)
            frame_array = self._extract_rgb_frame(step_data)
            if frame_array is None:
                continue
            frame = self._prepare_frame(frame_array)
            frames.append(frame)

        if not frames:
            raise ValueError(f"No RGB frames found in {self.pickle_paths[0] if self.pickle_paths else 'unknown path'}")

        return np.stack(frames, axis=0)


def _default_task_description(task_key: str) -> str:
    if task_key in TASK_DESCRIPTION_MAP:
        return TASK_DESCRIPTION_MAP[task_key]
    return task_key.replace("_", " ").capitalize()


def _parse_folder_metadata(folder_name: str) -> tuple[str, str, str]:
    parts = folder_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected folder name format: {folder_name}")
    attempt_id = parts[-1]
    optimality_key = parts[-2].lower()
    task_key = "_".join(parts[:-2])
    return task_key, optimality_key, attempt_id


def load_usc_xarm_policy_ranking_dataset(
    dataset_path: str, max_trajectories: int | None = None
) -> dict[str, list[dict]]:
    """Load USC xArm policy ranking trajectories grouped by language task."""

    root = Path(dataset_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    if (root / "xarm_policy_ranking").exists():
        root = root / "xarm_policy_ranking"

    folders = sorted([path for path in root.iterdir() if path.is_dir()])
    if not folders:
        raise ValueError(f"No task folders found in {root}")

    limit = None if max_trajectories is None or max_trajectories < 0 else int(max_trajectories)
    task_data: dict[str, list[dict]] = defaultdict(list)
    total = 0

    for folder in folders:
        if limit is not None and total >= limit:
            break

        task_key, optimality_key, attempt_id = _parse_folder_metadata(folder.name)
        if optimality_key not in QUALITY_LABEL_MAP:
            raise ValueError(f"Unknown optimality label '{optimality_key}' in folder {folder.name}")

        pickle_paths = sorted(str(p) for p in folder.glob("*.pkl"))
        if not pickle_paths:
            print(f"⚠️  Skipping {folder} (no pickle files found)")
            continue

        frame_loader = USCXArmFrameLoader(pickle_paths)
        task_description = _default_task_description(task_key)

        trajectory = {
            "id": generate_unique_id(),
            "task": task_description,
            "frames": frame_loader(),
            "is_robot": True,
            "quality_label": QUALITY_LABEL_MAP[optimality_key],
            "data_source": "usc_xarm_policy_ranking",
        }

        task_data[task_description].append(trajectory)
        total += 1

    print(f"Loaded {total} trajectories from {len(task_data)} tasks in USC xArm Policy Ranking dataset")
    return task_data
