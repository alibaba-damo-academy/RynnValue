import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset_upload.helpers import generate_unique_id


CAMERA_DIR_CANDIDATES = [
    "front_rgb",
    "left_shoulder_rgb",
    "right_shoulder_rgb",
    # "right_shoudler_rgb",  # sometimes misspelled in datasets
    # "wrist_rgb",
]


class RacerFrameListLoader:
    """Pickle-able loader that reads a list of image paths on demand (RGB, uint8)."""

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


def _sorted_pngs(dir_path: Path) -> list[str]:
    paths = [p for p in dir_path.glob("*.png")]
    paths.sort(key=lambda x: int(x.stem.split("_")[0]))
    return [str(p) for p in paths]


def _make_traj(image_paths: list[str], task_text: str, is_success: bool) -> dict[str, Any]:
    traj: dict[str, Any] = {}
    traj["id"] = generate_unique_id()
    traj["task"] = task_text
    traj["frames"] = RacerFrameListLoader(image_paths)
    traj["is_robot"] = True
    traj["quality_label"] = "successful" if is_success else "failure"
    traj["data_source"] = "racer"
    traj["preference_group_id"] = None
    traj["preference_rank"] = None
    return traj


def _collect_camera_views(sample_dir: Path) -> dict[str, list[str]]:
    views: dict[str, list[str]] = {}
    for cam in CAMERA_DIR_CANDIDATES:
        d = sample_dir / cam
        if d.exists() and d.is_dir():
            imgs = _sorted_pngs(d)
            if imgs:
                views[cam] = imgs
    return views


def load_racer_dataset(dataset_path: str, dataset_name: str) -> dict[str, list[dict]]:
    """Load RACER-augmented_rlbench dataset.

    Args:
        dataset_path: Path to dataset root containing 'train' and/or 'val' folders.
        dataset_name: Use to pick split: 'racer_train' -> train, 'racer_val' -> val.

    Behavior:
        - Uses task_goal from language_description.json as language instruction.
        - Creates success trajectories (full expert episode) per camera view.
        - For each expert subgoal frame that contains heuristic failures in 'augmentation',
          creates failure trajectories up to that expert frame index (inclusive), per camera view.

    Returns:
        Mapping: task_goal -> list of trajectory dicts.
    """
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"RACER dataset path not found: {root}")

    split = "val" if ("val" in dataset_name.lower()) else "train"

    # Some distributions include an extra 'samples' segment
    split_dir = root / split
    alt_split_dir = split_dir / "samples"
    if alt_split_dir.exists():
        split_dir = alt_split_dir

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Tasks are subdirectories under split_dir
    task_dirs = [p for p in split_dir.iterdir() if p.is_dir()]

    task_data: dict[str, list[dict]] = {}

    for task_dir in task_dirs:
        # Episodes are numeric directories under each task
        episode_dirs = [p for p in task_dir.iterdir() if p.is_dir()]
        for ep_dir in episode_dirs:
            json_path = ep_dir / "language_description.json"
            if not json_path.exists():
                continue

            try:
                with open(json_path, "r") as f:
                    desc = json.load(f)
            except Exception:
                continue

            task_goal: str = desc.get("task_goal", "").strip() or task_dir.name
            subgoal_dict: dict[str, Any] = desc.get("subgoal", {}) or {}

            # Gather camera views for this episode once
            views = _collect_camera_views(ep_dir)
            if not views:
                continue

            # Success: use full length per view
            for cam, img_list in views.items():
                if not img_list:
                    continue
                expert_img_list = [p for p in img_list if "expert" in p]
                traj = _make_traj(expert_img_list, task_goal, is_success=True)
                task_data.setdefault(task_goal, []).append(traj)

            # Failures: for each expert key that contains heuristic augmentations
            for key, content in subgoal_dict.items():
                # Expect keys like '0_expert', '48_expert', ...
                if not isinstance(key, str) or "expert" not in key:
                    continue
                try:
                    expert_frame_idx = int(key.split("_")[0])
                except Exception:
                    continue

                aug = content.get("augmentation", {}) if isinstance(content, dict) else {}
                if not isinstance(aug, dict) or not aug:
                    continue

                # Enumerate augmentations; select those labeled as heuristic failures
                has_failure = False
                for aug_image_name, aug_content in aug.items():
                    if not isinstance(aug_content, dict):
                        continue
                    label = str(aug_content.get("label", "")).lower()
                    if "failure" in label:  # e.g., 'recoverable_failure'
                        has_failure = True
                        break

                if not has_failure:
                    continue

                # Build failure trajectories by truncating expert frames up to expert_frame_idx
                for cam, img_list in views.items():
                    if not img_list:
                        continue

                    # Find frames with numeric names and truncate accordingly
                    def _frame_num(p: str) -> int:
                        try:
                            return int(Path(p).stem.split("_")[0])
                        except Exception:
                            return 1_000_000_000

                    # Keep frames with index < expert_frame_idx
                    subset = [p for p in img_list if _frame_num(p) < expert_frame_idx and "expert" in p]
                    # add the augmented failure frame
                    for img_name in img_list:
                        if aug_image_name in img_name:
                            subset.append(img_name)
                            break
                    if not subset:
                        continue
                    traj = _make_traj(subset, task_goal, is_success=False)
                    task_data.setdefault(task_goal, []).append(traj)
    return task_data
