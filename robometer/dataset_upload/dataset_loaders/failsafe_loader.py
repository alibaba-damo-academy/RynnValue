import glob
import json
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset_upload.helpers import generate_unique_id

TASK_TO_INSTRUCTION = {
    "FailPickCube-v1": "Pick up the red cube",
    "FailPushCube-v1": "Push and move a cube to a goal region in front of it",
    "FailStackCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling",
}


class FailSafeFrameListLoader:
    """Pickle-able loader that reads a list of image paths on demand.

    Returns np.ndarray (T, H, W, 3) uint8.
    """

    def __init__(self, image_paths: list[str]) -> None:
        self.image_paths = image_paths
        assert len(image_paths) > 0

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
    files = [str(p) for p in dir_path.glob("*.png")]

    def _key(s: str) -> tuple:
        name = os.path.splitext(os.path.basename(s))[0]
        try:
            return (int(name),)
        except Exception:
            return (name,)

    files.sort(key=_key)
    return files


def _make_traj(
    image_paths: list[str], task: str, instruction: str, is_success: bool, sub_task: str | None = None
) -> dict[str, Any]:
    traj: dict[str, Any] = {}
    traj["id"] = generate_unique_id()
    # Combine main instruction with optional sub_task for clarity
    if sub_task:
        traj["task"] = sub_task
    else:
        traj["task"] = instruction
    traj["frames"] = FailSafeFrameListLoader(image_paths)
    traj["is_robot"] = True
    traj["quality_label"] = "successful" if is_success else "failure"
    traj["data_source"] = "failsafe"
    traj["preference_group_id"] = None
    traj["preference_rank"] = None
    return traj


def _gather_full_episodes(task_dir: Path, view: str, instruction: str) -> list[dict]:
    episodes: list[dict] = []
    # Seeds are numbered directories directly under task_dir
    for seed_dir in sorted([p for p in task_dir.iterdir() if p.is_dir()]):
        # Ground truth (success)
        gt_view_dir = seed_dir / "Ground_Truth" / view
        if gt_view_dir.exists():
            imgs = _sorted_pngs(gt_view_dir)
            assert len(imgs) > 0
            if imgs:
                episodes.append(_make_traj(imgs, task_dir.name, instruction, is_success=True))

        # Failures: any subfolder except Ground_Truth
        for attempt_dir in sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name != "Ground_Truth"]):
            view_dir = attempt_dir / view
            if view_dir.exists():
                assert len(imgs) > 0
                imgs = _sorted_pngs(view_dir)
                if imgs:
                    episodes.append(_make_traj(imgs, task_dir.name, instruction, is_success=False))
    return episodes


def _gather_sub_episodes_from_json(dataset_root: Path, view: str) -> list[dict]:
    episodes: list[dict] = []
    # JSON files like vla_data_FailPickCube-v1.json, vla_data_GT_PickCube-v1.json etc.
    json_dir = dataset_root / "json_files"
    if not json_dir.exists():
        json_dir = dataset_root  # fallback if jsons are at root

    json_files = glob.glob(str(json_dir / "vla_data_*.json"))
    for jf in sorted(json_files):
        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        # sub sample 1/3 for 3 views
        for entry in random.sample(data, len(data) // 3):
            task_key = entry.get("task")
            instruction = entry.get("instruction") or TASK_TO_INSTRUCTION.get(task_key, task_key or "")
            sub_task = entry.get("sub_task")
            failure_type = entry.get("failure_type", "None")
            # Image list is relative to dataset root
            imgs_rel = entry.get("image", [])
            if not imgs_rel:
                continue
            # Filter by desired view: ensure each path contains "/<view>/"
            if view:
                imgs_rel = [p for p in imgs_rel if f"/{view}/" in p or f"\\{view}\\" in p]
                if len(imgs_rel) == 0:
                    continue
            image_paths = [str((dataset_root / p).resolve()) for p in imgs_rel]
            is_success = (failure_type is None) or (str(failure_type).lower() == "none")
            episodes.append(
                _make_traj(image_paths, task_key or "failsafe", instruction, is_success=is_success, sub_task=sub_task)
            )
    return episodes


def load_failsafe_dataset(dataset_path: str) -> dict[str, list[dict]]:
    """Load FailSafe dataset from local folders and JSON sub-trajectory annotations.

    Args:
        dataset_path: Root directory containing FailPickCube-v1/ FailPushCube-v1/ FailStackCube-v1/ and jsons

    Returns:
        Mapping: instruction string -> list of trajectory dicts
    """
    views = ["front", "side", "wrist"]
    include_sub_trajectories = True
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"FailSafe dataset path not found: {root}")

    task_dirs = [
        p for p in [root / "FailPickCube-v1", root / "FailPushCube-v1", root / "FailStackCube-v1"] if p.exists()
    ]

    task_data: dict[str, list[dict]] = {}

    # Sub-trajectory episodes from JSON
    if include_sub_trajectories:
        for view in views:
            # sample one view
            sub_episodes = _gather_sub_episodes_from_json(root, view=view)
            print(f"Found {len(sub_episodes)} sub-trajectories for {view} after sampling 1/3 of the data")
            for traj in sub_episodes:
                task = traj["task"]
                task_data.setdefault(task, []).append(traj)

    # Full episodes
    for tdir in task_dirs:
        instruction = TASK_TO_INSTRUCTION.get(tdir.name, tdir.name)
        print(f"Gathering full episodes for {instruction}")
        for view in views:
            episodes = _gather_full_episodes(tdir, view=view, instruction=instruction)
            print(f"Found {len(episodes)} episodes for {instruction} {view}")
            if episodes:
                task_data.setdefault(instruction, []).extend(episodes)

    # only keep tasks that have both failed and successful trajectories
    task_data_paired = {}
    for task, trajectories in task_data.items():
        failed_trajectories = [t for t in trajectories if t["quality_label"] == "failure"]
        successful_trajectories = [t for t in trajectories if t["quality_label"] == "successful"]
        if len(failed_trajectories) == 0 or len(successful_trajectories) == 0:
            continue
        task_data_paired[task] = failed_trajectories + successful_trajectories

    print(
        f"Found {len(task_data_paired)} tasks with both failed and successful trajectories from originally {len(task_data)} tasks"
    )

    # print how many failed and successful trajectories there are
    failed_trajectories = [
        sum([1 for t in traj if t["quality_label"] == "failure"]) for traj in task_data_paired.values()
    ]
    successful_trajectories = [
        sum([1 for t in traj if t["quality_label"] == "successful"]) for traj in task_data_paired.values()
    ]
    print(f"Found {sum(failed_trajectories)} failed trajectories")
    print(f"Found {sum(successful_trajectories)} successful trajectories")
    return task_data_paired
