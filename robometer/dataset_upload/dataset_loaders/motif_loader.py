import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset_upload.helpers import generate_unique_id


class MotifFrameLoader:
    """Pickle-able loader that reads frames for a single trajectory on demand.

    Supports two backing sources:
    - A video file path (e.g., .mp4)
    - A directory of image frames (sorted by filename)
    """

    def __init__(self, source_path: str) -> None:
        self.source_path = source_path

    def _load_from_video(self) -> np.ndarray:
        cap = cv2.VideoCapture(self.source_path)
        frames = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        frames_np = np.asarray(frames)
        if frames_np.ndim != 4 or frames_np.shape[-1] != 3:
            raise ValueError(
                f"Unexpected frames shape from video {self.source_path}: {getattr(frames_np, 'shape', None)}"
            )
        if frames_np.dtype != np.uint8:
            frames_np = frames_np.astype(np.uint8, copy=False)
        return frames_np

    def __call__(self) -> np.ndarray:
        p = Path(self.source_path)
        if p.is_file():
            return self._load_from_video()
        raise FileNotFoundError(f"Source path not found: {self.source_path}")


def _infer_is_robot_from_path(path: Path) -> bool:
    parts = [s.lower() for s in path.parts]
    # MotIF repo mentions 'human_motion' and 'stretch_motion'
    if any("stretch" in s for s in parts):
        return True
    elif any("human" in s for s in parts):
        return False
    else:
        raise ValueError(f"Unknown robot/human: {path}")


def _make_traj(source_path: Path, task_text: str) -> dict:
    traj: dict[str, Any] = {}
    traj["id"] = generate_unique_id()
    traj["task"] = task_text
    traj["frames"] = MotifFrameLoader(str(source_path))
    traj["is_robot"] = _infer_is_robot_from_path(source_path)
    traj["quality_label"] = "successful"
    # traj["partial_success"] = 1
    traj["data_source"] = "motif"
    return traj


def load_motif_dataset(dataset_path: str) -> dict[str, list[dict]]:
    """Load MoTiF dataset using FrameLoader without HF conversion.
    Returns mapping: task -> list of trajectory dicts.
    """
    import json

    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"MoTiF dataset path not found: {root}")

    task_to_trajs: dict[str, list[dict]] = {}

    # Annotations
    ann_dir = root / "annotations"

    all_human_trajs = {}
    path_precursor = "human_motion/videos_raw"
    json_data = json.load(open(ann_dir / "human_motion_data_info.json"))
    for item in json_data:
        src = item["video_path"].split("/")[-1]
        full_vid_path = root / path_precursor / src
        # assert the path exists
        if not full_vid_path.exists():
            print(f"Human video path not found: {full_vid_path}")
            continue
        instruction = item.get("task_instruction") + ": " + item.get("motion_description")
        all_human_trajs.setdefault(instruction, []).append(full_vid_path)

    all_stretch_trajs = {}
    path_precursor = "stretch_motion/videos_raw"
    json_data = json.load(open(ann_dir / "stretch_motion_data_info.json"))
    for item in json_data:
        src = item["video_path"].split("/")[-1]
        full_vid_path = root / path_precursor / src
        # assert the path exists
        if not full_vid_path.exists():
            print(f"Stretch video path not found: {full_vid_path}")
            continue
        instruction = item.get("task_instruction") + ": " + item.get("motion_description")
        all_stretch_trajs.setdefault(instruction, []).append(full_vid_path)

    # get the keys in both
    common_keys = set(all_human_trajs.keys()) & set(all_stretch_trajs.keys())
    all_stretch_trajs = {k: v for k, v in all_stretch_trajs.items() if k in common_keys}
    all_human_trajs = {k: v for k, v in all_human_trajs.items() if k in common_keys}

    print(f"Number of human tasks: {len(all_human_trajs)}")
    print(f"Number of stretch tasks: {len(all_stretch_trajs)}")

    for instruction, paths in all_human_trajs.items():
        for path in paths:
            traj = _make_traj(path, instruction)
            task_to_trajs.setdefault(instruction, []).append(traj)

    for instruction, paths in all_stretch_trajs.items():
        for path in paths:
            traj = _make_traj(path, instruction)
            task_to_trajs.setdefault(instruction, []).append(traj)

    return task_to_trajs
