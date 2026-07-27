import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from dataset_upload.helpers import create_hf_trajectory, load_sentence_transformer_model

MAX_EPISODES_PER_DATASET = 5  # 5 episodes per dataset; any extra is a mistake


def _require_lerobot_dataset_class():
    """
    Import LeRobotDataset lazily so this module can be imported without lerobot installed.

    Upstream: https://github.com/huggingface/lerobot
    """

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception as e:  # pragma: no cover - depends on local environment
        raise ImportError(
            "USC Koch P-Ranking robot demos require Hugging Face LeRobot. Install it in your env, e.g.\n"
            "  pip install lerobot\n"
            "Then re-run dataset conversion."
        ) from e
    return LeRobotDataset


def _load_lerobot_dataset(local_dataset_dir: Path):
    """Load a local LeRobot dataset directory (downloaded to disk)."""
    LeRobotDataset = _require_lerobot_dataset_class()

    constructors: list[Callable[[], Any]] = [
        lambda: LeRobotDataset(str(local_dataset_dir)),
        lambda: LeRobotDataset(repo_id=str(local_dataset_dir)),
        lambda: LeRobotDataset(path=str(local_dataset_dir)),
        lambda: LeRobotDataset(dataset_path=str(local_dataset_dir)),
    ]

    last_err: Exception | None = None
    for ctor in constructors:
        try:
            ds = ctor()
            _ = len(ds)  # force __len__
            return ds
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            break

    raise RuntimeError(
        f"Failed to construct LeRobotDataset from local path: {local_dataset_dir}\nLast error: {last_err}"
    )


def _infer_quality_from_dataset_dirname(dirname: str) -> str:
    name = dirname.lower()
    if "suboptimal" in name:
        return "suboptimal"
    if "failure" in name:
        return "failure"
    return "successful"


def _read_task_from_tasks_parquet(tasks_path: Path) -> str | None:
    tasks_df = pd.read_parquet(tasks_path)
    if len(tasks_df) == 0:
        return None
    # Most of these store the task as the index.
    if tasks_df.index is not None and len(tasks_df.index) > 0 and isinstance(tasks_df.index[0], str):
        return str(tasks_df.index[0])

    # Fallback: scan first row for a string value.
    first_row = tasks_df.iloc[0].to_dict()
    for v in first_row.values():
        if isinstance(v, str) and v.strip():
            return v
    return None


def _build_video_paths(output_dir: str, dataset_label: str, episode_idx: int, view: str) -> tuple[str, str]:
    shard_index = episode_idx // 1000
    shard_dir = f"shard_{shard_index:04d}"
    episode_dir = os.path.join(output_dir, dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}")
    os.makedirs(episode_dir, exist_ok=True)
    filename = f"{view}.mp4"
    full_path = os.path.join(episode_dir, filename)
    rel_path = os.path.join(dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}", filename)
    return full_path, rel_path


def _to_rgb_uint8_hwc(frame: Any) -> np.ndarray:
    """
    Convert a single frame to RGB uint8 HWC.
    Expected LeRobot image tensors are often float in [0,1] shaped (C,H,W).
    """
    if hasattr(frame, "detach") and hasattr(frame, "cpu") and hasattr(frame, "numpy"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)

    # CHW -> HWC
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255).astype(np.uint8)

    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)

    return frame


def _extract_episode_frames_from_lerobot(
    lerobot_dataset: Any,
    episode_idx: int,
    preferred_camera: str = "observation.images.top",
) -> list[np.ndarray]:
    """
    Extract full episode frames using LeRobot's episode metadata boundaries.
    """
    episode_meta = lerobot_dataset.meta.episodes[episode_idx]
    ep_start_idx = int(episode_meta["dataset_from_index"])
    ep_end_idx = int(episode_meta["dataset_to_index"])

    frames: list[np.ndarray] = []
    for frame_idx in range(ep_start_idx, ep_end_idx):
        item = lerobot_dataset[frame_idx]
        if preferred_camera not in item:
            raise KeyError(
                f"Preferred camera '{preferred_camera}' not found in dataset item keys: {sorted(item.keys())}"
            )
        frames.append(_to_rgb_uint8_hwc(item[preferred_camera]))
    return frames


def convert_usc_koch_p_ranking_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
    max_episodes_per_dataset: int = 10,
    preferred_camera: str = "observation.images.top",
) -> Dataset:
    """
    Convert USC Koch datasets (robot-only) in LeRobot format to HF format for reward training.

    Folder layout expected (see download script):
      <dataset_path>/robot/usc_koch_*_{task}/
        meta/info.json, meta/tasks.parquet, meta/episodes/... (LeRobot)
        videos/observation.images.top/... (LeRobot)
    """
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    robot_dir = root / "robot"
    if not robot_dir.exists():
        raise FileNotFoundError(f"Robot datasets directory not found: {robot_dir}")

    if num_workers == -1:
        num_workers = min(cpu_count(), 8)
    elif num_workers == 0:
        num_workers = 1

    # Load sentence transformer model once; cache task embeddings
    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    robot_dataset_dirs = [d for d in robot_dir.iterdir() if d.is_dir() and (d / "meta" / "info.json").exists()]
    robot_dataset_dirs = sorted(robot_dataset_dirs, key=lambda p: p.name)

    all_entries: list[dict[str, Any]] = []
    global_episode_idx = 0

    print("Processing USC Koch P-Ranking robot demonstrations via LeRobotDataset...")
    print(f"Found {len(robot_dataset_dirs)} robot dataset directories")

    for robot_dataset_path in robot_dataset_dirs:
        quality = _infer_quality_from_dataset_dirname(robot_dataset_path.name)

        tasks_path = robot_dataset_path / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            print(f"Warning: tasks.parquet not found in {robot_dataset_path}, skipping")
            continue

        task_instruction = _read_task_from_tasks_parquet(tasks_path)
        if not task_instruction:
            print(f"Warning: No task found in {robot_dataset_path}, skipping")
            continue

        if task_instruction not in lang_cache:
            lang_cache[task_instruction] = lang_model.encode(task_instruction)

        # Load LeRobot dataset
        lerobot_ds = _load_lerobot_dataset(robot_dataset_path)
        total_episodes = min(len(lerobot_ds.meta.episodes), MAX_EPISODES_PER_DATASET)

        print(f"  {robot_dataset_path.name}: quality={quality} | episodes={total_episodes}")

        for ep_idx in range(total_episodes):
            if max_trajectories is not None and max_trajectories > 0 and len(all_entries) >= max_trajectories:
                break

            try:
                frames = _extract_episode_frames_from_lerobot(
                    lerobot_dataset=lerobot_ds,
                    episode_idx=ep_idx,
                    preferred_camera=preferred_camera,
                )
                if not frames:
                    continue

                full_path, rel_path = _build_video_paths(output_dir, dataset_name, global_episode_idx, "robot")
                traj_dict = {
                    "id": str(uuid4()),
                    "frames": frames,
                    "task": task_instruction,
                    "is_robot": True,
                    "quality_label": quality,
                    "preference_group_id": None,
                    "preference_rank": None,
                }
                entry = create_hf_trajectory(
                    traj_dict=traj_dict,
                    video_path=full_path,
                    lang_vector=lang_cache[task_instruction],
                    max_frames=max_frames,
                    dataset_name=dataset_name,
                    use_video=True,
                    fps=fps,
                )
                if entry:
                    entry["frames"] = rel_path
                    all_entries.append(entry)
                    global_episode_idx += 1
            except Exception as e:
                print(f"Error processing {robot_dataset_path.name} episode {ep_idx}: {e}")
                continue

        if max_trajectories is not None and max_trajectories > 0 and len(all_entries) >= max_trajectories:
            break

    print(f"Total entries: {len(all_entries)}")
    print(f"Unique instructions: {len(lang_cache)}")

    if not all_entries:
        return Dataset.from_dict({
            "id": [],
            "task": [],
            "lang_vector": [],
            "data_source": [],
            "frames": [],
            "is_robot": [],
            "quality_label": [],
            "preference_group_id": [],
            "preference_rank": [],
        })

    return Dataset.from_list(all_entries)
