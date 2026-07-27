import json
import os
from difflib import SequenceMatcher
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from dataset_upload.helpers import (
    create_hf_trajectory,
    generate_unique_id,
    load_sentence_transformer_model,
)

MAX_EPISODES = 10  # at most 10 episodes per task, some have more but it's a mistake


def _load_video_frames(video_path: str) -> list[np.ndarray]:
    """Load frames from an MP4 video file."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames = []
    cap = cv2.VideoCapture(video_path)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        cap.release()

    return frames


def _string_similarity(s1: str, s2: str) -> float:
    """Calculate similarity between two strings (0-1)."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def _find_best_match(query: str, options: list[str], threshold: float = 0.6) -> str | None:
    """Find the best matching string from options, or None if below threshold."""
    best_match = None
    best_score = threshold

    for option in options:
        score = _string_similarity(query, option)
        if score > best_score:
            best_score = score
            best_match = option

    return best_match


def _require_lerobot_dataset_class():
    """
    Import LeRobotDataset lazily so this module can be imported without lerobot installed.

    Upstream: https://github.com/huggingface/lerobot
    """

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception as e:  # pragma: no cover - depends on local environment
        raise ImportError(
            "Robot demos require Hugging Face LeRobot. Install it in your env, e.g.\n"
            "  pip install lerobot\n"
            "Then re-run dataset conversion."
        ) from e
    return LeRobotDataset


def _load_lerobot_dataset(local_dataset_dir: Path):
    """Load a local LeRobot dataset directory (downloaded to disk)."""

    LeRobotDataset = _require_lerobot_dataset_class()

    # LeRobot's API has evolved; try a few common constructor styles.
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


def _build_video_paths(output_dir: str, dataset_label: str, episode_idx: int, view: str) -> tuple[str, str]:
    """Build output video paths with shard structure."""
    shard_index = episode_idx // 1000
    shard_dir = f"shard_{shard_index:04d}"
    episode_dir = os.path.join(output_dir, dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}")
    os.makedirs(episode_dir, exist_ok=True)
    filename = f"{view}.mp4"
    full_path = os.path.join(episode_dir, filename)
    rel_path = os.path.join(dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}", filename)
    return full_path, rel_path


def _process_human_episode(args):
    """Process a single human episode."""
    video_path, json_path, episode_idx, lang_vec, output_dir, dataset_label, max_frames, fps = args

    # Load video frames
    try:
        frames = _load_video_frames(video_path)
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None

    if not frames:
        return None

    # Load metadata
    with open(json_path, "r") as f:
        metadata = json.load(f)

    instruction = metadata.get("notes", "")
    if not instruction:
        return None

    # Build output video path
    full_path, rel_path = _build_video_paths(output_dir, dataset_label, episode_idx, "human")

    traj_dict = {
        "id": generate_unique_id(),
        "frames": frames,
        "task": instruction,
        "is_robot": False,  # Human demonstration
        "quality_label": "successful",  # All human demos are successful
        "preference_group_id": None,
        "preference_rank": None,
    }

    entry = create_hf_trajectory(
        traj_dict=traj_dict,
        video_path=full_path,
        lang_vector=lang_vec,
        max_frames=max_frames,
        dataset_name=dataset_label,
        use_video=True,
        fps=fps,
    )

    if entry:
        entry["frames"] = rel_path
        return entry
    return None


def _process_robot_episode_from_lerobot(
    lerobot_dataset: Any,
    episode_idx: int,
    task_instruction: str,
    lang_vec: Any,
    output_dir: str,
    dataset_label: str,
    max_frames: int,
    fps: int,
    global_episode_idx: int,
    preferred_camera: str = "observation.images.top",
):
    """Process a single robot episode from a loaded LeRobotDataset."""
    try:
        episode_meta = lerobot_dataset.meta.episodes[episode_idx]
        ep_start_idx = episode_meta["dataset_from_index"]
        ep_end_idx = episode_meta["dataset_to_index"]
        frames = []
        for frame_idx in range(ep_start_idx, ep_end_idx):
            frames.append(
                (lerobot_dataset[frame_idx][preferred_camera].numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            )

        assert len(frames) > 0, (
            f"No frames found for episode {episode_idx} with preferred camera {preferred_camera} and instruction {task_instruction}"
        )

        full_path, rel_path = _build_video_paths(output_dir, dataset_label, global_episode_idx, "robot")
        traj_dict = {
            "id": generate_unique_id(),
            "frames": frames,
            "task": task_instruction,
            "is_robot": True,
            "quality_label": "successful",
            "preference_group_id": None,
            "preference_rank": None,
        }

        entry = create_hf_trajectory(
            traj_dict=traj_dict,
            video_path=full_path,
            lang_vector=lang_vec,
            max_frames=max_frames,
            dataset_name=dataset_label,
            use_video=True,
            fps=fps,
        )

        if entry:
            entry["frames"] = rel_path
            return entry
        return None

    except Exception as e:
        print(f"Error processing robot episode {episode_idx}: {e}")
        return None


def convert_usc_koch_human_robot_paired_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    trajectory_type: str = "human",  # "human", "robot"
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
) -> Dataset:
    """Convert USC Koch Human-Robot Paired dataset to HF format.

    Args:
        dataset_path: Path to the dataset directory containing human/ and robot/ folders
        dataset_name: Name for the dataset
        output_dir: Output directory for processed videos
        trajectory_type: Type of trajectories to include ("human", "robot")
        max_trajectories: Maximum number of trajectories to process per type (None for all)
        max_frames: Maximum frames per trajectory
        fps: Frames per second for output videos
        num_workers: Number of worker processes (-1 for auto, 0 for sequential)

    Returns:
        HuggingFace Dataset
    """
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    human_dir = root / "human" / "recordings"
    robot_dir = root / "robot"

    if not human_dir.exists():
        raise FileNotFoundError(f"Human recordings directory not found: {human_dir}")

    if not robot_dir.exists():
        raise FileNotFoundError(f"Robot datasets directory not found: {robot_dir}")

    if num_workers == -1:
        num_workers = min(cpu_count(), 8)
    elif num_workers == 0:
        num_workers = 1

    # Load sentence transformer model
    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    all_entries = []
    global_episode_idx = 0

    # Process human demonstrations
    if trajectory_type in ["human"]:
        print("Processing human demonstrations...")
        human_videos = sorted(human_dir.glob("*.mp4"))

        if max_trajectories is not None and max_trajectories > 0:
            human_videos = human_videos[:max_trajectories]

        # Pre-compute language embeddings
        human_tasks = []
        for video_path in tqdm(human_videos, desc="Loading human metadata"):
            json_path = video_path.with_suffix(".json")
            if not json_path.exists():
                continue

            with open(json_path, "r") as f:
                metadata = json.load(f)
            instruction = metadata.get("notes", "")
            if instruction:
                human_tasks.append(instruction)
                if instruction not in lang_cache:
                    lang_cache[instruction] = lang_model.encode(instruction)

        print(f"Found {len(human_videos)} human videos with {len(set(human_tasks))} unique tasks")

        # Process human episodes
        human_entries = []
        if num_workers == 1:
            for video_path in tqdm(human_videos, desc="Processing human videos"):
                json_path = video_path.with_suffix(".json")
                if not json_path.exists():
                    continue

                with open(json_path, "r") as f:
                    metadata = json.load(f)
                instruction = metadata.get("notes", "")
                if not instruction:
                    continue

                result = _process_human_episode((
                    str(video_path),
                    str(json_path),
                    global_episode_idx,
                    lang_cache[instruction],
                    output_dir,
                    dataset_name,
                    max_frames,
                    fps,
                ))
                if result:
                    human_entries.append(result)
                    global_episode_idx += 1
        else:
            from multiprocessing import Pool

            args_list = []
            for video_path in human_videos:
                json_path = video_path.with_suffix(".json")
                if not json_path.exists():
                    continue

                with open(json_path, "r") as f:
                    metadata = json.load(f)
                instruction = metadata.get("notes", "")
                if not instruction:
                    continue

                args_list.append((
                    str(video_path),
                    str(json_path),
                    global_episode_idx,
                    lang_cache[instruction],
                    output_dir,
                    dataset_name,
                    max_frames,
                    fps,
                ))
                global_episode_idx += 1

            with Pool(processes=num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(_process_human_episode, args_list),
                        total=len(args_list),
                        desc="Processing human videos",
                    )
                )

            human_entries = [r for r in results if r is not None]

        all_entries.extend(human_entries)
        print(f"Successfully processed {len(human_entries)} human demonstrations")

    # Process robot demonstrations
    if trajectory_type in ["robot"]:
        print("Processing robot demonstrations via LeRobotDataset (top view)...")

        # Find all robot dataset directories that do not hae "suboptimal" or "failure" in the name
        robot_datasets = [
            d
            for d in robot_dir.iterdir()
            if d.is_dir()
            and (d / "meta" / "info.json").exists()
            and "suboptimal" not in d.name
            and "failure" not in d.name
        ]

        print(f"Found {len(robot_datasets)} robot datasets")

        robot_entries = []
        for robot_dataset_path in robot_datasets:
            # Load task information
            tasks_path = robot_dataset_path / "meta" / "tasks.parquet"
            if not tasks_path.exists():
                print(f"Warning: tasks.parquet not found in {robot_dataset_path}, skipping")
                continue

            tasks_df = pd.read_parquet(tasks_path)
            # tasks.parquet sometimes stores the task as index, sometimes as a column
            task_name = None
            if len(tasks_df) > 0:
                if tasks_df.index is not None and len(tasks_df.index) > 0 and isinstance(tasks_df.index[0], str):
                    task_name = tasks_df.index[0]
                else:
                    first_row = tasks_df.iloc[0].to_dict()
                    for v in first_row.values():
                        if isinstance(v, str) and v.strip():
                            task_name = v
                            break

            if not task_name:
                print(f"Warning: No task found in {robot_dataset_path}, skipping")
                continue

            # Pre-compute language embedding
            if task_name not in lang_cache:
                lang_cache[task_name] = lang_model.encode(task_name)

            # Load LeRobot dataset from disk once per task dataset.
            lerobot_ds = _load_lerobot_dataset(robot_dataset_path)
            total_episodes = min(MAX_EPISODES, len(lerobot_ds.meta.episodes))

            print(f"  Processing {total_episodes} episodes from: {robot_dataset_path.name}")

            # Process each episode
            for ep_idx in range(total_episodes):
                result = _process_robot_episode_from_lerobot(
                    lerobot_dataset=lerobot_ds,
                    episode_idx=ep_idx,
                    task_instruction=task_name,
                    lang_vec=lang_cache[task_name],
                    output_dir=output_dir,
                    dataset_label=dataset_name,
                    max_frames=max_frames,
                    fps=fps,
                    global_episode_idx=global_episode_idx,
                    preferred_camera="observation.images.top",
                )
                if result:
                    robot_entries.append(result)
                    global_episode_idx += 1

        all_entries.extend(robot_entries)
        print(f"Successfully processed {len(robot_entries)} robot demonstrations")

    print(f"Total entries: {len(all_entries)}")
    print(f"Unique instructions: {len(lang_cache)}")

    # Create HuggingFace dataset
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
