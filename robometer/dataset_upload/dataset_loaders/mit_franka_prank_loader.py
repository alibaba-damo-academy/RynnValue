import json
import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from datasets import Dataset
from tqdm import tqdm

from dataset_upload.helpers import (
    create_hf_trajectory,
    generate_unique_id,
    load_sentence_transformer_model,
)


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


def _build_video_paths(output_dir: str, dataset_label: str, episode_idx: int, task_name: str) -> tuple[str, str]:
    """Build output video paths with shard structure."""
    shard_index = episode_idx // 1000
    shard_dir = f"shard_{shard_index:04d}"
    episode_dir = os.path.join(output_dir, dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}")
    os.makedirs(episode_dir, exist_ok=True)
    filename = f"{task_name}.mp4"
    full_path = os.path.join(episode_dir, filename)
    rel_path = os.path.join(dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}", filename)
    return full_path, rel_path


def _process_episode(args):
    """Process a single episode."""
    episode_meta, video_dir, lang_vec, output_dir, dataset_label, max_frames, fps = args

    # Load video frames
    video_path = os.path.join(video_dir, episode_meta["filename"])
    try:
        frames = _load_video_frames(video_path)
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None

    if not frames:
        return None

    # Normalize quality label
    quality_label = episode_meta["success"]
    if quality_label == "success":
        quality_label = "successful"
    elif quality_label == "fail":
        quality_label = "failure"
    # "suboptimal" stays as is

    # Build output video path
    full_path, rel_path = _build_video_paths(
        output_dir, dataset_label, episode_meta["episode_idx"], episode_meta["task_name"]
    )

    traj_dict = {
        "id": generate_unique_id(),
        "frames": frames,
        "task": episode_meta["instruction"],
        "is_robot": True,
        "quality_label": quality_label,
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


def convert_mit_franka_prank_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
) -> Dataset:
    """Convert MIT-Franka-Prank dataset to HF format.

    Args:
        dataset_path: Path to the dataset directory containing metadata files and videos
        dataset_name: Name for the dataset
        output_dir: Output directory for processed videos
        max_trajectories: Maximum number of trajectories to process (None for all)
        max_frames: Maximum frames per trajectory
        fps: Frames per second for output videos
        num_workers: Number of worker processes (-1 for auto, 0 for sequential)

    Returns:
        HuggingFace Dataset
    """
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    # Find the subdirectory with videos (20251210)
    video_dir = None
    for subdir in root.iterdir():
        if subdir.is_dir() and (subdir / "foldtowel_metadata.json").exists():
            video_dir = subdir
            break

    if video_dir is None:
        raise FileNotFoundError(f"Could not find video directory with metadata in: {dataset_path}")

    if num_workers == -1:
        num_workers = min(cpu_count(), 8)
    elif num_workers == 0:
        num_workers = 1

    # Load sentence transformer model
    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    # Load all metadata files
    all_episodes = []
    metadata_files = list(video_dir.glob("*_metadata.json"))

    for metadata_file in metadata_files:
        with open(metadata_file, "r") as f:
            episodes = json.load(f)
            all_episodes.extend(episodes)

    print(f"Found {len(all_episodes)} episodes across {len(metadata_files)} metadata files")

    # Limit trajectories if specified
    if max_trajectories is not None and max_trajectories > 0:
        all_episodes = all_episodes[:max_trajectories]

    # Pre-compute language embeddings
    print("Computing language embeddings...")
    for episode in tqdm(all_episodes, desc="Computing embeddings"):
        instruction = episode["instruction"]
        if instruction not in lang_cache:
            lang_cache[instruction] = lang_model.encode(instruction)

    # Process episodes
    entries = []
    if num_workers == 1:
        # Sequential processing
        for episode in tqdm(all_episodes, desc="Processing episodes"):
            lang_vec = lang_cache[episode["instruction"]]
            result = _process_episode((episode, str(video_dir), lang_vec, output_dir, dataset_name, max_frames, fps))
            if result:
                entries.append(result)
    else:
        # Parallel processing
        from multiprocessing import Pool

        args_list = [
            (episode, str(video_dir), lang_cache[episode["instruction"]], output_dir, dataset_name, max_frames, fps)
            for episode in all_episodes
        ]

        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(pool.imap_unordered(_process_episode, args_list), total=len(args_list), desc="Processing episodes")
            )

        entries = [r for r in results if r is not None]

    print(f"Successfully processed {len(entries)} episodes")

    # Create HuggingFace dataset
    if not entries:
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

    return Dataset.from_list(entries)
