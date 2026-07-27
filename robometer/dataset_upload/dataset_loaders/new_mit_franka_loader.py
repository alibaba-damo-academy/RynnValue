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


# Task name to language instruction mapping
TASK_INSTRUCTIONS = {
    "foldtowel": "Fold the towel in half",
    "movebanana": "Pick up the banana from the blue plate and place it on the green plate",
    "movemouse": "Pick up the mouse and place it right next to the laptop, while avoiding spilling coffee",
    "pourpebble": "Pour the pebbles from the cup onto the plate",
    "pulltissue": "Pull a tissue from the tissue box",
    "putspoon": "Pick up the spoon and place it inside the cup",
    "stirpot": "Pick up the spatula and stir the beans in the pot",
}


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


def _build_video_paths(
    output_dir: str, dataset_label: str, episode_idx: int, task_name: str, view: str
) -> tuple[str, str]:
    """Build output video paths with shard structure."""
    shard_index = episode_idx // 1000
    shard_dir = f"shard_{shard_index:04d}"
    episode_dir = os.path.join(output_dir, dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}")
    os.makedirs(episode_dir, exist_ok=True)
    filename = f"{task_name}_{view}.mp4"
    full_path = os.path.join(episode_dir, filename)
    rel_path = os.path.join(dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}", filename)
    return full_path, rel_path


def _process_episode(args):
    """Process a single episode."""
    episode_meta, lang_vec, output_dir, dataset_label, max_frames, fps = args

    # Load video frames
    video_path = episode_meta["video_path"]
    try:
        frames = _load_video_frames(video_path)
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None

    if not frames:
        return None

    # Normalize quality label
    quality_label = episode_meta["quality_label"]
    if quality_label == "success":
        quality_label = "successful"
    elif quality_label == "fail":
        quality_label = "failure"
    # "suboptimal" stays as is

    # Build output video path
    full_path, rel_path = _build_video_paths(
        output_dir, dataset_label, episode_meta["episode_idx"], episode_meta["task_name"], episode_meta["view"]
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


def convert_new_mit_franka_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
    exclude_wrist_cam: bool = False,
) -> Dataset:
    """Convert New MIT Franka dataset to HF format.

    Args:
        dataset_path: Path to the dataset directory containing task folders
        dataset_name: Name for the dataset
        output_dir: Output directory for processed videos
        max_trajectories: Maximum number of trajectories to process (None for all)
        max_frames: Maximum frames per trajectory
        fps: Frames per second for output videos
        num_workers: Number of worker processes (-1 for auto, 0 for sequential)
        exclude_wrist_cam: If True, only process external camera views and skip wrist camera

    Returns:
        HuggingFace Dataset
    """
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    if num_workers == -1:
        num_workers = min(cpu_count(), 8)
    elif num_workers == 0:
        num_workers = 1

    # Load sentence transformer model
    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    # Collect all episodes
    all_episodes = []
    episode_idx = 0

    # Iterate through task folders
    task_folders = [d for d in root.iterdir() if d.is_dir()]

    for task_folder in sorted(task_folders):
        task_name = task_folder.name
        if task_name not in TASK_INSTRUCTIONS:
            print(f"Warning: Unknown task '{task_name}', skipping...")
            continue

        instruction = TASK_INSTRUCTIONS[task_name]

        # Iterate through quality folders (suboptimal, failure, success)
        for quality_folder in ["suboptimal", "failure", "success"]:
            quality_dir = task_folder / quality_folder
            if not quality_dir.exists():
                continue

            # Get all video files
            video_files = sorted(quality_dir.glob("*.mp4"))

            for video_file in video_files:
                # Determine view type (ext or wrist)
                if "_ext.mp4" in video_file.name:
                    view = "ext"
                elif "_wrist.mp4" in video_file.name:
                    view = "wrist"
                else:
                    print(f"Warning: Unknown view type in {video_file.name}, skipping...")
                    continue

                # Skip wrist camera if exclude_wrist_cam is True
                if exclude_wrist_cam and view == "wrist":
                    continue

                episode_meta = {
                    "episode_idx": episode_idx,
                    "task_name": task_name,
                    "video_path": str(video_file),
                    "instruction": instruction,
                    "quality_label": quality_folder,  # "success", "failure", or "suboptimal"
                    "view": view,
                }
                all_episodes.append(episode_meta)
                episode_idx += 1

    print(f"Found {len(all_episodes)} episodes across {len(task_folders)} tasks")

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
            result = _process_episode((episode, lang_vec, output_dir, dataset_name, max_frames, fps))
            if result:
                entries.append(result)
    else:
        # Parallel processing
        from multiprocessing import Pool

        args_list = [
            (episode, lang_cache[episode["instruction"]], output_dir, dataset_name, max_frames, fps)
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
