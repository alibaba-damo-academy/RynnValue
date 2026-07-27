import json
import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, concatenate_datasets
from PIL import Image
from tqdm import tqdm

from dataset_upload.helpers import (
    create_hf_trajectory,
    generate_unique_id,
    load_sentence_transformer_model,
)


def _load_gif_frames(gif_path: str) -> list[np.ndarray]:
    """Load frames from a GIF file."""
    if not os.path.exists(gif_path):
        raise FileNotFoundError(f"GIF file not found: {gif_path}")

    frames = []
    with Image.open(gif_path) as img:
        try:
            for frame_idx in range(img.n_frames):
                img.seek(frame_idx)
                # Convert to RGB and then to numpy array
                frame = img.convert("RGB")
                frames.append(np.array(frame))
        except EOFError:
            pass  # End of frames

    return frames


def _parse_quality_label(folder_name: str) -> str:
    """Parse quality label from folder name."""
    if "closs_succ" in folder_name.lower():
        return "suboptimal"
    elif "fail" in folder_name.lower():
        return "failure"
    elif "succ" in folder_name.lower():
        return "successful"
    else:
        raise ValueError(f"Unknown quality label in folder: {folder_name}")


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


def _process_episode(args):
    """Process a single episode for a specific view."""
    gif_path, episode_idx, instruction, quality_label, view, lang_vec, output_dir, dataset_label, max_frames, fps = args

    # Load GIF frames
    try:
        frames = _load_gif_frames(gif_path)
    except Exception as e:
        print(f"Error loading GIF {gif_path}: {e}")
        return None

    if not frames:
        return None

    # Build output video path
    full_path, rel_path = _build_video_paths(output_dir, dataset_label, episode_idx, view)

    traj_dict = {
        "id": generate_unique_id(),
        "frames": frames,
        "task": instruction,
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


def convert_utd_so101_clean_policy_ranking_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    view: str = "wrist",  # "wrist" or "top"
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
) -> Dataset:
    """Convert UTD SO101 Clean Policy Ranking dataset to HF format.

    Args:
        dataset_path: Path to the dataset directory containing quality-labeled subfolders
        dataset_name: Name for the dataset
        output_dir: Output directory for processed videos
        view: Camera view to use ("wrist" or "top")
        max_trajectories: Maximum number of trajectories to process per quality label (None for all)
        max_frames: Maximum frames per trajectory
        fps: Frames per second for output videos
        num_workers: Number of worker processes (-1 for auto, 0 for sequential)

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

    # Find all quality-labeled subdirectories
    all_episodes = []
    quality_dirs = []

    for item in root.iterdir():
        if item.is_dir() and any(label in item.name.lower() for label in ["succ", "fail", "closs_succ"]):
            # Find the inner directory containing the actual data
            inner_dirs = [d for d in item.iterdir() if d.is_dir()]
            if inner_dirs:
                quality_dirs.append((inner_dirs[0], _parse_quality_label(item.name)))

    if not quality_dirs:
        raise FileNotFoundError(f"No quality-labeled directories found in: {dataset_path}")

    print(f"Found {len(quality_dirs)} quality-labeled directories")

    # Process each quality directory
    global_episode_idx = 0
    for data_dir, quality_label in quality_dirs:
        print(f"Processing {quality_label} episodes from: {data_dir}")

        # Load task instructions
        vla_task_path = data_dir / "vla_task.json"
        if not vla_task_path.exists():
            print(f"Warning: vla_task.json not found in {data_dir}, skipping")
            continue

        with open(vla_task_path, "r") as f:
            instructions = json.load(f)

        # Find all episode GIFs for the specified view
        gif_files = sorted(data_dir.glob(f"episode_*_{view}.gif"))

        print(f"  Found {len(gif_files)} episodes for {view} view with quality: {quality_label}")

        # Limit trajectories if specified
        if max_trajectories is not None and max_trajectories > 0:
            gif_files = gif_files[:max_trajectories]

        # Match episodes to instructions
        for gif_file in gif_files:
            # Extract episode number from filename (e.g., episode_0_wrist.gif -> 0)
            episode_num = int(gif_file.stem.split("_")[1])

            if episode_num >= len(instructions):
                print(f"Warning: Episode {episode_num} has no instruction, skipping")
                continue

            instruction = instructions[episode_num]

            # Pre-compute language embedding
            if instruction not in lang_cache:
                lang_cache[instruction] = lang_model.encode(instruction)

            all_episodes.append({
                "gif_path": str(gif_file),
                "episode_idx": global_episode_idx,
                "instruction": instruction,
                "quality_label": quality_label,
                "view": view,
                "lang_vec": lang_cache[instruction],
            })
            global_episode_idx += 1

    print(f"Total episodes to process: {len(all_episodes)}")
    print(f"Unique instructions: {len(lang_cache)}")

    # Process episodes
    entries = []
    if num_workers == 1:
        # Sequential processing
        for episode in tqdm(all_episodes, desc=f"Processing {view} episodes"):
            result = _process_episode((
                episode["gif_path"],
                episode["episode_idx"],
                episode["instruction"],
                episode["quality_label"],
                episode["view"],
                episode["lang_vec"],
                output_dir,
                dataset_name,
                max_frames,
                fps,
            ))
            if result:
                entries.append(result)
    else:
        # Parallel processing
        from multiprocessing import Pool

        args_list = [
            (
                episode["gif_path"],
                episode["episode_idx"],
                episode["instruction"],
                episode["quality_label"],
                episode["view"],
                episode["lang_vec"],
                output_dir,
                dataset_name,
                max_frames,
                fps,
            )
            for episode in all_episodes
        ]

        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_process_episode, args_list),
                    total=len(args_list),
                    desc=f"Processing {view} episodes",
                )
            )

        entries = [r for r in results if r is not None]

    print(f"Successfully processed {len(entries)} episodes for {view} view")

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
