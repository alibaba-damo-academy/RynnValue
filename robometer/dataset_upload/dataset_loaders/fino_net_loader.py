#!/usr/bin/env python3
"""
FinoNet dataset loader for the generic dataset converter for Robometer model training.
https://huggingface.co/datasets/jesbu1/fino-net-dataset
This module contains FinoNet-specific logic for loading and processing image sequences.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from dataset_upload.helpers import (
    create_hf_trajectory,
    generate_unique_id,
    load_sentence_transformer_model,
)
from tqdm import tqdm
from datasets import Dataset


# Task mapping from task names to instructions
TASK_TO_INSTRUCTION = {
    "put_on": "put the single block on the table onto the stack",
    "put_in": "put the object on the table into the container",
    "place": "place the left object on the table onto the stack",
    "pour": "pour the contents of the cup into the receptacle on the table without spilling",
    "push": "push the object towards the right without knocking it over",
}


def _load_annotation_files(dataset_path: Path) -> dict[str, dict[int, int]]:
    """Load annotation files for all tasks.

    Returns:
        Dictionary mapping task name to {episode_number: label} where label is 0 for success, 1 for failure
    """
    annotations = {}

    # The annotation files are in the root directory
    annotation_files = {
        "put_on": "put_on_annotation.txt",
        "put_in": "put_in_annotation.txt",
        "place": "place_annotation.txt",
        "pour": "pour_annotation.txt",
        "push": "push_annotation.txt",
    }

    for task_name, filename in annotation_files.items():
        annot_file = dataset_path / filename
        if not annot_file.exists():
            print(f"Warning: {filename} not found, skipping {task_name}")
            continue

        task_annotations = {}
        with open(annot_file, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                # Parse format: name, label
                parts = line.split(",")
                if len(parts) >= 2 and i > 0:  # i ==0 is the header line
                    episode_num = int(parts[0].strip())
                    label = int(parts[1].strip())
                    task_annotations[episode_num] = label

        annotations[task_name] = task_annotations
        print(f"Loaded {len(task_annotations)} annotations for {task_name}")

    return annotations


def _load_episode_images(episode_dir: Path) -> list[Path]:
    """Load all image files from an episode directory, sorted by frame number.

    Args:
        episode_dir: Path to episode directory containing PNG files

    Returns:
        List of image file paths sorted by frame number
    """
    if not episode_dir.exists():
        return []

    # Find all PNG files
    image_files = []
    for img_file in episode_dir.glob("*.png"):
        image_files.append(img_file)

    # Sort by frame number (e.g., frame0000000.png, frame0000024.png)
    def get_frame_num(path: Path) -> int:
        name = path.stem  # e.g., "frame0000000"
        try:
            return int(name.replace("frame", ""))
        except:
            return 0

    image_files.sort(key=get_frame_num)
    return image_files


def _load_image_as_numpy(img_path: Path) -> np.ndarray:
    """Load a PNG image and return as numpy array in RGB format."""
    with Image.open(img_path) as img:
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Return as numpy array
        return np.array(img)


def _discover_episodes(dataset_path: Path) -> list[tuple[str, int, int]]:
    """Discover all episodes in the FinoNet dataset structure.

    Expected structure (after unzipping failure.zip):
    dataset_path/
        failnet_dataset/
            rgb_imgs/
                put_on/
                    9/
                        frame0000000.png
                        frame0000024.png
                        ...
                put_in/
                place/
                pour/
                push/

    Returns:
        List of tuples: (task_name, episode_number, label)
    """
    episodes = []

    # Load annotations
    annotations = _load_annotation_files(dataset_path)

    # Find the unzipped dataset directory
    rgb_imgs_dir = dataset_path / "failnet_dataset" / "rgb_imgs"
    if not rgb_imgs_dir.exists():
        print(f"Warning: rgb_imgs directory not found at {rgb_imgs_dir}")
        return episodes

    # Iterate over task directories
    for task_dir in rgb_imgs_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        if task_name not in annotations:
            print(f"Skipping task {task_name} (no annotations)")
            continue

        task_annotations = annotations[task_name]

        # Iterate over episode subdirectories
        for episode_dir in task_dir.iterdir():
            if not episode_dir.is_dir():
                continue

            try:
                episode_num = int(episode_dir.name)
            except ValueError:
                continue

            # Get label from annotations
            if episode_num not in task_annotations:
                print(f"Warning: Episode {episode_num} for task {task_name} not in annotations")
                continue

            label = task_annotations[episode_num]
            episodes.append((task_name, episode_num, label))

    print(f"Discovered {len(episodes)} episodes across {len(annotations)} tasks")
    return episodes


def _process_single_episode(args):
    """Worker to process a single episode into a trajectory entry.

    Returns a single entry dict or empty list if failed.
    """
    (
        task_name,
        episode_num,
        label,
        dataset_name,
        output_dir,
        max_frames,
        fps,
        task_instruction,
        lang_vec,
        rgb_imgs_dir,
    ) = args

    try:
        # Load images for this episode
        episode_dir = rgb_imgs_dir / task_name / str(episode_num)
        image_files = _load_episode_images(episode_dir)

        if not image_files:
            print(f"Warning: No images found for episode {episode_num} in task {task_name}")
            return []

        # Load all frames into memory
        frames = []
        for img_path in image_files:
            frame = _load_image_as_numpy(img_path)
            frames.append(frame)

        frames = np.array(frames)  # Shape: (T, H, W, 3)

        # skip first 10 frames because they typically don't show the arm
        frames = frames[10:]

        # Determine quality label (0 = success, 1 = failure)
        quality_label = "failed" if label == 1 else "successful"

        # Create video path
        episode_video_dir = os.path.join(output_dir, dataset_name.lower(), task_name, f"episode_{episode_num:06d}")
        os.makedirs(episode_video_dir, exist_ok=True)
        video_filename = "clip.mp4"
        full_video_path = os.path.join(episode_video_dir, video_filename)
        rel_video_path = os.path.join(dataset_name.lower(), task_name, f"episode_{episode_num:06d}", video_filename)
        # Create trajectory dict
        traj_dict = {
            "id": generate_unique_id(),
            "frames": frames,
            "task": task_instruction,
            "is_robot": True,
            "quality_label": quality_label,
            "preference_group_id": None,
            "preference_rank": None,
        }

        # Create HF trajectory entry
        entry = create_hf_trajectory(
            traj_dict=traj_dict,
            video_path=full_video_path,
            lang_vector=lang_vec,
            max_frames=max_frames,
            dataset_name=dataset_name,
            use_video=True,
            fps=fps,
        )

        if entry:
            entry["frames"] = rel_video_path
            return [entry]

        return []

    except Exception as e:
        print(f"Error processing episode {episode_num} for task {task_name}: {e}")
        return []


def _stable_shard_for_index(index: int, shard_modulus: int = 1000) -> str:
    """Deterministically bucket an index into a shard directory name."""
    try:
        idx = int(index)
    except Exception:
        idx = abs(hash(str(index)))
    shard_index = idx // shard_modulus
    return f"shard_{shard_index:04d}"


def convert_fino_net_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
) -> Dataset:
    """Convert the FinoNet dataset to HF format by writing videos directly.

    This follows the streaming approach: iterate episodes, write videos,
    assemble entries, and return a Dataset at the end.
    """

    if dataset_name is None:
        raise ValueError("dataset_name is required")

    base_path = Path(dataset_path)
    if not base_path.exists():
        raise FileNotFoundError(f"FinoNet dataset path not found: {base_path}")

    # Discover all episodes
    episodes = _discover_episodes(base_path)
    if len(episodes) == 0:
        # Return empty dataset
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
            "partial_success": [],
        })

    # Limit trajectories if specified
    if max_trajectories is not None and max_trajectories > 0:
        episodes = episodes[:max_trajectories]

    # Language model and cache
    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    # Determine workers
    if num_workers == -1:
        try:
            from multiprocessing import cpu_count as _cpu_count

            num_workers = min(_cpu_count(), 8)
        except Exception:
            num_workers = 1
    elif num_workers == 0:
        num_workers = 1

    batch_size = 64

    entries: list[dict[str, Any]] = []
    produced_count = 0
    max_limit = float("inf") if (max_trajectories is None or max_trajectories <= 0) else int(max_trajectories)

    print(f"Found {len(episodes)} episodes; processing in batches of {batch_size} with {num_workers} workers...")

    # Path to rgb_imgs directory
    rgb_imgs_dir = base_path / "failnet_dataset" / "rgb_imgs"

    # Process in batches
    episode_batch: list[tuple[str, int, int]] = []
    info_batch: list[tuple[str, Any]] = []  # (task_instruction, lang_vec)

    for idx, (task_name, episode_num, label) in enumerate(tqdm(episodes, desc="Queuing FinoNet episodes")):
        if produced_count >= max_limit:
            break

        # Get task instruction
        if task_name not in TASK_TO_INSTRUCTION:
            print(f"Skipping unknown task: {task_name}")
            continue

        task_instruction = TASK_TO_INSTRUCTION[task_name]

        # Get or create language embedding
        if task_instruction not in lang_cache:
            lang_cache[task_instruction] = lang_model.encode(task_instruction)
        lang_vec = lang_cache[task_instruction]

        episode_batch.append((task_name, episode_num, label))
        info_batch.append((task_instruction, lang_vec))

        if len(episode_batch) >= batch_size or idx + 1 == len(episodes):
            # Build worker args
            worker_args = list(
                zip(
                    [t for (t, _, _) in episode_batch],
                    [e for (_, e, _) in episode_batch],
                    [l for (_, _, l) in episode_batch],
                    [dataset_name] * len(episode_batch),
                    [output_dir] * len(episode_batch),
                    [max_frames] * len(episode_batch),
                    [fps] * len(episode_batch),
                    [ti for (ti, _) in info_batch],
                    [lv for (_, lv) in info_batch],
                    [rgb_imgs_dir] * len(episode_batch),
                    strict=False,
                )
            )

            if num_workers == 1:
                # Sequential processing
                for args in worker_args:
                    entries.extend(_process_single_episode(args))
                    produced_count += 1
                    if produced_count >= max_limit:
                        break
            else:
                from multiprocessing import Pool

                with Pool(processes=num_workers) as pool:
                    results = list(
                        tqdm(
                            pool.imap_unordered(_process_single_episode, worker_args),
                            total=len(worker_args),
                            desc=f"Processing batch (workers={num_workers})",
                        )
                    )
                for res in results:
                    entries.extend(res)
                    produced_count += 1
                    if produced_count >= max_limit:
                        break

            # Clear batch
            episode_batch = []
            info_batch = []

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
            "partial_success": [],
        })

    print(f"Successfully created {len(entries)} entries")
    return Dataset.from_list(entries)
