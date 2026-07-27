import os
import json
import shutil
import gc
import glob
from re import T
import zipfile
import tempfile
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset

from dataset_upload.helpers import (
    create_hf_trajectory,
    generate_unique_id,
    load_sentence_transformer_model,
)
from tqdm import tqdm

# Disable GPUs for TensorFlow in this loader to avoid CUDA context issues in workers
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

to_skip = set(["pull_out_tissue_from_tissue_box_h1.zip"])  # skip because it's incorrect videos

# Google Sheet with task descriptions
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/158Wzf8Xywky3aHJSCfp3OZxf4bkhzAJdcG94eHf8gVc/export?format=csv&gid=1307250382"


def _load_google_sheet_tasks() -> dict[str, str]:
    """Load task descriptions from Google Sheet.

    Returns:
        Dictionary mapping task names (from zip filenames) to task descriptions.
    """
    try:
        # Read the Google Sheet as CSV, skipping the first 2 rows and using row 3 as header
        df = pd.read_csv(GOOGLE_SHEET_URL, header=2)

        # Create a mapping from task name to description
        task_map = {}
        for _, row in df.iterrows():
            # Check if we have valid task name and description
            if pd.notna(row.get("Task Name")) and pd.notna(row.get("Task Description")):
                task_name = row["Task Name"]
                description = row["Task Description"]

                # Create mapping for both with and without .zip extension
                task_map[f"{task_name}.zip"] = description
                task_map[task_name] = description

        print(f"Loaded {len(task_map) // 2} task descriptions from Google Sheet")
        return task_map
    except Exception as e:
        print(f"Warning: Failed to load Google Sheet: {e}")
        import traceback

        traceback.print_exc()
        return {}


def _stable_shard_for_index(index: int, shard_modulus: int = 1000) -> str:
    """Generate stable shard directory name for trajectory indexing."""
    try:
        idx = int(index)
    except Exception:
        idx = abs(hash(str(index)))
    shard_index = idx // shard_modulus
    return f"shard_{shard_index:04d}"


def _build_humanoid_video_paths(
    output_dir: str,
    dataset_label: str,
    episode_idx: int,
    zip_file: str,
) -> tuple[str, str]:
    """Build video paths for humanoid everyday dataset."""
    shard_dir = _stable_shard_for_index(episode_idx)
    task_prefix = zip_file.split("/")[-2]
    episode_dir = os.path.join(output_dir, dataset_label.lower(), task_prefix, shard_dir, f"episode_{episode_idx:06d}")
    os.makedirs(episode_dir, exist_ok=True)
    full_path = os.path.join(episode_dir, f"clip.mp4")
    rel_path = os.path.join(dataset_label.lower(), task_prefix, shard_dir, f"episode_{episode_idx:06d}", f"clip.mp4")
    return full_path, rel_path


def _process_single_humanoid_episode(args):
    """Process a single episode from humanoid everyday dataset."""
    episode_data, ep_idx, task, lang_vec, output_dir, dataset_name, max_frames, fps, zip_file = args

    episode_entries = []

    try:
        # Extract frames from episode data
        frames = []
        for step_data in episode_data:
            if "image" in step_data:
                # Convert numpy array to uint8 if needed
                img = step_data["image"]
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                frames.append(img)

        if not frames:
            return episode_entries

        full_path, rel_path = _build_humanoid_video_paths(
            output_dir=output_dir,
            dataset_label=dataset_name,
            episode_idx=ep_idx,
            zip_file=zip_file,
        )

        # Create trajectory dictionary
        traj_dict = {
            "id": generate_unique_id(),
            "frames": frames,
            "task": task,
            "is_robot": True,
            "quality_label": "successful",
            "preference_group_id": None,
            "preference_rank": None,
        }

        try:
            entry = create_hf_trajectory(
                traj_dict=traj_dict,
                video_path=full_path,
                lang_vector=lang_vec,
                max_frames=max_frames,
                dataset_name=dataset_name,
                use_video=True,
                fps=fps,
            )
        except Exception as e:
            print(f"Warning: Failed to create HF trajectory for ep {ep_idx}: {e}")
            return episode_entries

        if entry:
            entry["frames"] = rel_path
            episode_entries.append(entry)

    except Exception as e:
        print(f"Warning: Failed to process episode {ep_idx}: {e}")
        return episode_entries

    return episode_entries


def _create_humanoid_dataloader(zip_path: str):
    """Create a humanoid everyday dataloader for a zip file."""
    try:
        # Import humanoid_everyday dataloader
        from humanoid_everyday import Dataloader

        # Load dataset from zip file
        ds = Dataloader(zip_path)
        return ds

    except ImportError:
        print(f"Warning: humanoid_everyday package not found. Please install it with: pip install humanoid_everyday")
        return None
    except Exception as e:
        print(f"Warning: Failed to create dataloader from {zip_path}: {e}")
        return None


def _load_single_humanoid_episode(ds, episode_idx: int):
    """Load a single episode from an existing humanoid everyday dataloader."""
    try:
        # Get the specific episode
        episode = ds[episode_idx]

        # Convert episode to list of step dictionaries
        episode_data = []
        for step in episode:
            episode_data.append(step)

        return episode_data

    except Exception as e:
        print(f"Warning: Failed to load episode {episode_idx}: {e}")
        return None


def convert_humanoid_everyday_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
) -> Dataset:
    """Convert Humanoid Everyday datasets to HF format by writing videos directly.

    Args:
        dataset_path: Root path that contains zip files with humanoid everyday datasets.
        dataset_name: Name to tag the resulting dataset (e.g., 'humanoid_everyday').
        output_dir: Where to write video files and dataset.
        max_trajectories: Limit number of produced trajectories (None/-1 for all).
        max_frames: Max frames per video.
        fps: Video fps.
        num_workers: Number of workers for parallel processing.
    """

    # Normalize and checks
    if dataset_name is None:
        raise ValueError("dataset_name is required")

    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Find all zip files in the dataset path
    zip_files = glob.glob(os.path.join(root, "**/*.zip"), recursive=True)
    if not zip_files:
        raise FileNotFoundError(f"No .zip files found in {dataset_path}")

    print(f"Found {len(zip_files)} zip files to process")

    # Determine workers
    if num_workers == -1:
        num_workers = min(cpu_count(), 8)
    elif num_workers == 0:
        num_workers = 1

    # Language model and cache
    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    # Process all zip files
    all_entries: list[dict[str, Any]] = []
    produced = 0
    max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)

    print("Loading task descriptions from Google Sheet...")
    google_sheet_tasks = _load_google_sheet_tasks()

    for zip_file in tqdm(zip_files, desc="Processing zip files"):
        print(f"Processing zip file: {zip_file}")

        if zip_file.split("/")[-1] in to_skip:
            print(f"Skipping zip file: {zip_file}")
            continue

        # Create dataloader once for this zip file
        ds = _create_humanoid_dataloader(zip_file)
        if ds is None:
            print(f"Failed to create dataloader for {zip_file}")
            continue

        # Try to find task in Google Sheet
        zip_filename = zip_file.split("/")[-1]
        if zip_filename in google_sheet_tasks:
            task_name = google_sheet_tasks[zip_filename]
            print(f"Found task description from Google Sheet: {task_name}")
        else:
            try:
                # Get the metadata.json for getting task description
                # find metadata.json in the unzipped directory using correct glob pattern
                metadata_paths = glob.glob(
                    os.path.join(zip_file.replace(".zip", ""), "**", "metadata.json"), recursive=True
                )
                if not metadata_paths:
                    print(f"metadata.json not found in extracted directory for {zip_file}")

                else:
                    metadata_path = metadata_paths[0]
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    task_name = metadata["description"]
                    print(f"Found task description from metadata.json: {task_name}")
            except Exception as e:
                print(f"Warning: Failed to load metadata.json for {zip_file}: {e}")
            print(f"Warning: No task description found for {zip_filename} in Google Sheet, skipping")
            shutil.rmtree(zip_file.replace(".zip", ""))
            continue

        # Precompute embedding for this task
        if task_name not in lang_cache:
            lang_cache[task_name] = lang_model.encode(task_name)
        lang_vec = lang_cache[task_name]

        episode_count = len(ds)
        if episode_count == 0:
            print(f"No episodes found in {zip_file}")
            shutil.rmtree(zip_file.replace(".zip", ""))
            continue

        print(f"Found {episode_count} episodes in {zip_file}")

        # Process episodes one at a time to save memory
        for ep_idx in tqdm(range(episode_count), desc=f"Processing episodes in {zip_file}"):
            # Load single episode using the existing dataloader
            episode_data = _load_single_humanoid_episode(ds, ep_idx)
            if episode_data is None:
                print(f"Failed to load episode {ep_idx} from {zip_file}")
                continue

            # Process single episode
            episode_entries = _process_single_humanoid_episode((
                episode_data,
                ep_idx,
                task_name,
                lang_vec,
                output_dir,
                dataset_name,
                max_frames,
                fps,
                zip_file,
            ))

            all_entries.extend(episode_entries)
            produced += len(episode_entries)

            # Clean up episode data to free memory
            del episode_data

            if produced >= max_limit:
                break

        # remove the unzipped file after done since humanoid loader unzips it
        shutil.rmtree(zip_file.replace(".zip", ""))

        if produced >= max_limit:
            break

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
