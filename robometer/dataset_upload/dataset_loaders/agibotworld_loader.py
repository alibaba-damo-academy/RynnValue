"""
AgiBotWorld dataset loader for the generic dataset converter for Robometer model training.
This module contains AgiBotWorld-specific logic for loading and processing data using
HuggingFace streaming to efficiently handle large datasets.
"""

import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from dataset_upload.helpers import (
    create_hf_trajectory,
    create_trajectory_video_optimized,
    load_sentence_transformer_model,
    generate_unique_id,
)
from dataset_upload.video_helpers import load_video_frames
from tqdm import tqdm

import datasets as hfds
from datasets import Dataset, load_dataset

# Episode/task helpers built earlier
from dataset_upload.data_scripts.agibot.agibot_helper import get_episode_record

# ------------------------------
# Small utilities
# ------------------------------

CAMERA_KEYS = {
    "head_color",
    "head_left_fisheye_color",
    "head_right_fisheye_color",
    "head_center_fisheye_color",
}


def _stable_shard_for_episode(episode_id: str, shard_modulus: int = 1000) -> str:
    """Return a stable top-level shard name based on episode id.

    Keeps at most ~shard_modulus episode directories per shard.
    """

    try:
        idx = int(episode_id)
    except Exception:
        idx = abs(hash(episode_id))
    shard_index = idx // shard_modulus
    return f"shard_{shard_index:04d}"


def _parse_episode_and_camera(key: str) -> tuple[str, str | None]:
    """Parse __key__ like '678985/videos/head_color' -> ('678985', 'head_color')."""
    parts = key.split("/")
    if len(parts) < 3:
        return parts[0], None
    return parts[0], parts[2]


def _build_video_paths(
    output_dir: str,
    dataset_name: str,
    episode_id: str,
    subtask_idx: int,
    camera: str,
) -> tuple[str, str]:
    """Return (full_path, relative_path) using a two-level shard + per-episode layout.

    Layout:
      <output>/<dataset>/<shard_X>/<episode_id>/clip_<k>@<camera>.mp4
    This avoids >1k files per directory while keeping resume-friendly structure.
    """
    shard_dir = _stable_shard_for_episode(episode_id)
    episode_dir = os.path.join(output_dir, dataset_name.lower(), shard_dir, f"episode_{episode_id}")
    os.makedirs(episode_dir, exist_ok=True)
    filename = f"clip_{subtask_idx}@{camera}.mp4"
    full_path = os.path.join(episode_dir, filename)
    rel_path = os.path.join(dataset_name.lower(), shard_dir, f"episode_{episode_id}", filename)
    return full_path, rel_path


def _collect_unique_texts_for_batch(records: list[tuple[str, dict]]) -> list[str]:
    """Collect unique instruction texts from a list of (episode_id, record) pairs."""
    texts: list[str] = []
    seen: set = set()
    for _episode_id, rec in records:
        # Full trajectory instruction
        full_text = rec.get("task_name") or rec.get("task_description") or ""
        if full_text and full_text not in seen:
            seen.add(full_text)
            texts.append(full_text)

        # Subtasks
        actions = rec.get("label_info", {}).get("action_config", [])
        for a in actions:
            t = (a or {}).get("action_text", "").strip()
            if t and t not in seen:
                seen.add(t)
                texts.append(t)
    return texts


def _encode_texts(texts: list[str], model) -> dict[str, Any]:
    """Encode a list of texts to vectors using a preloaded model, with caching."""
    if not texts:
        return {}
    vectors = model.encode(texts)
    return dict(zip(texts, vectors, strict=False))


def _frames_for_subrange(frames: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return frames[start:end] with guardrails; [start, end) semantics."""
    start = max(int(start), 0)
    end = min(int(end), len(frames))
    if start >= end:
        return np.empty((0,), dtype=object)
    return frames[start:end]


def _process_single_stream_sample(
    sample: dict[str, Any],
    embeddings: dict[str, Any],
    output_dir: str,
    dataset_name: str,
    max_frames: int,
    fps: int,
) -> list[dict]:
    """Process one streaming sample: returns zero or more HF entries.

    This function does not load any language model; it expects embeddings for
    the relevant instruction texts to be provided.
    """

    result_entries: list[dict] = []

    # Extract key and keep only camera samples we care about
    key = sample.get("__key__") or ""
    episode_id, camera = _parse_episode_and_camera(key)
    if not camera or camera not in CAMERA_KEYS:
        return result_entries

    # Load associated episode record for task/subtasks
    try:
        _json_path, rec = get_episode_record(episode_id)
    except Exception:
        return result_entries

    # Get video bytes (dataset exposes only 'mp4', '__key__', '__url__')
    video_bytes = sample.get("mp4")
    if not video_bytes:
        return result_entries

    # Decode the video to frames once
    try:
        frames = load_video_frames(video_bytes)
    except Exception:
        return result_entries

    if frames is None or len(frames) == 0:
        return result_entries

    # Build entries: full + subtasks
    # Full trajectory
    full_text = rec.get("task_name") or rec.get("task_description") or ""
    if full_text:
        subtask_idx = 0
        full_out_path, rel_path = _build_video_paths(output_dir, dataset_name, episode_id, subtask_idx, camera)
        # Create video if missing
        if not os.path.exists(full_out_path):
            _ = create_trajectory_video_optimized(frames, full_out_path, max_frames=max_frames, fps=fps)

        lang_vec = embeddings.get(full_text)
        if lang_vec is None:
            # As a fallback, keep empty vector to avoid crashing
            lang_vec = np.zeros((384,), dtype=np.float32)

        traj_dict = {
            "id": generate_unique_id(),
            "frames": frames,  # Not used by create_hf_trajectory now since we already wrote, but pass for API
            "task": full_text,
            "is_robot": True,
            "quality_label": "successful",
            "preference_group_id": None,
            "preference_rank": None,
        }
        entry = create_hf_trajectory(
            traj_dict=traj_dict,
            video_path=full_out_path,
            lang_vector=lang_vec,
            max_frames=max_frames,
            dataset_name=dataset_name,
            use_video=True,
            fps=fps,
        )
        if entry:
            entry["frames"] = rel_path
            result_entries.append(entry)

    # Subtasks
    actions = rec.get("label_info", {}).get("action_config", [])
    for i, a in enumerate(actions, start=1):
        if not isinstance(a, dict):
            continue
        text = (a.get("action_text") or "").strip()
        if not text:
            continue
        start = a.get("start_frame", 0)
        end = a.get("end_frame", len(frames))
        sub_frames = _frames_for_subrange(frames, start, end)
        if sub_frames.size == 0:
            continue

        out_path, rel_path = _build_video_paths(output_dir, dataset_name, episode_id, i, camera)
        if not os.path.exists(out_path):
            _ = create_trajectory_video_optimized(sub_frames, out_path, max_frames=max_frames, fps=fps)

        lang_vec = embeddings.get(text)
        if lang_vec is None:
            lang_vec = np.zeros((384,), dtype=np.float32)

        traj_dict = {
            "id": generate_unique_id(),
            "frames": sub_frames,
            "task": text,
            "is_robot": True,
            "quality_label": "successful",
            "preference_group_id": None,
            "preference_rank": None,
        }
        entry = create_hf_trajectory(
            traj_dict=traj_dict,
            video_path=out_path,
            lang_vector=lang_vec,
            max_frames=max_frames,
            dataset_name=dataset_name,
            use_video=True,
            fps=fps,
        )
        if entry:
            entry["frames"] = rel_path
            result_entries.append(entry)

    return result_entries


def convert_agibotworld_streaming_to_hf(
    dataset_name: str,
    output_dir: str,
    dataset_label: str = "agibotworld",
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
) -> Dataset:
    """Stream AgiBotWorld, extract camera videos, and write HF entries.

    Returns a datasets.Dataset built from the collected entries. All videos are
    saved to disk under output_dir.
    """

    # Load streaming dataset
    ds = load_dataset(dataset_name, streaming=True, split="train")
    # Some shards expose PNG frames instead of MP4. Widen features so casting
    # does not fail during iteration; we'll simply skip non-MP4 samples.
    widened = hfds.Features({
        "__key__": hfds.Value("string"),
        "__url__": hfds.Value("string"),
        "mp4": hfds.Value("binary"),
        "png": hfds.Value("binary"),
    })
    try:
        ds = ds.cast(widened)
    except Exception:
        pass

    # Determine workers
    if num_workers == -1:
        num_workers = max(1, min(cpu_count(), 8))
    elif num_workers == 0:
        num_workers = 1

    # Language model for batch embedding
    lang_model = load_sentence_transformer_model()

    entries: list[dict] = []
    processed = 0  # number of streaming samples actually flushed/processed
    default_batch_size = 64
    batch_size = default_batch_size if (max_trajectories is None) else min(default_batch_size, max_trajectories)
    batch_samples: list[dict[str, Any]] = []
    batch_records: list[tuple[str, dict]] = []

    # Simple live stats
    seen_samples = 0
    skipped_camera = 0
    skipped_no_record = 0
    skipped_no_mp4 = 0

    def flush_batch():
        nonlocal entries, processed, batch_samples, batch_records
        if not batch_samples:
            return

        # Collect unique texts and encode once
        unique_texts = _collect_unique_texts_for_batch(batch_records)
        emb_map = _encode_texts(unique_texts, lang_model)

        if num_workers == 1:
            for sample in tqdm(batch_samples, desc="Batch (seq)", leave=False):
                res = _process_single_stream_sample(
                    sample=sample,
                    embeddings=emb_map,
                    output_dir=output_dir,
                    dataset_name=dataset_label,
                    max_frames=max_frames,
                    fps=fps,
                )
                # res is a list; extend and update decode_fail if nothing produced due to decode error
                entries.extend(res)
        else:
            with Pool(processes=num_workers) as pool:
                worker = partial(
                    _process_single_stream_sample,
                    embeddings=emb_map,
                    output_dir=output_dir,
                    dataset_name=dataset_label,
                    max_frames=max_frames,
                    fps=fps,
                )
                for res in tqdm(
                    pool.imap_unordered(worker, batch_samples),
                    total=len(batch_samples),
                    desc=f"Batch (workers={num_workers})",
                    leave=False,
                ):
                    entries.extend(res)

        processed += len(batch_samples)
        batch_samples = []
        batch_records = []

    print(f"Streaming {dataset_name}; workers={num_workers}, batch_size={batch_size}")
    stream_pbar = tqdm(desc="Streaming samples", unit="sample", dynamic_ncols=True)

    for sample in ds:
        if max_trajectories is not None and processed >= max_trajectories:
            break

        key = sample.get("__key__", "")
        episode_id, camera = _parse_episode_and_camera(key)
        seen_samples += 1
        stream_pbar.update(1)
        if not camera or camera not in CAMERA_KEYS:
            skipped_camera += 1
            continue

        # Ensure episode record exists; gather for embedding planning
        try:
            _json_path, rec = get_episode_record(episode_id)
        except Exception:
            skipped_no_record += 1
            continue

        # Require mp4 content; if absent (e.g., png-only shard), skip early
        if not sample.get("mp4"):
            skipped_no_mp4 += 1
            continue

        batch_samples.append(sample)
        batch_records.append((episode_id, rec))

        if len(batch_samples) >= batch_size:
            flush_batch()

        # If user asked for a very small number, don't wait for another full batch
        if max_trajectories is not None and (processed + len(batch_samples)) >= max_trajectories:
            flush_batch()
            break

    # Final flush
    flush_batch()
    stream_pbar.close()

    # Build HF dataset from entries
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

    # datasets can infer features; rely on default
    print(
        f"Done. seen={seen_samples}, entries={len(entries)}, "
        f"skipped_camera={skipped_camera}, skipped_no_record={skipped_no_record}, "
        f"skipped_no_mp4={skipped_no_mp4}"
    )
    return Dataset.from_list(entries)


def load_agibotworld_dataset(dataset_name_or_path: str, max_trajectories: int = 100) -> dict[str, list[dict]]:
    """Load AgiBotWorld dataset using HuggingFace streaming and extract head_color.mp4 files.

    Args:
        dataset_name_or_path: HuggingFace dataset name (e.g. "agibot-world/AgiBotWorld-Alpha")
                             or local path to the dataset

    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """

    print(f"Loading AgiBotWorld dataset from: {dataset_name_or_path}")
    print("=" * 100)
    print("LOADING AGIBOTWORLD DATASET")
    print("=" * 100)

    task_data = {}

    # Check if it's a local path or HuggingFace dataset name
    if os.path.exists(dataset_name_or_path):
        # Local dataset
        task_data = _load_local_agibotworld(dataset_name_or_path, max_trajectories)
    else:
        # HuggingFace dataset - use streaming
        task_data = _load_streaming_agibotworld(dataset_name_or_path, max_trajectories)

    print(
        f"Loaded {sum(len(trajectories) for trajectories in task_data.values())} trajectories from {len(task_data)} tasks"
    )
    return task_data


# NOTE: As the dataset is too large, we did not test this function extensively and it may be out of date.
def _load_local_agibotworld(base_path: str, max_trajectories: int = 100, max_frames: int = 32) -> dict[str, list[dict]]:
    """Load AgiBotWorld dataset from local files, starting with task_info JSON files."""
    base_path = Path(base_path)
    task_data = {}

    # Define required directories
    observations_dir = base_path / "observations"
    task_info_dir = base_path / "task_info"
    proprio_stats_dir = base_path / "proprio_stats"

    if not observations_dir.exists():
        raise FileNotFoundError(f"Observations directory not found: {observations_dir}")
    if not task_info_dir.exists():
        raise FileNotFoundError(f"Task info directory not found: {task_info_dir}")

    # Start by iterating over task_info JSON files to get proper task names
    task_info_files = list(task_info_dir.glob("*.json"))

    if not task_info_files:
        raise FileNotFoundError(f"No task info JSON files found in: {task_info_dir}")

    print(f"Found {len(task_info_files)} task info files")

    total_trajectories = 0

    for task_info_file in tqdm(task_info_files, desc="Processing tasks"):
        if total_trajectories >= max_trajectories:
            print(f"Reached max_trajectories limit ({max_trajectories}), stopping...")
            break

        # Extract task ID from filename (e.g., "task_392.json" -> "392")
        task_id = task_info_file.stem.replace("task_", "")

        # Load task information from JSON
        task_info = _load_task_info(task_info_file)

        if not task_info:
            print(f"Skipping task {task_id} - no valid task info")
            continue

        # Extract proper task name from first episode (they should all have the same task)
        if task_info and len(task_info) > 0:
            first_episode = task_info[0]
            task_name = first_episode.get("task_name", f"Task {task_id}")
            first_episode.get("task_description", f"AgiBotWorld Task {task_id}")
        else:
            task_name = f"Task {task_id}"

        print(f"Processing task {task_id}: '{task_name}'")

        # Get the corresponding task directory
        task_dir = observations_dir / task_id
        if not task_dir.exists():
            print(f"Task directory not found: {task_dir}, skipping...")
            continue

        trajectories = []

        # Process episodes based on the information in task_info JSON
        for episode_info in task_info:
            if total_trajectories >= max_trajectories:
                break

            episode_id = str(episode_info.get("episode_id", ""))
            if not episode_id:
                continue

            # Check if episode directory exists
            episode_dir = task_dir / episode_id
            if not episode_dir.exists():
                print(f"Episode directory not found: {episode_dir}, skipping episode {episode_id}")
                continue

            # Look for head_color.mp4 file
            videos_dir = episode_dir / "videos"
            head_color_video = videos_dir / "head_color.mp4"

            if head_color_video.exists():
                # Load proprioceptive data
                proprio_file = proprio_stats_dir / task_id / episode_id / "proprio_stats.h5"
                actions = _load_actions_from_h5(proprio_file)

                # Process video: resize to 256x256 and downsample frames
                try:
                    processed_frames = load_video_frames(head_color_video)

                    trajectory = {
                        "frames": processed_frames,  # Processed video frames
                        "actions": actions,
                        "is_robot": True,  # AgiBotWorld is robot data
                        "task": task_name,  # Use the descriptive task name from JSON
                        "optimal": "optimal",  # Assume all AgiBotWorld trajectories are optimal
                    }
                except Exception as e:
                    print(f"  ❌ Failed to process video {head_color_video}: {e}")
                    continue

                trajectories.append(trajectory)
                total_trajectories += 1

                print(f"  ✅ Loaded episode {episode_id} ({total_trajectories}/{max_trajectories})")
            else:
                print(f"  ❌ head_color.mp4 not found for episode {episode_id}")

        if trajectories:
            # Use proper task name from JSON instead of generic "task_{id}"
            task_data[task_name] = trajectories
            print(f"Added {len(trajectories)} trajectories for task '{task_name}'")

    print(f"Loaded {total_trajectories} total trajectories from {len(task_data)} tasks")
    return task_data


def _load_streaming_agibotworld(dataset_name: str, max_trajectories: int = 100) -> dict[str, list[dict]]:
    """Legacy helper no longer used. Kept for compatibility."""
    raise NotImplementedError("Use convert_agibotworld_streaming_to_hf() for streaming conversion.")


def _load_task_info(task_info_file: Path) -> list[dict]:
    """Load task information from JSON file."""
    if not task_info_file.exists():
        print(f"Task info file not found: {task_info_file}")
        return []

    try:
        with open(task_info_file) as f:
            task_info = json.load(f)
        return task_info if isinstance(task_info, list) else [task_info]
    except Exception as e:
        print(f"Error loading task info from {task_info_file}: {e}")
        return []


def _load_actions_from_h5(proprio_file: Path) -> np.ndarray:
    """Load actions from proprioceptive H5 file."""
    if not proprio_file.exists():
        print(f"Proprioceptive file not found: {proprio_file}")
        return np.array([])

    try:
        with h5py.File(proprio_file, "r") as f:
            # According to AgiBotWorld docs, actions are stored under /action
            if "action" in f:
                action_group = f["action"]

                # Try to extract joint actions (most common for manipulation)
                if "joint" in action_group and "position" in action_group["joint"]:
                    actions = action_group["joint"]["position"][:]
                elif "end" in action_group and "position" in action_group["end"]:
                    # Use end-effector positions if joint positions not available
                    end_positions = action_group["end"]["position"][:]
                    end_orientations = (
                        action_group["end"]["orientation"][:] if "orientation" in action_group["end"] else None
                    )

                    if end_orientations is not None:
                        # Concatenate position and orientation for full 6-DOF actions
                        # Reshape orientations from [N, 2, 4] to [N, 8] (both arms)
                        end_orientations_flat = end_orientations.reshape(end_orientations.shape[0], -1)
                        # Reshape positions from [N, 2, 3] to [N, 6]
                        end_positions_flat = end_positions.reshape(end_positions.shape[0], -1)
                        actions = np.concatenate([end_positions_flat, end_orientations_flat], axis=1)
                    else:
                        actions = end_positions.reshape(end_positions.shape[0], -1)
                else:
                    print(f"No recognizable action data found in {proprio_file}")
                    return np.array([])

                return actions
            else:
                print(f"No action group found in {proprio_file}")
                return np.array([])

    except Exception as e:
        print(f"Error loading actions from {proprio_file}: {e}")
        return np.array([])
