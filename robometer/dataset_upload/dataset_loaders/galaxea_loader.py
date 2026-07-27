import os
import gc
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, concatenate_datasets

from dataset_upload.helpers import (
    create_hf_trajectory,
    generate_unique_id,
    load_sentence_transformer_model,
)
from tqdm import tqdm

# Disable GPUs for TensorFlow in this loader to avoid CUDA context issues in workers
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow_datasets as tfds


def _stable_shard_for_index(index: int, shard_modulus: int = 1000) -> str:
    try:
        idx = int(index)
    except Exception:
        idx = abs(hash(str(index)))
    shard_index = idx // shard_modulus
    return f"shard_{shard_index:04d}"


def _build_galaxea_video_paths(
    output_dir: str,
    dataset_label: str,
    episode_idx: int,
    view_key: str,
) -> tuple[str, str]:
    shard_dir = _stable_shard_for_index(episode_idx)
    episode_dir = os.path.join(output_dir, dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}")
    os.makedirs(episode_dir, exist_ok=True)
    filename = f"clip@{view_key}.mp4"
    full_path = os.path.join(episode_dir, filename)
    rel_path = os.path.join(dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}", filename)
    return full_path, rel_path


def _parse_low_level_english(instruction: bytes | str) -> str | None:
    """Galaxea's language_instruction format: "high@low_cn@low_en". Return low_en."""
    try:
        instruction = instruction.decode("utf-8")
        parts = instruction.split("@")
        if len(parts) >= 3:
            return parts[2].strip()
        # If not delimited, return as-is
        return instruction.strip()
    except Exception:
        return None


def _process_single_galaxea_episode(args):
    episode, ep_idx, task, lang_vec, output_dir, dataset_name, max_frames, fps, valid_img_keys = args

    episode_entries = []
    first_step = next(episode)
    assert len(valid_img_keys) == 1, (
        "Galaxea only has one valid image key for now. No support for multiple because of the way we iterate over the episode."
    )
    for img_key in valid_img_keys:
        # Validate key presence
        if img_key not in first_step["observation"]:
            continue
        # Prune trivial black frames
        if np.all(first_step["observation"][img_key] == 0):
            continue

        frames = [first_step["observation"][img_key]] + [
            s["observation"][img_key] for s in episode if img_key in s["observation"]
        ]
        if not frames:
            continue
        # skip anything > 800 frames for now because memory usage
        elif len(frames) > 1000:
            print(f"Skipping episode {ep_idx} because it's too long, length is {len(frames)}")
            del frames
            continue

        full_path, rel_path = _build_galaxea_video_paths(
            output_dir=output_dir,
            dataset_label=dataset_name,
            episode_idx=ep_idx,
            view_key=img_key,
        )

        # Pass frames as list avoid doubling memory from np.stack
        traj_dict = {
            "id": generate_unique_id(),
            "frames": frames,  # Pass as list, let create_hf_trajectory handle it
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
            continue
        if entry:
            entry["frames"] = rel_path
            episode_entries.append(entry)
        del frames

    return episode_entries


def convert_galaxea_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
) -> Dataset:
    """Convert Galaxea RLDS datasets to HF format by writing videos directly (OXE-style).

    Args:
        dataset_path: Root path that contains an 'rlds' directory with builders.
        dataset_name: Name to tag the resulting dataset (e.g., 'galaxea').
        output_dir: Where to write video files and dataset.
        max_trajectories: Limit number of produced trajectories (None/-1 for all).
        max_frames: Max frames per video.
        fps: Video fps.
    """

    # Normalize and checks
    if dataset_name is None:
        raise ValueError("dataset_name is required")

    root = Path(os.path.expanduser(dataset_path)) / "rlds"
    if not root.exists():
        raise FileNotFoundError(f"'rlds' directory not found under: {dataset_path}")

    # Determine workers
    if num_workers == -1:
        num_workers = min(cpu_count(), 8)
    elif num_workers == 0:
        num_workers = 1

    # Language model and cache
    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    rlds_name = dataset_name.replace("galaxea_", "")

    # Find builder directory/version: root/rlds_name/<version>
    ds_root = root / rlds_name
    versions = os.listdir(str(ds_root)) if ds_root.exists() else []
    if len(versions) == 0:
        raise ValueError(f"No versions found for {rlds_name} in {ds_root}")

    builder = None
    for version in versions:
        if "incomplete" in version:
            continue
        try:
            builder = tfds.builder_from_directory(f"{ds_root}/{version}")
            break
        except Exception:
            continue
    if builder is None:
        raise ValueError(f"No valid builder found for {rlds_name} in {ds_root}")

    # to keep memory usage low, use 1 worker for decoding and interleave files
    dataset = builder.as_dataset(split="train", shuffle_files=False)

    # Determine valid image observation keys for Galaxea (head and both wrists)
    valid_img_keys = [
        "image_camera_head",
    ]

    # Batch/process episodes
    batch_size = 1
    num_workers = min(num_workers, 1)
    entries: list[dict[str, Any]] = []
    produced = 0
    max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)
    episode_batch = []
    info_batch = []

    # split up
    for ep_idx, episode in enumerate(tqdm(dataset, desc=f"Processing {rlds_name} episodes")):
        if produced >= max_limit:
            break

        # Materialize first step for language instruction
        try:
            first_step = next(iter(tfds.as_numpy(episode["steps"])))
        except StopIteration:
            continue

        # Galaxea stores 'language_instruction' at step-level; parse low-level English
        task = None
        if "language_instruction" in first_step:
            task = _parse_low_level_english(first_step["language_instruction"])  # type: ignore[index]
        if not task:
            continue

        # Precompute embedding
        if task not in lang_cache:
            lang_cache[task] = lang_model.encode(task)
        lang_vec = lang_cache[task]

        # Convert episode to numpy (list of steps)
        try:
            # episode_np = list(tfds.as_numpy(episode["steps"]))
            episode_np = iter(tfds.as_numpy(episode["steps"]))
        except Exception as e:
            print(f"Warning: Failed to convert episode {ep_idx} to numpy: {e}")
            continue

        episode_batch.append(episode_np)
        info_batch.append((ep_idx, task, lang_vec))

        if len(episode_batch) >= batch_size or ep_idx + 1 == len(dataset):
            if num_workers == 1:
                for args in zip(
                    episode_batch,
                    [i for (i, _, _) in info_batch],
                    [t for (_, t, _) in info_batch],
                    [v for (_, _, v) in info_batch],
                    [output_dir] * len(episode_batch),
                    [dataset_name] * len(episode_batch),
                    [max_frames] * len(episode_batch),
                    [fps] * len(episode_batch),
                    [valid_img_keys] * len(episode_batch),
                    strict=False,
                ):
                    episode_entries = _process_single_galaxea_episode(args)
                    entries.extend(episode_entries)
                    produced += len(episode_entries)
                    if produced >= max_limit:
                        break
            else:
                raise ValueError("num_workers > 1 not supported for Galaxea due to the way the frame loader works.")
                # from multiprocessing import Pool

                # worker_args = list(
                #    zip(
                #        episode_batch,
                #        [i for (i, _, _) in info_batch],
                #        [t for (_, t, _) in info_batch],
                #        [v for (_, _, v) in info_batch],
                #        [output_dir] * len(episode_batch),
                #        [dataset_name] * len(episode_batch),
                #        [rlds_name] * len(episode_batch),
                #        [max_frames] * len(episode_batch),
                #        [fps] * len(episode_batch),
                #        [valid_img_keys] * len(episode_batch),
                #        strict=False,
                #    )
                # )

                # with Pool(processes=num_workers) as pool:
                #    results = list(
                #        tqdm(
                #            pool.imap_unordered(_process_single_galaxea_episode, worker_args),
                #            total=len(worker_args),
                #            desc=f"Processing batch (workers={num_workers})",
                #        )
                #    )
                # for res in results:
                #    entries.extend(res)
                #    produced += len(res)
                #    if produced >= max_limit:
                #        break

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
        })
    return Dataset.from_list(entries)
