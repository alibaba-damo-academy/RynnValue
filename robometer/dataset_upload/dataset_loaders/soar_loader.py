import os
import json
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

from dataset_upload.helpers import (
    create_hf_trajectory,
    generate_unique_id,
    load_sentence_transformer_model,
)

# We do not stream; assume RLDS TFDS builders are already downloaded locally.
import tensorflow_datasets as tfds

# soar_new_success_labels_path = "dataset_upload/dataset_helpers/soar_vlm_labels_checkpoint.json"
soar_new_success_labels_path = "dataset_upload/dataset_helpers/soar_label_corrections_full.json"


def _build_video_paths(output_dir: str, dataset_label: str, episode_idx: int, view_key: str) -> tuple[str, str]:
    shard_index = episode_idx // 1000
    shard_dir = f"shard_{shard_index:04d}"
    episode_dir = os.path.join(output_dir, dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}")
    os.makedirs(episode_dir, exist_ok=True)
    filename = f"clip@{view_key}.mp4"
    full_path = os.path.join(episode_dir, filename)
    rel_path = os.path.join(dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}", filename)
    return full_path, rel_path


def _process_episode(args):
    episode_steps, ep_idx, task, lang_vec, output_dir, dataset_label, max_frames, fps, img_key, quality_label = args

    # Collect frames for the given image key
    frames: list[np.ndarray] = []
    for step in episode_steps:
        obs = step.get("observation", {}) if isinstance(step, dict) else {}
        if img_key not in obs:
            continue
        frame = obs[img_key]
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[-1] in (1, 3, 4):
                if frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                elif frame.shape[-1] == 4:
                    frame = frame[..., :3]
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8, copy=False)
                frames.append(frame)

    if not frames:
        return []

    full_path, rel_path = _build_video_paths(output_dir, dataset_label, ep_idx, img_key)

    traj_dict = {
        "id": generate_unique_id(),
        "frames": frames,
        "task": task,
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
        return [entry]
    return []


def convert_soar_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
) -> Dataset:
    """Convert SOAR RLDS (local TFDS) to HF dataset. Non-streaming, local builders only.

    Expects directory structure:
      <dataset_path>/rlds/<split>/<version>/ (TFDS builder dir)
    """

    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"'rlds' directory not found under: {dataset_path}")

    if num_workers == -1:
        num_workers = min(cpu_count(), 8)
    elif num_workers == 0:
        num_workers = 1

    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    label_corrections = {}
    if os.path.exists(soar_new_success_labels_path):
        with open(soar_new_success_labels_path) as f:
            data = json.load(f)
        # Keys are strings in JSON, convert to int
        label_corrections = {int(k): v for k, v in data.get("label_corrections", {}).items()}
        print(f"Loaded label corrections for {len(label_corrections)} trajectories")

    datasets_list: list[Dataset] = []

    builder = tfds.builder_from_directory(root)
    success_episode_instructions = set()  # to only upload failures that have a corresponding success episode
    for split_name in ["success", "failure"]:
        ds = builder.as_dataset(split=split_name, shuffle_files=False)

        if split_name == "success":
            with open(soar_new_success_labels_path, "r") as f:
                # new_success_labels = json.load(f)["results"]
                new_success_labels = json.load(f)["label_corrections"]
                # episodes where qwen-3-vl predicted success
                # new_success_labels = [
                #    result["predicted_label"] for result in new_success_labels if result["original_label"] == "success"
                # ]

                # convert to int keys
                new_success_labels = {int(k): v for k, v in new_success_labels.items()}

        entries: list[dict] = []
        produced = 0
        max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)

        for ep_idx, episode in enumerate(tqdm(ds, desc=f"SOAR {split_name} episodes")):
            if split_name == "success":
                if new_success_labels[ep_idx] != "successful":
                    # disagree with qwen-3-vl's prediction, skip this episode
                    continue
            if produced >= max_limit:
                break

            # Convert to numpy steps list
            try:
                steps_np = list(tfds.as_numpy(episode["steps"]))
            except Exception:
                continue

            # Extract language instruction from first step
            task_text: str | None = None
            first = steps_np[0] if steps_np else None
            if first is not None:
                # First try step-level keys
                if "language_instruction" in first:
                    val = first["language_instruction"]
                    task_text = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)

            if not task_text:
                continue
            elif split_name == "failure":
                if new_success_labels[ep_idx] != "failure":  # skip if the label is not correct
                    continue
                elif task_text not in success_episode_instructions:
                    # no corresponding success episode, skip this failure
                    print(f"No corresponding success episode for failure {ep_idx}, skipping")
                    continue

            if task_text not in lang_cache:
                lang_cache[task_text] = lang_model.encode(task_text)
            lang_vec = lang_cache[task_text]

            # Choose a valid image key
            valid_img_key: str | None = None
            valid_img_key = "image_0"

            # Determine quality label
            quality_label = "successful" if split_name.lower().startswith("success") else "failure"

            # Build entry for this view
            episode_entries = _process_episode((
                steps_np,
                ep_idx,
                task_text,
                lang_vec,
                output_dir,
                dataset_name,
                max_frames,
                fps,
                valid_img_key,
                quality_label,
            ))
            entries.extend(episode_entries)
            produced += len(episode_entries)

        if not entries:
            ds_out = Dataset.from_dict({
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
        else:
            ds_out = Dataset.from_list(entries)

        datasets_list.append(ds_out)

    if not datasets_list:
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

    if len(datasets_list) == 1:
        return datasets_list[0]
    return concatenate_datasets(datasets_list)
