import io
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm

from dataset_upload.helpers import (
    create_hf_trajectory,
    generate_unique_id,
    load_sentence_transformer_model,
)


def _stable_shard_for_index(index: int, shard_modulus: int = 1000) -> str:
    try:
        idx = int(index)
    except Exception:
        idx = abs(hash(str(index)))
    shard_index = idx // shard_modulus
    return f"shard_{shard_index:04d}"


def _build_molmo_video_paths(
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


def _to_rgb_numpy(img_cell: Any) -> Optional[np.ndarray]:
    """Convert a datasets Image cell (dict with bytes/path, PIL.Image, or np.ndarray) to RGB uint8 ndarray."""
    if img_cell is None:
        return None
    # Already numpy HxWxC
    if isinstance(img_cell, np.ndarray):
        if img_cell.ndim == 3 and img_cell.shape[-1] in (1, 3, 4):
            if img_cell.shape[-1] == 1:
                img_cell = np.repeat(img_cell, 3, axis=-1)
            elif img_cell.shape[-1] == 4:
                img_cell = img_cell[..., :3]
            if img_cell.dtype != np.uint8:
                img_cell = img_cell.astype(np.uint8, copy=False)
            return img_cell
        return None
    # PIL
    if isinstance(img_cell, Image.Image):
        return np.asarray(img_cell.convert("RGB"), dtype=np.uint8)
    # dict with bytes
    if isinstance(img_cell, dict):
        data = img_cell.get("bytes")
        if data is None:
            path = img_cell.get("path")
            if path and os.path.exists(path):
                with Image.open(path) as im:
                    return np.asarray(im.convert("RGB"), dtype=np.uint8)
            return None
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    # Unknown
    return None


def convert_molmoact_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
) -> Dataset:
    """Stream MolmoAct LeRobot (parquet) and convert to HF, using episodes.jsonl for task text.

    Assumes dataset_path contains one or more subdirectories, each with parquet files and an
    associated episodes.jsonl. We iterate per subdirectory to avoid episode_index collisions,
    grouping rows by `episode_index` and writing videos for `first_view`, `second_view`, and `wrist_image`.
    """

    root = Path(os.path.expanduser(dataset_path)) / dataset_name
    if not root.exists():
        raise FileNotFoundError(f"MolmoAct dataset path not found: {root}")

    # Discover dataset subdirectories that have episodes.jsonl; if none, fallback to root
    assert (root / "train" / "meta" / "episodes.jsonl").exists(), "episodes.jsonl not found"

    # Language model and cache
    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    entries: list[dict] = []
    produced = 0
    max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)

    def load_episode_text_map(ds_dir: Path) -> dict[int, str]:
        mapping: dict[int, str] = {}
        jsonl_path = ds_dir / "train" / "meta" / "episodes.jsonl"
        if not jsonl_path.exists():
            return mapping
        try:
            import json

            with open(jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    ep_idx = obj.get("episode_index")
                    if ep_idx is None:
                        ep_idx = obj.get("index")
                    if ep_idx is None:
                        continue
                    text = (obj.get("tasks"))[0]
                    if isinstance(text, str) and text.strip():
                        mapping[int(ep_idx)] = text.strip()
        except Exception:
            pass
        return mapping

    def flush_episode(ep_idx: int, task_text: str, label: str, frames_by_view: dict[str, list[np.ndarray]]) -> None:
        nonlocal produced, entries
        if not frames_by_view:
            return
        if task_text not in lang_cache:
            lang_cache[task_text] = lang_model.encode(task_text)
        lang_vec = lang_cache[task_text]

        for view_key, frames in frames_by_view.items():
            if not frames:
                continue
            if isinstance(frames[0], np.ndarray) and np.all(frames[0] == 0):
                continue

            full_path, rel_path = _build_molmo_video_paths(
                output_dir=output_dir,
                dataset_label=label,
                episode_idx=ep_idx,
                view_key=view_key,
            )

            traj_dict = {
                "id": generate_unique_id(),
                "frames": frames,
                "task": task_text,
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
                dataset_name=dataset_name,
                use_video=True,
                fps=fps,
            )
            if entry:
                entry["frames"] = rel_path
                entries.append(entry)
                produced += 1

    # Process each dataset directory independently to avoid ep-index collisions
    ep_text_map = load_episode_text_map(root)

    # Discover parquet files in ds_dir
    data_files: list[str] = []
    for pat in ("**/*.parquet", "*.parquet"):
        data_files.extend([str(p) for p in root.glob(pat)])
    if not data_files:
        raise ValueError("No parquet files found")

    ds_iter = load_dataset(
        "parquet",
        data_files={"train": data_files},
        split="train",
        streaming=True,
    )

    current_ep: Optional[int] = None
    frames_by_view: dict[str, list[np.ndarray]] = {}
    label = f"{dataset_name}"

    for row in tqdm(ds_iter, desc=f"MolmoAct rows ({dataset_name})"):
        if produced >= max_limit:
            break
        ep_idx = int(row.get("episode_index", -1))
        if ep_idx < 0:
            continue

        if current_ep is None:
            current_ep = ep_idx
            frames_by_view = {"first_view": [], "second_view": []}
        elif ep_idx != current_ep:
            task_text = ep_text_map.get(current_ep)
            print(f"{task_text} episode loaded")
            flush_episode(current_ep, task_text, label, frames_by_view)
            current_ep = ep_idx
            frames_by_view = {"first_view": [], "second_view": []}

        for view_key in ("first_view", "second_view"):
            cell = row.get(view_key)
            img = _to_rgb_numpy(cell)
            if img is not None:
                frames_by_view[view_key].append(img)

        if produced >= max_limit:
            break

    if current_ep is not None and produced < max_limit:
        task_text = ep_text_map.get(current_ep)
        print(f"{task_text} episode loaded")
        flush_episode(current_ep, task_text, label, frames_by_view)

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
