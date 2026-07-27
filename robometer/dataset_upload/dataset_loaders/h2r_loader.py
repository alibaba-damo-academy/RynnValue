#!/usr/bin/env python3
"""
H2R dataset loader for the generic dataset converter for Robometer model training.
https://huggingface.co/datasets/dannyXSC/HumanAndRobot
Human2Robot: Learning Robot Actions from Paired Human-Robot Videos
This module contains H2R-specific logic for loading and processing HDF5 files.

Updated to support OXE-style streaming conversion: write videos and build
HF entries on the fly, and return a ready `datasets.Dataset` to be pushed
or saved by the caller.
"""

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from dataset_upload.helpers import (
    create_hf_trajectory,
    generate_unique_id,
    load_sentence_transformer_model,
)
from tqdm import tqdm
from datasets import Dataset


class H2RFrameLoader:
    """Pickle-able loader that reads H2R frames from an HDF5 dataset on demand.

    Stores only simple fields so it can be safely passed across processes.
    """

    def __init__(self, hdf5_path: str, convert_to_rgb: bool = True):
        self.hdf5_path = hdf5_path
        self.convert_to_rgb = convert_to_rgb

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        """Load frames from HDF5 when called.

        Returns:
            np.ndarray of shape (T, H, W, 3), dtype uint8
        """
        with h5py.File(self.hdf5_path, "r") as f:
            human_frames = f["/cam_data/human_camera"][:]
            robot_frames = f["/cam_data/robot_camera"][:]

        if self.convert_to_rgb:
            human_frames = human_frames[..., [2, 1, 0]]
            robot_frames = robot_frames[..., [2, 1, 0]]

        # Ensure shape and dtype sanity
        if not isinstance(human_frames, np.ndarray) or human_frames.ndim != 4 or human_frames.shape[-1] != 3:
            raise ValueError(f"Unexpected frames shape for {self.hdf5_path}: {getattr(human_frames, 'shape', None)}")

        if not isinstance(robot_frames, np.ndarray) or robot_frames.ndim != 4 or robot_frames.shape[-1] != 3:
            raise ValueError(f"Unexpected frames shape for {self.hdf5_path}: {getattr(robot_frames, 'shape', None)}")

        # Ensure uint8
        if human_frames.dtype != np.uint8:
            human_frames = human_frames.astype(np.uint8, copy=False)
        if robot_frames.dtype != np.uint8:
            robot_frames = robot_frames.astype(np.uint8, copy=False)

        return human_frames, robot_frames


# Task mapping from folder names to task descriptions
FOLDER_TO_TASK_NAME = {
    "grab_both_cubes_v1": "pick up each cube individually and place them onto the plate",
    "grab_cube2_v1": "pick up the cube and place it onto the plate",
    "grab_cup_v1": "move the cup from left to right",
    "grab_pencil1_v1": "pick up the marker and place it on the plate",
    "grab_pencil2_v1": "pick up the marker and place it on the plate",
    "grab_pencil_v1": "pick up the marker and place it on the plate",
    "grab_two_cubes2_v1": "pick up the green cube and place it onto the plate",
    "grab_to_plate1_and_back_v1": "put the red cube on the darker plate",
    "grab_to_plate1_v1": "pick up the red cube and place it onto the darker plate",
    "grab_to_plate2_v1": "pick up the red cube and place it onto the lighter plate",
    "grab_to_plate2_and_back_v1": "put the red cube on the yellow plate",
    "grab_to_plate2_and_pull_v1": "put the cube on the plate, then pull the plate from bottom to top",
    "pull_plate_grab_cube": "pull the plate from bottom to top, then pick up the cube and place it onto the plate",
    "pull_plate_v1": "pull the plate from bottom to top",
    "push_box_common_v1": "push the box from left to right",
    "push_box_random_v1": "push the box from left to right",
    "push_box_two_v1": "push the tissues from left to right",
    "push_plate_v1": "push the plate from top to bottom",
    # "roll": "pick up the brush and write on the table", # skipped because it's weird
    # "writing": "write aimlessly on the desk", # skipped because writing aimlessly is not helpful for reward modeling
}


def _get_task_name_from_folder(folder_name: str) -> str:
    """Convert folder name to task name using the mapping."""
    # First try to find exact match
    if folder_name in FOLDER_TO_TASK_NAME:
        return FOLDER_TO_TASK_NAME[folder_name]
    else:
        return None


def _discover_h2r_files(dataset_path: Path) -> list[tuple[Path, str]]:
    """Discover all video files in the H2R dataset structure.

    Expected structure:
    dataset_path/
        folder_name_1/
            hdf5_file_1.hdf5
            hdf5_file_2.hdf5
            hdf5_file_3.hdf5
            ...
        folder_name_2/
            hdf5_file_1.hdf5
            hdf5_file_2.hdf5
            hdf5_file_3.hdf5
            ...
        ...

    Returns:
        List of tuples: (hdf5_file_path, task_name)
    """
    trajectory_files: list[tuple[Path, str]] = []
    for folder in dataset_path.iterdir():
        if folder.is_dir():
            for file in folder.glob("*.hdf5"):
                trajectory_files.append((file, folder.name))

    return trajectory_files


def _stable_shard_for_index(index: int, shard_modulus: int = 1000) -> str:
    """Deterministically bucket an index into a shard directory name.

    Matches the naming style used in the OXE loader for consistent layout.
    """
    try:
        idx = int(index)
    except Exception:
        idx = abs(hash(str(index)))
    shard_index = idx // shard_modulus
    return f"shard_{shard_index:04d}"


def _build_h2r_video_paths(
    output_dir: str,
    dataset_label: str,
    episode_idx: int,
    role: str,
) -> tuple[str, str]:
    shard_dir = _stable_shard_for_index(episode_idx)
    episode_dir = os.path.join(output_dir, dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}")
    os.makedirs(episode_dir, exist_ok=True)
    filename = f"clip@{role}.mp4"
    full_path = os.path.join(episode_dir, filename)
    rel_path = os.path.join(dataset_label.lower(), shard_dir, f"episode_{episode_idx:06d}", filename)
    return full_path, rel_path


def _process_single_h2r_file(args):
    """Worker to process a single H2R HDF5 file into up to two entries.

    Returns a list of entries (human and/or robot), each with relative frame paths.
    """
    (
        file_path,
        folder_name,
        ep_idx,
        dataset_name,
        output_dir,
        max_frames,
        fps,
        task,
        lang_vec,
    ) = args

    entries: list[dict[str, Any]] = []

    # Load frames for this file (human and robot)
    human_frames, robot_frames = H2RFrameLoader(str(file_path))()

    # HUMAN entry
    full_h_path, rel_h_path = _build_h2r_video_paths(
        output_dir=output_dir,
        dataset_label=dataset_name,
        episode_idx=ep_idx,
        role="human",
    )
    human_traj = {
        "id": generate_unique_id(),
        "frames": human_frames,
        "task": task,
        "is_robot": False,
        "quality_label": "successful",
        "preference_group_id": None,
        "preference_rank": None,
    }
    human_entry = create_hf_trajectory(
        traj_dict=human_traj,
        video_path=full_h_path,
        lang_vector=lang_vec,
        max_frames=max_frames,
        dataset_name=dataset_name,
        use_video=True,
        fps=fps,
    )
    if human_entry:
        human_entry["frames"] = rel_h_path
        entries.append(human_entry)

    # ROBOT entry
    full_r_path, rel_r_path = _build_h2r_video_paths(
        output_dir=output_dir,
        dataset_label=dataset_name,
        episode_idx=ep_idx,
        role="robot",
    )
    robot_traj = {
        "id": generate_unique_id(),
        "frames": robot_frames,
        "task": task,
        "is_robot": True,
        "quality_label": "successful",
        "preference_group_id": None,
        "preference_rank": None,
    }
    robot_entry = create_hf_trajectory(
        traj_dict=robot_traj,
        video_path=full_r_path,
        lang_vector=lang_vec,
        max_frames=max_frames,
        dataset_name=dataset_name,
        use_video=True,
        fps=fps,
    )
    if robot_entry:
        robot_entry["frames"] = rel_r_path
        entries.append(robot_entry)

    return entries


def convert_h2r_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
) -> Dataset:
    """Convert the H2R dataset to HF format by writing videos directly.

    This mirrors the OXE loader's streaming approach: iterate files, write videos,
    assemble entries, and return a Dataset at the end.
    """

    if dataset_name is None:
        raise ValueError("dataset_name is required")

    base_path = Path(dataset_path)
    if not base_path.exists():
        raise FileNotFoundError(f"H2R dataset path not found: {base_path}")

    discovered = _discover_h2r_files(base_path)
    if len(discovered) == 0:
        # Return an empty dataset with expected columns
        return Dataset.from_dict({
            "id": [],
            "task": [],
            "lang_vector": [],
            "data_source": [],
            "frames": [],
            "is_robot": [],
            "quality_label": [],
            # keep schema compatible with helpers/create_hf_trajectory usage
            "preference_group_id": [],
            "preference_rank": [],
        })

    # Language model and cache (avoid recomputing for identical tasks)
    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    # Determine workers and batching (match OXE approach to control memory)
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
    produced_pairs = 0  # count by file; each file can produce up to 2 entries
    max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)

    print(f"Found {len(discovered)} HDF5 files; processing in batches of {batch_size} with {num_workers} workers...")

    # Process files in batches
    file_batch: list[tuple[Path, str]] = []
    info_batch: list[tuple[int, str, Any]] = []  # (ep_idx, task, lang_vec)

    for ep_idx, (file_path, folder_name) in enumerate(tqdm(discovered, desc="Queuing H2R files")):
        if produced_pairs >= max_limit:
            break

        task = _get_task_name_from_folder(folder_name)
        if task is None:
            print("Skipping file: ", file_path)
            continue
        if task not in lang_cache:
            lang_cache[task] = lang_model.encode(task)
        lang_vec = lang_cache[task]

        file_batch.append((file_path, folder_name))
        info_batch.append((ep_idx, task, lang_vec))

        if len(file_batch) >= batch_size or ep_idx + 1 == len(discovered):
            # Build worker args
            worker_args = list(
                zip(
                    [f for (f, _) in file_batch],
                    [fn for (_, fn) in file_batch],
                    [i for (i, _, _) in info_batch],
                    [dataset_name] * len(file_batch),
                    [output_dir] * len(file_batch),
                    [max_frames] * len(file_batch),
                    [fps] * len(file_batch),
                    [t for (_, t, _) in info_batch],
                    [lv for (_, _, lv) in info_batch],
                    strict=False,
                )
            )

            if num_workers == 1:
                # Sequential processing
                for args in worker_args:
                    entries.extend(_process_single_h2r_file(args))
                    produced_pairs += 1
                    if produced_pairs >= max_limit:
                        break
            else:
                from multiprocessing import Pool

                with Pool(processes=num_workers) as pool:
                    results = list(
                        tqdm(
                            pool.imap_unordered(_process_single_h2r_file, worker_args),
                            total=len(worker_args),
                            desc=f"Processing batch (workers={num_workers})",
                        )
                    )
                for res in results:
                    entries.extend(res)
                    produced_pairs += 1
                    if produced_pairs >= max_limit:
                        break

            # Clear batch
            file_batch = []
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
