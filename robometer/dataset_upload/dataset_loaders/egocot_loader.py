import os
import json
from collections import defaultdict
from tqdm import tqdm

import numpy as np
from dataset_upload.helpers import generate_unique_id

trajectory_info_template = {
    "id": [],
    "task": [],
    # "lang_vector": [],
    "data_source": None,
    "frames": None,
    "is_robot": None,
    "quality_label": None,
    "partial_success": None,  # in [0, 1]
}


class EgoCOTFrameloader:
    """Pickle-able loader that reads EgoCoT frames from disk on demand.

    Stores only simple fields so it can be safely passed across processes.
    """

    def __init__(self, frames_path: str) -> None:
        self.frames_path = frames_path

    def __call__(self) -> np.ndarray:
        """Load frames from disk when called.

        Returns:
            np.ndarray of shape (T, H, W, 3), dtype uint8
        """
        # Load the numpy array containing 8 consecutive frames
        frames = np.load(self.frames_path)

        # Ensure the frames are in the correct format
        # EgoCoT frames are stored as numpy arrays with 8 consecutive frames
        assert frames.ndim == 4, f"Expected 4D array, got {frames.ndim}D array"

        # Ensure shape and dtype sanity
        if not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"Unexpected frames shape for {self.frames_path}: {getattr(frames, 'shape', None)}")

        # Ensure uint8
        if frames.dtype != np.uint8:
            # Convert from float to uint8 if necessary
            if frames.dtype in [np.float32, np.float64]:
                if frames.max() <= 1.0:
                    # Values in [0, 1] range - just scale to [0, 255]
                    frames = (frames * 255).astype(np.uint8)
                else:
                    # Values appear to be ImageNet normalized - denormalize first
                    # ImageNet normalization: (pixel - mean) / std
                    # Reverse: pixel * std + mean
                    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
                    imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

                    # Denormalize: multiply by std and add mean
                    frames = frames * imagenet_std + imagenet_mean

                    # Clamp to [0, 1] and convert to [0, 255]
                    frames = np.clip(frames, 0, 1)
                    frames = (frames * 255).astype(np.uint8)
            else:
                frames = frames.astype(np.uint8, copy=False)

        # now convert frames to (T, H, W, 3) from (T, C, H, W)
        frames = frames.transpose(0, 2, 3, 1)

        return frames


def create_new_trajectory(frames_path: str, caption: str) -> dict:
    """Create a new trajectory from EgoCoT data."""
    trajectory_info = {}
    trajectory_info["id"] = generate_unique_id()
    trajectory_info["task"] = caption  # Use caption as the task description
    trajectory_info["frames"] = EgoCOTFrameloader(frames_path)
    trajectory_info["is_robot"] = False  # EgoCoT is human egocentric data
    trajectory_info["quality_label"] = "successful"
    trajectory_info["partial_success"] = 1
    trajectory_info["data_source"] = "egocot"
    return trajectory_info


def load_egocot_dataset(dataset_path: str) -> dict[str, list[dict]]:
    """Load EgoCoT dataset from results.json and .npy frame files (EgoCOT_clear).

    Expected layout (example):
        <dataset_path>/
          results.json
          EgoCOT_clear/
            EGO_0000.npy
            EGO_0001.npy

    Args:
        dataset_path: Path to a directory containing one or more EgoCoT result folders

    Returns:
        Dictionary mapping task descriptions to lists of trajectory dictionaries
    """
    # Locate results.json files
    json_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower() == "results.json":
                json_files.append(os.path.join(root, file))

    if not json_files:
        raise FileNotFoundError(f"No results.json files found in {dataset_path}")

    task_data = defaultdict(list)
    total_trajectories = 0

    for json_file in json_files:
        print(f"Loading annotations from {json_file}")

        with open(json_file, "r") as f:
            annotations = json.load(f)

        # Normalize JSON structures
        if isinstance(annotations, list):
            data_items = annotations
        elif isinstance(annotations, dict):
            if "data" in annotations:
                data_items = annotations["data"]
            elif "results" in annotations:
                data_items = annotations["results"]
            elif "annotations" in annotations:
                data_items = annotations["annotations"]
            elif "samples" in annotations:
                data_items = annotations["samples"]
            elif "image" in annotations:
                data_items = [annotations]
            else:
                raise ValueError(f"Unexpected JSON structure in {json_file}: cannot find data list")
        else:
            raise ValueError(f"Unexpected JSON structure in {json_file}")

        for item in tqdm(data_items, desc="Processing trajectories"):
            # Extract required fields (handle common variants and typos)
            image_filename = item.get("image")
            # this caption is the llm generated one that's much more detailed from EgoCOT's processing
            caption = item.get("planing").split("\n")[0][1:]
            if not caption:
                caption = item.get("caption")  # use the backup original caption
            score = item.get("score")

            if not image_filename or not caption:
                print(f"Skipping item with missing image or caption: {item}")
                continue

            # Construct full path to the .npy file, prioritizing EgoCOT_clear next to results.json
            base_dir = os.path.dirname(json_file)
            candidate_paths = [
                os.path.join(base_dir, "EgoCOT_clear", image_filename) if image_filename else None,
                # os.path.join(base_dir, image_filename) if image_filename else None,
                # os.path.join(dataset_path, "EgoCOT_clear", image_filename) if image_filename else None,
                # os.path.join(dataset_path, image_filename) if image_filename else None,
            ]

            frames_path = None
            for cand in candidate_paths:
                if cand and os.path.exists(cand):
                    frames_path = cand
                    break

            if frames_path is None:
                print(f"Warning: .npy frame file not found for: {image_filename}")
                continue

            if not frames_path.lower().endswith(".npy"):
                print(f"Warning: expected .npy file, got: {frames_path}. Skipping.")
                continue

            # Create trajectory
            trajectory = create_new_trajectory(frames_path, caption)

            # Group by task/caption for organization
            task_key = caption[:50] + "..." if len(caption) > 50 else caption
            task_data[task_key].append(trajectory)
            total_trajectories += 1

    print(f"Loaded {total_trajectories} trajectories from {len(task_data)} unique tasks")
    return task_data
