import os
import json
from collections import defaultdict

import cv2
import h5py
import numpy as np
from dataset_upload.helpers import generate_unique_id


DEBUG = False


class Ph2dFrameloader:
    """Pickle-able loader to read frames from an HDF5 file on demand.

    Reads frames from either 'observation.image.right' or 'observation.image.left'.
    Each stored frame is expected to be an encoded image buffer which is decoded
    with cv2.imdecode and converted to RGB.
    """

    def __init__(self, hdf5_path: str, camera: str = "right") -> None:
        self.hdf5_path = hdf5_path
        self.camera = camera.lower()

    def __call__(self) -> np.ndarray:
        key = "observation.image.right" if self.camera == "right" else "observation.image.left"

        with h5py.File(self.hdf5_path, "r") as f:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {self.hdf5_path}")

            data = f[key]
            encoded_frames = data[()]
            frames = []
            for encoded_frame in encoded_frames:
                frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
                frames.append(frame)

        frames_np = np.asarray(frames, dtype=np.uint8)
        if frames_np.ndim != 4 or frames_np.shape[-1] != 3:
            raise ValueError(f"Unexpected frames shape for {self.hdf5_path}: {frames_np.shape} (expected (T,H,W,3))")
        return frames_np


def create_new_trajectory(hdf5_path: str, is_robot: bool, caption: str, camera: str) -> dict:
    trajectory_info = {}
    trajectory_info["id"] = generate_unique_id()
    trajectory_info["task"] = caption
    trajectory_info["frames"] = Ph2dFrameloader(hdf5_path, camera=camera)
    trajectory_info["is_robot"] = is_robot
    trajectory_info["quality_label"] = "successful"
    trajectory_info["partial_success"] = 1
    trajectory_info["data_source"] = "ph2d"
    return trajectory_info


def load_ph2d_dataset(dataset_path: str) -> dict[str, list[dict]]:
    """Load Ph2d dataset organized as folders with HDF5 files and optional metadata.json.

    Expected layout:
        <dataset_path>/
            ph2d_metadata.json   # Optional (dataset-specific). TODO: parse to captions/tasks
            <sequence_1>/
                *.hdf5
            <sequence_2>/
                ...

    Args:
        dataset_path: Root directory containing sequence folders of HDF5 files.

    Returns:
        Mapping: task/caption -> list of trajectory dicts.
    """

    task_data: dict[str, list[dict]] = defaultdict(list)

    # load metadata.json
    all_task_attributes = json.load(open(os.path.join(dataset_path, "ph2d_metadata.json")))["per_task_attributes"]
    for task_name, task_attributes in all_task_attributes.items():
        # load all sequences
        embodiment_type = task_attributes["embodiment_type"]
        if "human" in embodiment_type:
            is_robot = False
        else:
            is_robot = True

        for h5_file in os.listdir(os.path.join(dataset_path, task_name)):
            if not h5_file.endswith(".hdf5"):
                continue

            h5_path = os.path.join(dataset_path, task_name, h5_file)
            loaded_data = h5py.File(h5_path, "r")
            task_description = loaded_data.attrs["description"]

            if "observation.image.right" in loaded_data:
                task_data[task_name].append(create_new_trajectory(h5_path, is_robot, task_description, camera="right"))
            if "observation.image.left" in loaded_data:
                task_data[task_name].append(create_new_trajectory(h5_path, is_robot, task_description, camera="left"))

        if DEBUG:
            break
    print(f"Loaded {sum(len(task_list) for task_list in task_data.values())} trajectories from {len(task_data)} tasks")
    return task_data
