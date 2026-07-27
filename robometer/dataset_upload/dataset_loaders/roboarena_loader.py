import os
from collections import defaultdict

import cv2
import numpy as np
import yaml
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


class RoboarenaFrameloader:
    """Pickle-able loader that reads Roboarena frames from disk on demand.

    Stores only simple fields so it can be safely passed across processes.
    """

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path

    def __call__(self) -> np.ndarray:
        """Load frames from disk when called.

        Returns:
            np.ndarray of shape (T, H, W, 3), dtype uint8
        """
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        frames = np.array(frames)

        # Ensure shape and dtype sanity
        if not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"Unexpected frames shape for {self.video_path} in {self.video_path}: {getattr(frames, 'shape', None)}"
            )

        # Ensure uint8
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8, copy=False)

        return frames


def create_new_trajectory(video_path: str, partial_success: int, task_name: str) -> dict:
    trajectory_info = {}
    trajectory_info["id"] = generate_unique_id()
    trajectory_info["task"] = task_name
    trajectory_info["frames"] = RoboarenaFrameloader(video_path)
    trajectory_info["is_robot"] = True
    trajectory_info["quality_label"] = "successful" if partial_success == 1.0 else "failure"
    trajectory_info["partial_success"] = partial_success
    trajectory_info["data_source"] = "roboarena"
    return trajectory_info


def load_roboarena_dataset(dataset_path: str) -> dict[str, list[dict]]:
    eval_folder = os.path.join(dataset_path, "evaluation_sessions")
    eval_sessions = os.listdir(eval_folder)

    # tasks_to_videos = dict()
    # task : {video_path: ..., partial_success}
    task_data = defaultdict(list)

    for eval_session in eval_sessions:
        eval_session_path = os.path.join(eval_folder, eval_session)
        metadata_path = os.path.join(eval_session_path, "metadata.yaml")
        # load metadata
        with open(metadata_path) as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)
        task = metadata["language_instruction"]
        # if task in tasks_to_videos:
        #    print(f"Task {task} already in tasks")
        # tasks_to_videos[task] = []
        # pprint.pprint(metadata)
        policies = metadata["policies"]
        for policy_id, policy_info in policies.items():
            # get the partial success
            partial_success = policy_info["partial_success"]
            policy_name = policy_info["policy_name"]
            policy_folder_name = f"{policy_id}_{policy_name}"
            # get the videos of _left and _right
            policy_folder_path = os.path.join(eval_session_path, policy_folder_name)
            files_in_policy_folder = os.listdir(policy_folder_path)
            video_left = [f for f in files_in_policy_folder if f.endswith("_left.mp4")]
            video_right = [f for f in files_in_policy_folder if f.endswith("_right.mp4")]
            # video_wrist = [f for f in files_in_policy_folder if f.endswith("_wrist.mp4")]
            if len(video_left) > 0:
                video_path = os.path.join(policy_folder_path, video_left[0])
                task_data[task].append(
                    create_new_trajectory(video_path, partial_success=partial_success, task_name=task)
                )
            if len(video_right) > 0:
                video_path = os.path.join(policy_folder_path, video_right[0])
                task_data[task].append(
                    create_new_trajectory(video_path, partial_success=partial_success, task_name=task)
                )
            # if len(video_wrist) > 0:
            #    video_path = os.path.join(policy_folder_path, video_wrist[0])
            #    task_data[task].append(
            #        create_new_trajectory(video_path, partial_success=partial_success, task_name=task)
            #    )
    print(
        f"Loaded {sum([len(task_list) for task_list in task_data.values()])} trajectories from {len(task_data)} tasks"
    )
    return task_data
