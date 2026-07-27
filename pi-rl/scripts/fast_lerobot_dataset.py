# adapted from openpi
"""
FastLeRobotDataset: a drop-in replacement for LeRobotDataset.create() that skips
the temporary PNG file dance (numpy → PIL → PNG → disk → read back → embed in parquet).

Instead, it keeps raw image numpy arrays in memory and encodes them to PNG only once
when writing the final parquet. This eliminates redundant disk I/O and double-encoding.
"""

import io
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import datasets
import numpy as np
import PIL.Image
from datasets import concatenate_datasets

from lerobot.common.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME,
    LeRobotDatasetMetadata,
    get_hf_features_from_features,
    hf_transform_to_torch,
    get_episode_data_index,
    check_timestamps_sync,
)
from lerobot.common.datasets.compute_stats import get_feature_stats, sample_indices, auto_downsample_height_width


def _encode_numpy_to_png_bytes(img: np.ndarray) -> bytes:
    """Encode a HWC uint8 numpy array to PNG bytes in memory."""
    pil_img = PIL.Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _compute_episode_stats_from_arrays(episode_buffer: dict, features: dict) -> dict:
    """Compute episode stats directly from in-memory numpy arrays (no disk reads)."""
    ep_stats = {}
    for key in features:
        if key in ["index", "episode_index", "task_index"]:
            continue
        if features[key]["dtype"] == "string":
            continue

        if features[key]["dtype"] in ["image", "video"]:
            images_list = episode_buffer[key]
            sampled_idxs = sample_indices(len(images_list))
            sampled = []
            for idx in sampled_idxs:
                img = images_list[idx]
                img_chw = np.transpose(img, (2, 0, 1))
                img_chw = auto_downsample_height_width(img_chw)
                sampled.append(img_chw)
            ep_ft_array = np.stack(sampled, axis=0).astype(np.uint8)
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            data = episode_buffer[key]
            if isinstance(data, list):
                data = np.stack(data)
            ep_ft_array = data
            axes_to_reduce = 0
            keepdims = data.ndim == 1

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                for k, v in ep_stats[key].items()
            }

    return ep_stats


class FastLeRobotDataset:
    """
    Fast dataset writer that avoids temporary image files.
    
    Usage is identical to LeRobotDataset.create() + add_frame() + save_episode().
    """

    def __init__(
        self,
        repo_id: str,
        robot_type: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        tolerance_s: float = 1e-4,
        num_encode_threads: int = 8,
    ):
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        if self.root.exists():
            shutil.rmtree(self.root)

        self.fps = fps
        self.tolerance_s = tolerance_s
        self.features = features
        self._encode_pool = ThreadPoolExecutor(max_workers=num_encode_threads)

        self.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            root=self.root,
            robot_type=robot_type,
            features=features,
            use_videos=False,
        )
        self.features = self.meta.info["features"]

        self.hf_features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in self.hf_features}
        self.hf_dataset = datasets.Dataset.from_dict(ft_dict, features=self.hf_features, split="train")
        self.hf_dataset.set_transform(hf_transform_to_torch)

        self.episode_buffer = self._create_episode_buffer()

    def _create_episode_buffer(self) -> dict:
        ep_buffer = {"size": 0, "task": []}
        for key in self.features:
            ep_buffer[key] = self.meta.total_episodes if key == "episode_index" else []
        return ep_buffer

    def add_frame(self, frame: dict) -> None:
        """Add a frame. Images should be numpy uint8 HWC arrays."""
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key == "task":
                self.episode_buffer["task"].append(frame["task"])
                continue
            if key not in self.features:
                raise ValueError(f"'{key}' not in features: {list(self.features.keys())}")
            self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self) -> None:
        """Encode images and write episode parquet in one shot."""
        episode_buffer = self.episode_buffer
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self.meta.total_frames, self.meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        episode_buffer["task_index"] = np.array(
            [self.meta.get_task_index(task) for task in tasks]
        )

        ep_stats = _compute_episode_stats_from_arrays(episode_buffer, self.features)

        # Build the parquet-ready dict
        ep_dict = {}
        image_keys = []
        for key in self.hf_features:
            if key in ["index", "episode_index", "task_index"]:
                ep_dict[key] = episode_buffer[key]
            elif self.features.get(key, {}).get("dtype") in ["image", "video"]:
                image_keys.append(key)
            else:
                ep_dict[key] = np.stack(episode_buffer[key])

        # Parallel PNG encoding across all image keys and frames
        if image_keys:
            futures = {}
            for key in image_keys:
                for i, img in enumerate(episode_buffer[key]):
                    futures[(key, i)] = self._encode_pool.submit(_encode_numpy_to_png_bytes, img)

            for key in image_keys:
                ep_dict[key] = [
                    {"bytes": futures[(key, i)].result(), "path": f"frame_{i:06d}.png"}
                    for i in range(len(episode_buffer[key]))
                ]

        ep_dataset = datasets.Dataset.from_dict(ep_dict, features=self.hf_features, split="train")
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)

        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        # Validate timestamps
        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            np.array(episode_buffer["timestamp"], dtype=np.float32),
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        self.episode_buffer = self._create_episode_buffer()

    def push_to_hub(self, **kwargs):
        """Delegate to underlying hf dataset push."""
        raise NotImplementedError("Use LeRobotDataset for push_to_hub support.")
