from typing import Dict, List, Any, Optional
from itertools import cycle

import numpy as np
from collections import defaultdict
from robometer.data.dataset_types import ProgressSample
from robometer.data.samplers.base import RBMBaseSampler
from robometer.utils.logger import get_logger

logger = get_logger()


class ProgressPolicyRankingSampler(RBMBaseSampler):
    """Dataset that generates progress samples for policy ranking by selecting N trajectories per quality label for tasks with multiple quality labels."""

    def __init__(
        self,
        num_examples_per_quality_pr: int = 5,
        num_partial_successes: Optional[int] = None,
        frame_step: int = 1,
        use_frame_steps: bool = True,
        max_tasks: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_examples_per_quality_pr is None:
            num_examples_per_quality_pr = float("inf")
        self.num_examples_per_quality_pr = num_examples_per_quality_pr
        self.num_partial_successes = num_partial_successes
        self.frame_step = frame_step
        self.use_frame_steps = use_frame_steps
        self.max_tasks = max_tasks
        logger.info(f"ProgressPolicyRankingSampler initialized with {len(self.robot_trajectories)} trajectories")

        self.sample_indices = self._generate_all_sample_indices()

        logger.info(f"Generated {len(self.sample_indices)} sample indices")

    def _generate_all_sample_indices(self) -> List[Dict[str, Any]]:
        """Generate sample indices by selecting tasks with multiple quality labels/partial_success values and sampling N trajectories per group.

        For non-RoboArena: Groups by task and quality_label.
        For RoboArena: Groups by task and partial_success values.

        If use_frame_steps=True, generates subsequence samples like reward_alignment (0:frame_step, 0:2*frame_step, etc.).
        If use_frame_steps=False, generates one sample per trajectory (whole trajectory).
        """

        # Check if this is RoboArena (has partial_success)
        is_roboarena = False
        if self.robot_trajectories:
            first_traj = self.dataset[self.robot_trajectories[0]]
            is_roboarena = first_traj.get("partial_success") is not None

        # Group trajectories by task and grouping key (quality_label or partial_success)
        task_to_key_to_trajs = defaultdict(lambda: defaultdict(list))

        for traj_idx in self.robot_trajectories:
            traj = self.dataset[traj_idx]
            task = traj["task"]

            if is_roboarena:
                # RoboArena: Use rounded partial_success as key to group similar values
                partial_success_val = traj.get("partial_success")
                if partial_success_val is not None:
                    partial_success = round(float(partial_success_val), 2)
                    task_to_key_to_trajs[task][partial_success].append(traj_idx)
            else:
                # Non-RoboArena: Use quality_label
                quality = traj["quality_label"]
                task_to_key_to_trajs[task][quality].append(traj_idx)

        # Filter to tasks that have multiple grouping values
        tasks_with_multiple_values = {
            task: key_to_trajs for task, key_to_trajs in task_to_key_to_trajs.items() if len(key_to_trajs) > 1
        }

        dataset_type_str = "partial_success values" if is_roboarena else "quality labels"
        logger.info(f"Found {len(tasks_with_multiple_values)} tasks with multiple {dataset_type_str}")

        # Limit number of tasks if max_tasks is specified
        if self.max_tasks is not None and self.max_tasks > 0:
            # Convert to list, shuffle, and take first max_tasks
            # Sort tasks first to ensure deterministic ordering before shuffling
            tasks_list = sorted(tasks_with_multiple_values.items())
            self._local_random.shuffle(tasks_list)
            tasks_with_multiple_values = dict(tasks_list[: self.max_tasks])
            logger.info(f"Limited to {len(tasks_with_multiple_values)} tasks (max_tasks={self.max_tasks})")

        # Sample trajectories for each task
        sample_indices = []
        all_sampled_traj_indices = []
        # Sort tasks to ensure deterministic processing order
        for task, key_to_trajs in sorted(tasks_with_multiple_values.items()):
            if is_roboarena:
                # RoboArena: Use num_partial_successes for circular sampling
                num_to_sample_total = self.num_partial_successes

                # Build lists of available indices per partial_success (sorted for deterministic sampling)
                available_lists = []
                for partial_success in sorted(key_to_trajs.keys()):
                    traj_indices = sorted(key_to_trajs[partial_success])
                    if traj_indices:
                        available_lists.append(traj_indices)

                # Circular sampling: cycle through partial_success groups until we reach max
                sampled_traj_indices = []
                for available_indices in cycle(available_lists):
                    if len(sampled_traj_indices) >= num_to_sample_total:
                        break
                    if not available_indices:
                        # If all lists are empty, stop
                        if all(not lst for lst in available_lists):
                            break
                        continue

                    # Sample one trajectory from this group
                    sampled_idx = self._local_random.choice(available_indices)
                    sampled_traj_indices.append(sampled_idx)
                    # Remove the sampled index from this list
                    available_indices.remove(sampled_idx)

                # Generate samples for all sampled trajectories
                for traj_idx in sampled_traj_indices:
                    traj = self.dataset[traj_idx]
                    sample_indices.extend(self._generate_indices_for_trajectory(traj_idx, traj))
                    all_sampled_traj_indices.append(traj_idx)
            else:
                # Non-RoboArena: Sample N trajectories per quality label
                # Sort quality labels to ensure deterministic order
                for quality in sorted(key_to_trajs.keys()):
                    traj_indices = key_to_trajs[quality]
                    # Sort trajectory indices to ensure deterministic sampling
                    traj_indices = sorted(traj_indices)
                    # Sample up to num_examples_per_quality_pr trajectories for this quality label
                    num_to_sample = min(self.num_examples_per_quality_pr, len(traj_indices))
                    sampled_traj_indices = self._local_random.sample(traj_indices, num_to_sample)
                    for traj_idx in sampled_traj_indices:
                        traj = self.dataset[traj_idx]
                        sample_indices.extend(self._generate_indices_for_trajectory(traj_idx, traj))
                        all_sampled_traj_indices.append(traj_idx)

        logger.info(f"Sampled {len(sample_indices)} samples across {len(tasks_with_multiple_values)} tasks")
        logger.info(f"Sampled trajectory indices: {all_sampled_traj_indices}")

        return sample_indices

    def _generate_indices_for_trajectory(self, traj_idx: int, traj: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sample indices for a single trajectory.

        Args:
            traj_idx: Index of the trajectory in the dataset
            traj: Trajectory dictionary

        Returns:
            List of sample index dictionaries
        """
        num_frames = traj["num_frames"]
        indices = []

        if self.use_frame_steps:
            # Generate subsequence indices like reward_alignment: 0:frame_step, 0:2*frame_step, etc.
            for end_idx in range(self.frame_step, num_frames + 1, self.frame_step):
                frame_indices = list(range(end_idx))
                indices.append({
                    "traj_idx": traj_idx,
                    "frame_indices": frame_indices,
                    "num_frames": num_frames,
                    "video_path": traj["frames"],
                    "id": traj["id"],
                    "use_frame_steps": True,
                })
        else:
            # Generate one sample per trajectory (whole trajectory)
            indices.append({
                "traj_idx": traj_idx,
                "video_path": traj["frames"],
                "id": traj["id"],
                "use_frame_steps": False,
            })

        return indices

    def _generate_sample_from_indices(self, sample_idx_info: dict) -> ProgressSample:
        """Generate a single progress sample from trajectory index."""
        traj_idx = sample_idx_info["traj_idx"]
        use_frame_steps = sample_idx_info.get("use_frame_steps", True)

        traj = self.dataset[traj_idx]

        if use_frame_steps:
            # Frame steps mode: create subsequence like reward_alignment
            frame_indices = sample_idx_info["frame_indices"]
            num_frames = sample_idx_info["num_frames"]

            metadata = {
                "quality_label": traj["quality_label"],
                "data_source": traj["data_source"],
                "task": traj["task"],
                "id": traj["id"],
                "video_path": sample_idx_info["video_path"],
                "frame_step": frame_indices[-1] if frame_indices else 0,
            }

            trajectory = self._get_traj_from_data(
                traj=traj,
                frame_indices=frame_indices,
                metadata=metadata,
                pad_frames=self.pad_frames,
            )
        else:
            # Whole trajectory mode
            metadata = {
                "quality_label": traj["quality_label"],
                "data_source": traj["data_source"],
                "task": traj["task"],
                "id": traj["id"],
                "video_path": sample_idx_info["video_path"],
            }

            trajectory = self._get_traj_from_data(
                traj=traj,
                metadata=metadata,
                pad_frames=self.pad_frames,
            )

        sample = ProgressSample(trajectory=trajectory)
        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
