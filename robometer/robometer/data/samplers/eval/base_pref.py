from typing import Dict, Any

import numpy as np

from robometer.data.dataset_types import PreferenceSample, Trajectory
from robometer.data.samplers.base import RBMBaseSampler


class BaseQualityPreferenceSampler(RBMBaseSampler):
    """Base class for quality preference samplers.

    Subclasses should implement `_generate_all_sample_indices` to define how
    trajectories are paired. This base class provides the common `_generate_sample_from_indices`
    method that loads and processes the trajectories.
    """

    def _generate_sample_from_indices(self, sample_idx_info: Dict[str, Any]) -> PreferenceSample:
        """Generate a single sample from stored indices."""
        chosen_idx = sample_idx_info["chosen_traj_idx"]
        rejected_idx = sample_idx_info["rejected_traj_idx"]

        # Get the trajectories
        chosen_traj = self.dataset[chosen_idx]
        rejected_traj = self.dataset[rejected_idx]

        chosen_metadata = {
            "quality_label": chosen_traj["quality_label"],
            "data_source": chosen_traj["data_source"],
            "task": chosen_traj["task"],
            "id": chosen_traj["id"],
            "video_path": chosen_traj["frames"],
        }
        # Add partial_success if available
        if chosen_traj.get("partial_success") is not None:
            chosen_metadata["partial_success"] = chosen_traj.get("partial_success")

        chosen_trajectory = self._get_traj_from_data(
            traj=chosen_traj,
            metadata=chosen_metadata,
        )

        rejected_metadata = {
            "quality_label": rejected_traj["quality_label"],
            "data_source": rejected_traj["data_source"],
            "task": rejected_traj["task"],
            "id": rejected_traj["id"],
            "video_path": rejected_traj["frames"],
        }
        # Add partial_success if available
        if rejected_traj.get("partial_success") is not None:
            rejected_metadata["partial_success"] = rejected_traj.get("partial_success")

        rejected_trajectory = self._get_traj_from_data(
            traj=rejected_traj,
            metadata=rejected_metadata,
        )

        data_gen_strategy = getattr(self, "data_gen_strategy", "quality_preference")

        # Create preference sample
        sample = PreferenceSample(
            chosen_trajectory=chosen_trajectory,
            rejected_trajectory=rejected_trajectory,
            data_gen_strategy=data_gen_strategy,
        )

        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
