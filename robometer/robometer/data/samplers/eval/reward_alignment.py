#!/usr/bin/env python3
"""
Data generator for reward alignment evaluation.

This generator creates subsequence samples from trajectories for progress prediction evaluation.
For each trajectory, it creates multiple subsequences (0:2, 0:4, 0:6, etc.) and formats them
as PreferenceSample objects that can be evaluated by the model.
"""

from typing import Dict, List, Any
import numpy as np
import torch
from tqdm import tqdm

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.data.samplers.base import RBMBaseSampler
from robometer.utils.distributed import rank_0_print
from robometer.utils.logger import get_logger

logger = get_logger()


class RewardAlignmentSampler(RBMBaseSampler):
    """
    Data generator that creates subsequence samples for reward alignment evaluation.

    For each trajectory, creates subsequences of frames (0:2, 0:4, 0:6, etc.)
    and formats them as PreferenceSample objects for evaluation.
    """

    def __init__(
        self,
        max_trajectories: int | None = None,
        frame_step: int = 1,
        use_frame_steps: bool = True,
        subsample_n_frames: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_trajectories = max_trajectories
        self.frame_step = frame_step
        self.use_frame_steps = use_frame_steps
        self.subsample_n_frames = subsample_n_frames
        self.sample_indices = self._generate_all_sample_indices()

        rank_0_print(
            f"Generated {len(self.sample_indices)} reward alignment sample indices from {min(len(self.robot_trajectories), self.max_trajectories) if self.max_trajectories else len(self.robot_trajectories)} trajectories",
            verbose=self.verbose,
        )

    def _generate_all_sample_indices(self) -> List[Dict[str, Any]]:
        """Generate all possible subsequence sample indices (not the actual samples)."""
        sample_indices = []

        # Limit number of trajectories if specified
        trajectories_to_process = self.robot_trajectories
        if self.max_trajectories is not None and self.max_trajectories < len(self.robot_trajectories):
            trajectories_to_process = self._local_random.sample(self.robot_trajectories, self.max_trajectories)

        rank_0_print(
            f"Generating subsequence samples for {len(trajectories_to_process)} trajectories", verbose=self.verbose
        )

        all_num_frames = []
        for traj_idx in trajectories_to_process:
            traj = self.dataset[traj_idx]
            num_frames = traj["num_frames"]
            all_num_frames.append(num_frames)
            sample_indices.extend(self._generate_indices_for_trajectory(traj_idx, traj))

        logger.info(f"All num frames: {all_num_frames}")
        return sample_indices

    def _generate_indices_for_trajectory(self, traj_idx: int, traj: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sample indices for a single trajectory.

        Args:
            traj_idx: Index of the trajectory in the dataseqt
            traj: Trajectory dictionary

        Returns:
            List of sample index dictionaries
        """
        num_frames = traj["num_frames"]
        indices = []

        if self.use_frame_steps:
            if self.subsample_n_frames:
                if self.subsample_n_frames > num_frames:
                    end_indices = list(range(num_frames))
                else:
                    end_indices = np.linspace(0, num_frames - 1, self.subsample_n_frames)
                for end_idx in end_indices:
                    frame_indices = list(range(int(end_idx) + 1))
                    indices.append({
                        "traj_idx": traj_idx,
                        "frame_indices": frame_indices,
                        "num_frames": num_frames,
                        "video_path": traj["frames"],
                        "id": traj["id"],
                        "use_frame_steps": True,
                    })
            else:
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
        """Generate a single subsequence sample from stored indices."""
        traj_idx = sample_idx_info["traj_idx"]
        use_frame_steps = sample_idx_info.get("use_frame_steps", True)

        traj = self.dataset[traj_idx]

        if use_frame_steps:
            # Frame steps mode: create subsequence like reward_alignment
            frame_indices = sample_idx_info["frame_indices"]
            num_frames = sample_idx_info["num_frames"]

            metadata = {
                "data_gen_strategy": "reward_alignment",
                "id": traj["id"],
                "video_path": sample_idx_info["video_path"],
                "frame_step": frame_indices[-1] if frame_indices else 0,
                "num_frames": num_frames,
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
                "data_gen_strategy": "reward_alignment",
                "id": traj["id"],
                "video_path": sample_idx_info["video_path"],
            }

            trajectory = self._get_traj_from_data(
                traj=traj,
                metadata=metadata,
                pad_frames=self.pad_frames,
            )

        sample = ProgressSample(trajectory=trajectory, sample_type="progress")
        return sample

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self._generate_sample_from_indices(self.sample_indices[idx])
