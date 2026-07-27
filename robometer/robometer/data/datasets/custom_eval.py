#!/usr/bin/env python3
import torch
from typing import List, Optional

from robometer.data.datasets.base import BaseDataset
from robometer.data.samplers import *
from robometer.configs.experiment_configs import DataConfig


class CustomEvalDataset(BaseDataset):
    """Dataset that wraps a sampler for custom evaluation purposes."""

    def __init__(
        self,
        sampler_type: str,
        config: DataConfig,
        verbose: bool = True,
        sampler_kwargs: dict = None,
    ):
        """Initialize custom eval dataset with a sampler type.

        Args:
            sampler_type: Type of sampler to create (e.g., "confusion_matrix", "reward_alignment", "policy_ranking", "success_failure")
            config: Configuration object
            verbose: Verbose flag
            sampler_kwargs: Additional keyword arguments for the sampler
        """
        filter_quality_labels: Optional[List[str]] = None

        if sampler_type == "reward_alignment":
            filter_quality_labels = ["successful"]
        elif sampler_type == "confusion_matrix":
            filter_quality_labels = ["successful", "suboptimal"]

        # Special case: roboreward datasets should not filter by quality
        if len(config.eval_datasets) == 1 and "roboreward" in config.eval_datasets[0]:
            filter_quality_labels = None

        super().__init__(config=config, is_evaluation=True, filter_quality_labels=filter_quality_labels)

        sampler_cls = {
            "confusion_matrix": ConfusionMatrixSampler,
            "reward_alignment": RewardAlignmentSampler,
            "policy_ranking": ProgressPolicyRankingSampler,
            "quality_preference": QualityPreferenceSampler,
        }

        if "roboarena" in self.config.eval_datasets:
            sampler_cls["quality_preference"] = RoboArenaQualityPreferenceSampler

        if sampler_type not in sampler_cls:
            raise ValueError(f"Unknown sampler type: {sampler_type}. Available: {list(sampler_cls.keys())}")

        self.sampler = sampler_cls[sampler_type](
            config=config,
            dataset=self.dataset,
            combined_indices=self._combined_indices,
            dataset_success_cutoff_map=self.dataset_success_cutoff_map,
            verbose=verbose,
            **sampler_kwargs,
        )

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        return self.sampler[idx]
