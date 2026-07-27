#!/usr/bin/env python3
"""
PrefSampler class for producing batches of preference data.
"""

from typing import Dict, List, Optional, Any

import random

from robometer.data.dataset_types import PreferenceSample, Trajectory
from robometer.data.samplers.base import RBMBaseSampler
from robometer.data.datasets.helpers import (
    DataGenStrat,
    convert_continuous_to_discrete_bins,
)
from robometer.utils.logger import get_logger, rank_0_info, trace
from robometer.utils.timer import timer

logger = get_logger()


class PrefSampler(RBMBaseSampler):
    """Data generator for producing batches of preference prediction data."""

    def __init__(self, is_evaluation=False, **kwargs):
        super().__init__(**kwargs)

        self.dataset_preference_ratio = self.config.dataset_preference_ratio
        self.preference_strategy_ratio: List[float] = self.config.preference_strategy_ratio
        self._has_suboptimal = (
            any(len(indices) > 0 for indices in self.suboptimal_by_task.values()) if self.suboptimal_by_task else False
        )
        rank_0_info(f"[PREF SAMPLER] Has suboptimal: {self._has_suboptimal}")

        # Initialize preference dataset
        self._load_preference_dataset()

    def _generate_sample(self, item: dict, preferred_strategy: Optional[DataGenStrat] = None):
        """Generate a preference sample from an item.

        If the item has a non-successful quality label, it will be used as the rejected
        trajectory and an optimal trajectory from the same task will be found as the chosen one.
        Otherwise, normal preference sampling logic is used.

        Args:
            item: The trajectory item
            preferred_strategy: Optional strategy to use (if None, will select strategy based on ratios)
        """
        quality_label = item["quality_label"]
        use_partial_success = item.get("partial_success") is not None

        # Handle non-successful trajectories: use as rejected, find optimal from same task as chosen
        # skip this for trajectories with partial_success which we will handle with partial success logic
        if quality_label != "successful" and not use_partial_success:
            traj_id = item["id"]
            task_name = item["task"]

            logger.trace(
                f"[PREF SAMPLER] Non-successful quality detected for ID={traj_id}, using as rejected trajectory, task={task_name}"
            )

            # Find optimal trajectories from the same task
            same_task_optimal_indices = self.optimal_by_task.get(task_name, [])

            if not same_task_optimal_indices:
                logger.trace(
                    f"[PREF SAMPLER] No optimal trajectories found for task '{task_name}', falling through to normal sampling"
                )
                return self._create_pref_sample(item, preferred_strategy=preferred_strategy)

            # Select a random optimal trajectory from the same task as chosen
            chosen_idx = random.choice(same_task_optimal_indices)
            chosen_traj_dict = self.dataset[chosen_idx]

            chosen_trajectory = self._get_traj_from_data(chosen_traj_dict)
            rejected_trajectory = self._get_traj_from_data(item)

            sample = PreferenceSample(
                chosen_trajectory=chosen_trajectory,
                rejected_trajectory=rejected_trajectory,
                data_gen_strategy=DataGenStrat.SUBOPTIMAL.value,
            )

            logger.trace(
                f"[PREF SAMPLER] Created preference sample for non-successful traj ID={traj_id} with optimal traj from same task"
            )
            return sample

        return self._create_pref_sample(item, preferred_strategy=preferred_strategy)

    def _execute_strategy(
        self, strategy: DataGenStrat, chosen_traj: Dict[str, Any], use_partial_success: bool
    ) -> tuple[Dict[str, Any], str, Dict[str, Any]] | None:
        """Execute a strategy to get rejected trajectory.

        Args:
            strategy: The strategy to execute
            chosen_traj: The chosen trajectory
            use_partial_success: Whether this trajectory uses partial_success

        Returns:
            Tuple of (rejected_traj, rejected_subsample_strategy, chosen_traj) or None if failed
            Note: chosen_traj may be swapped with rejected_traj for partial_success trajectories
        """
        max_retries = 3
        rejected_subsample_strategy = None
        rejected_traj = None

        if strategy == DataGenStrat.REWIND:
            rejected_traj = chosen_traj.copy()
            rejected_subsample_strategy = "subsample_rewind"
        elif strategy == DataGenStrat.SUBOPTIMAL:
            for _ in range(max_retries):
                rejected_traj = self._get_same_task_suboptimal(chosen_traj)
                if rejected_traj is not None:
                    # For trajectories with partial_success, if the returned trajectory has higher partial_success, swap them
                    if use_partial_success:
                        chosen_partial_success = chosen_traj.get("partial_success")
                        rejected_partial_success = rejected_traj.get("partial_success")
                        if rejected_partial_success is not None and chosen_partial_success is not None:
                            if rejected_partial_success > chosen_partial_success:
                                logger.trace(
                                    f"[PREF SAMPLER] Swapping trajectories: found higher partial_success "
                                    f"({rejected_partial_success} > {chosen_partial_success})"
                                )
                                rejected_traj, chosen_traj = chosen_traj, rejected_traj
                    break
            rejected_subsample_strategy = "subsample_forward"
        elif strategy == DataGenStrat.DIFFERENT_TASK:
            for _ in range(max_retries):
                rejected_traj = self._get_different_video_traj(chosen_traj)
                if rejected_traj is not None:
                    break
            rejected_subsample_strategy = "subsample_forward"
        elif strategy == DataGenStrat.REVERSE_PROGRESS:
            rejected_traj = chosen_traj.copy()
            rejected_subsample_strategy = "subsample_reverse"
        else:
            return None

        if rejected_traj is None:
            return None

        return (rejected_traj, rejected_subsample_strategy, chosen_traj)

    def _create_pref_sample_from_dataset(self) -> PreferenceSample:
        """Create a preference sample from the loaded preference dataset."""
        if not self.preferences:
            return None

        # For now, return a simple preference sample
        # This can be enhanced later when we have actual preference data
        random.choice(self.preferences)

        # This is a placeholder - would need to be implemented based on actual preference data structure
        return None

    def _load_preference_dataset(self):
        """Load the preference dataset from disk or hub if provided."""
        self.preferences = []

        # For now, we'll use empty preferences since the config structure has changed
        # This can be updated later if needed
        rank_0_info("[PREF SAMPLER] No preference dataset provided, will use random sampling for preferences")
        return

    def _create_preference_sample(self) -> PreferenceSample:
        """Create a preference prediction sample: chosen vs rejected where chosen is preferred.
        Either from dataset or from generated trajectories.

        Returns:
            PreferenceSample: A preference sample with chosen (preferred) vs rejected
            (suboptimal) trajectories and associated metadata
        """

        with timer("create_preference_sample", verbose=False):
            if random.random() < self.dataset_preference_ratio and self.preferences:
                # Use preference trajectories from dataset
                return self._create_pref_sample_from_dataset()
            else:
                return self._create_pref_sample()

    def _create_pref_sample(
        self, chosen_traj: Optional[Dict[str, Any]] = None, preferred_strategy: Optional[DataGenStrat] = None
    ) -> PreferenceSample:
        """Create a preference prediction sample using various rejected trajectory generation strategies.

        Rewind Same Task
        - Creates a suboptimal trajectory by rewinding the chosen trajectory

        Suboptimal/Failure Same Task
        - Uses existing suboptimal/failure trajectories from the same task

        Different Task
        - Uses trajectories from completely different tasks

        Returns:
            PreferenceSample: A preference sample with chosen (preferred) vs rejected
            (suboptimal) trajectories and associated metadata

        Raises:
            ValueError: If no chosen trajectories are available for preference generation
            RuntimeError: If all strategies fail and fallback rewind also fails
        """
        # Log when preference sampler is called
        traj_id = chosen_traj["id"] if chosen_traj is not None else "sampling_new"
        logger.trace(f"[PREF SAMPLER] Creating preference sample for trajectory ID: {traj_id}")

        # Use provided chosen trajectory if given; otherwise sample one
        if chosen_traj is None:
            # Use preprocessed chosen trajectories from index maps
            if not self.optimal_by_task:
                return None

            # Filter out tasks with empty optimal_indices to avoid infinite loop
            valid_tasks = {
                task: indices
                for task, indices in self.optimal_by_task.items()
                if indices  # Only include tasks with non-empty indices
            }

            if not valid_tasks:
                # No valid tasks with optimal trajectories available
                return None

            # Get a random task and chosen trajectory from it
            task_name = random.choice(list(valid_tasks.keys()))
            optimal_indices = valid_tasks[task_name]

            # Double-check that we have valid indices (should always be true now)
            if not optimal_indices:
                return None

            chosen_idx = random.choice(optimal_indices)
            chosen_traj = self.dataset[chosen_idx]

        # Initialize variables for strategy selection
        rejected_traj = None
        strategy_used = None
        rejected_subsample_strategy = None

        # Check if this trajectory uses partial_success
        use_partial_success = chosen_traj.get("partial_success") is not None
        if use_partial_success:
            partial_success = chosen_traj.get("partial_success")
            logger.trace(
                f"[PREF SAMPLER] Trajectory with partial_success detected (ID: {chosen_traj.get('id', 'unknown')}, partial_success: {partial_success})"
            )

        # Strategy selection: use preferred_strategy if provided, otherwise select based on ratios
        if preferred_strategy is not None:
            # Use the preferred strategy directly
            logger.trace(f"[PREF SAMPLER] Using preferred strategy: {preferred_strategy.value}")
            result = self._execute_strategy(preferred_strategy, chosen_traj, use_partial_success)
            if result is None:
                logger.trace(f"[PREF SAMPLER] Preferred strategy {preferred_strategy.value} failed, returning None")
                return None
            rejected_traj, rejected_subsample_strategy, chosen_traj = result
            strategy_used = preferred_strategy
            attempt = 1  # Set attempt for preferred strategy path
        else:
            # Strategy selection with rebalancing on failure
            strategies = []
            if self.preference_strategy_ratio[0] > 0:
                strategies.append((DataGenStrat.REWIND, self.preference_strategy_ratio[0]))
            if self._has_suboptimal and self.preference_strategy_ratio[1] > 0:
                strategies.append((DataGenStrat.SUBOPTIMAL, self.preference_strategy_ratio[1]))
            if self.preference_strategy_ratio[2] > 0:
                strategies.append((DataGenStrat.DIFFERENT_TASK, self.preference_strategy_ratio[2]))
            if self.preference_strategy_ratio[3] > 0:
                strategies.append((DataGenStrat.REVERSE_PROGRESS, self.preference_strategy_ratio[3]))

            max_attempts = 10  # Limit retry attempts to prevent infinite loops
            max_strategy_attempts = 3  # Maximum attempts per strategy before removing it
            attempt = 0

            # Track attempts per strategy
            strategy_attempt_counts = {strat: 0 for strat, _ in strategies}

            while rejected_traj is None and attempt < max_attempts:
                attempt += 1

                # Check if we have any strategies left
                if not strategies:
                    return None

                # Rebalance probabilities based on remaining strategies
                total_prob = sum(prob for _, prob in strategies)
                if total_prob == 0:
                    return None

                # Normalize probabilities
                normalized_strategies = [(strat, prob / total_prob) for strat, prob in strategies]

                # Select strategy based on rebalanced probabilities
                prob = random.random()
                cumulative_prob = 0.0
                selected_strategy = None

                for strat, normalized_prob in normalized_strategies:
                    cumulative_prob += normalized_prob
                    if prob <= cumulative_prob:
                        selected_strategy = strat
                        break

                # Log strategy attempt
                logger.trace(
                    f"[PREF SAMPLER] Attempt {attempt}/{max_attempts}: Trying strategy {selected_strategy.value if selected_strategy else 'None'}"
                )

                # Execute selected strategy
                result = self._execute_strategy(selected_strategy, chosen_traj, use_partial_success)
                if result is not None:
                    rejected_traj, rejected_subsample_strategy, chosen_traj = result
                    strategy_used = selected_strategy
                    logger.trace(f"[PREF SAMPLER] Strategy {selected_strategy.value} succeeded on attempt {attempt}")
                else:
                    # Strategy failed - increment attempt count
                    strategy_attempt_counts[selected_strategy] = strategy_attempt_counts.get(selected_strategy, 0) + 1
                    failed_count = strategy_attempt_counts[selected_strategy]

                    logger.trace(
                        f"[PREF SAMPLER] Strategy {selected_strategy.value} failed (failure count: {failed_count}/{max_strategy_attempts})"
                    )

                    # Only remove strategy if it has failed max_strategy_attempts times
                    if strategy_attempt_counts[selected_strategy] >= max_strategy_attempts:
                        logger.trace(
                            f"[PREF SAMPLER] Removing strategy {selected_strategy.value} after {max_strategy_attempts} consecutive failures"
                        )
                        strategies = [(strat, prob) for strat, prob in strategies if strat != selected_strategy]
                        continue

            # If we still don't have a sample after all attempts, return None
            if rejected_traj is None:
                logger.trace(
                    f"[PREF SAMPLER] Failed to generate preference sample after {max_attempts} attempts - all strategies exhausted"
                )
                return None

        chosen_subsample_strategy = "subsample_forward"
        chosen_trajectory = self._get_traj_from_data(chosen_traj, subsample_strategy=chosen_subsample_strategy)

        rejected_trajectory = self._get_traj_from_data(rejected_traj, subsample_strategy=rejected_subsample_strategy)

        if rejected_trajectory is None or chosen_trajectory is None:
            return None

        # If our strategy is different task, make sure the rejected trajectory has 0 progress and 0 success labels
        if strategy_used in [
            DataGenStrat.DIFFERENT_TASK,
            DataGenStrat.DIFFERENT_TASK_INSTRUCTION,
        ]:
            rejected_trajectory.target_progress = [0.0] * len(rejected_trajectory.target_progress)
            if self.config.progress_loss_type.lower() == "discrete":
                rejected_trajectory.target_progress = convert_continuous_to_discrete_bins(
                    rejected_trajectory.target_progress, self.config.progress_discrete_bins
                )
            # Also set success labels to 0.0 (predict 0 success for different task trajectories)
            if rejected_trajectory.success_label is not None:
                rejected_trajectory.success_label = [0.0] * len(rejected_trajectory.success_label)

        # Create preference sample structure
        sample = PreferenceSample(
            chosen_trajectory=chosen_trajectory,
            rejected_trajectory=rejected_trajectory,
            data_gen_strategy=strategy_used.value,
        )
        sample.resample_attempts = attempt
        return sample
