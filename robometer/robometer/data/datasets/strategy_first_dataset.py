from collections import defaultdict
from typing import Dict, List, Optional, Any
from random import Random

from robometer.data.datasets.base import BaseDataset
from robometer.data.samplers.pref import PrefSampler
from robometer.data.samplers.progress import ProgressSampler
from robometer.data.datasets.helpers import DataGenStrat
from robometer.data.dataset_category import (
    is_preference_only,
    is_suboptimal_fail_ds,
    is_paired_ds,
)
from robometer.utils.logger import get_logger

logger = get_logger()


class StrategyFirstDataset(BaseDataset):
    """
    Dataset that first selects sample type, then strategy, then picks a data source uniformly.

    This is different from RBMDataset which selects a trajectory first based on dataset iteration,
    and from StrategyBalancedDataset which selects sample type then data source (with optional weights).

    Sampling flow:
    1. Select sample type (preference/progress) based on sample_type_ratio
    2. Select strategy for that sample type based on strategy ratios
    3. Select data source uniformly from all available data sources
    4. Sample trajectory from selected data source and generate sample
    """

    def __init__(self, config, is_evaluation=False, max_samples=None, sampler_kwargs=None, random_seed=42, **kwargs):
        super().__init__(config, is_evaluation, **kwargs)

        # Initialize local random instance for deterministic sampling
        self._local_random = Random(random_seed)

        self.pref_sampler = None
        self.progress_sampler = None

        if sampler_kwargs is None:
            sampler_kwargs = {}

        base_sampler_kwargs = {
            "config": config,
            "dataset": self.dataset,
            "combined_indices": self._combined_indices,
            "dataset_success_cutoff_map": self.dataset_success_cutoff_map,
            "verbose": False,
            **sampler_kwargs,
        }

        if self.config.sample_type_ratio[0] > 0:
            self.pref_sampler = PrefSampler(is_evaluation=is_evaluation, **base_sampler_kwargs)
        if self.config.sample_type_ratio[1] > 0:
            self.progress_sampler = ProgressSampler(is_evaluation=is_evaluation, **base_sampler_kwargs)

        self.sample_type_ratio = config.sample_type_ratio
        self.max_samples = max_samples
        self.data_len = len(self.dataset)

        # Build source indices for efficient uniform sampling
        self.source_indices = defaultdict(list)
        if "data_source" in self.dataset.column_names:
            sources = self.dataset["data_source"]
            for i, source in enumerate(sources):
                self.source_indices[source].append(i)

        # Handle data source weights if provided
        self.data_source_weights = getattr(config, "data_source_weights", None)
        if self.data_source_weights:
            self._normalize_data_source_weights()
        else:
            self.normalized_weights = None

        # Build set of tasks with optimal trajectories for efficient filtering
        self.tasks_with_optimal = set(self._combined_indices.get("optimal_by_task", {}).keys())

        # Build set of tasks with both optimal and suboptimal trajectories for SUBOPTIMAL strategy
        suboptimal_by_task = self._combined_indices.get("suboptimal_by_task", {})
        # Only include tasks that have non-empty suboptimal indices
        tasks_with_suboptimal = {task for task, indices in suboptimal_by_task.items() if indices}
        self.tasks_with_both = self.tasks_with_optimal & tasks_with_suboptimal

        # Build set of all indices from tasks with optimal trajectories for efficient filtering
        task_indices = self._combined_indices.get("task_indices", {})
        self.optimal_task_indices = set()
        for task in self.tasks_with_optimal:
            if task in task_indices:
                self.optimal_task_indices.update(task_indices[task])

        # Build set of all indices from tasks with both optimal and suboptimal trajectories
        self.tasks_with_both_indices = set()
        for task in self.tasks_with_both:
            if task in task_indices:
                self.tasks_with_both_indices.update(task_indices[task])

        # Build set of successful trajectory indices for REWIND strategy filtering
        quality_indices = self._combined_indices.get("quality_indices", {})
        self.successful_indices = set(quality_indices.get("successful", []))

        logger.info(f"StrategyFirstDataset initialized with {len(self.dataset)} trajectories")
        logger.info(f"Sample type ratios: pref={self.sample_type_ratio[0]}, progress={self.sample_type_ratio[1]}")
        logger.info(f"Available data sources: {list(self.source_indices.keys())}")
        if self.normalized_weights:
            logger.info("Data source weights enabled:")
            for source, weight in sorted(self.normalized_weights.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {source}: {weight:.3f}")
        logger.info(f"Tasks with optimal trajectories: {len(self.tasks_with_optimal)}")
        logger.info(f"Tasks with both optimal and suboptimal trajectories: {len(self.tasks_with_both)}")

    def get_random_state(self) -> dict:
        """Get random state from dataset and all samplers for checkpointing.

        Returns:
            Dictionary containing random state for dataset and all samplers
        """
        state = {
            "dataset": self._local_random.getstate() if hasattr(self, "_local_random") else None,
            "pref_sampler": self.pref_sampler._local_random.getstate() if self.pref_sampler else None,
            "progress_sampler": self.progress_sampler._local_random.getstate() if self.progress_sampler else None,
        }
        return state

    def set_random_state(self, state: dict):
        """Restore random state from checkpoint.

        Args:
            state: Dictionary containing random state for dataset and all samplers
        """
        if state.get("dataset") and hasattr(self, "_local_random"):
            self._local_random.setstate(state["dataset"])
        if state.get("pref_sampler") and self.pref_sampler:
            self.pref_sampler._local_random.setstate(state["pref_sampler"])
        if state.get("progress_sampler") and self.progress_sampler:
            self.progress_sampler._local_random.setstate(state["progress_sampler"])

    def __len__(self):
        if self.max_samples is None:
            return self.data_len
        else:
            return self.max_samples

    def __getitem__(self, idx):
        """Create a sample by selecting sample type, then strategy, then data source uniformly."""
        logger.trace(f"[StrategyFirstDataset] __getitem__: Starting for idx={idx}")

        # Step 1: Select sample type based on ratios
        sample_type = self._select_sample_type()
        logger.trace(f"[StrategyFirstDataset] Selected sample type: {sample_type}")

        # Step 2: Select strategy for this sample type
        strategy = self._select_strategy(sample_type)
        if strategy is None:
            # Fallback: try to generate sample without specific strategy
            logger.trace(f"[StrategyFirstDataset] No strategy selected, using sampler default")
            return self._generate_without_specific_strategy(sample_type)

        logger.trace(
            f"[StrategyFirstDataset] Selected strategy: {strategy.value if hasattr(strategy, 'value') else strategy}"
        )

        # Step 3: Filter data sources based on strategy requirements
        filtered_sources = self._filter_data_sources_by_strategy(strategy)
        if not filtered_sources:
            logger.trace(
                f"[StrategyFirstDataset] No viable data sources for strategy {strategy.value if hasattr(strategy, 'value') else strategy}, retrying..."
            )
            # Retry by selecting a different strategy/sample type
            return self._generate_without_specific_strategy(sample_type)

        # Step 4-6: Sample and generate using helper method
        # First try with preferred strategy
        sample = self._try_generate_sample(
            sample_type=sample_type,
            filtered_sources=filtered_sources,
            strategy=strategy,
            preferred_strategy=strategy,
        )
        if sample is not None:
            return sample

        # If preferred strategy failed, try without strategy (let sampler choose its own)
        logger.trace(
            f"[StrategyFirstDataset] Failed to generate {sample_type} sample with preferred strategy {strategy.value if hasattr(strategy, 'value') else strategy}, "
            f"trying without strategy..."
        )
        sample = self._try_generate_sample(
            sample_type=sample_type,
            filtered_sources=filtered_sources,
            strategy=strategy,  # Keep strategy for filtering, but let sampler choose
            preferred_strategy=None,  # Let sampler select its own strategy
        )
        if sample is not None:
            return sample

        # All attempts failed for the selected sample type, try other samplers as fallback
        logger.trace(f"[StrategyFirstDataset] Failed to generate {sample_type} sample, trying other samplers...")
        return self._try_other_samplers(sample_type)

    def _select_sample_type(self) -> str:
        """Select a sample type based on sample_type_ratio."""
        available_types = []
        available_probs = []

        if self.sample_type_ratio[0] > 0 and self.pref_sampler is not None:
            available_types.append("pref")
            available_probs.append(self.sample_type_ratio[0])

        if self.sample_type_ratio[1] > 0 and self.progress_sampler is not None:
            available_types.append("progress")
            available_probs.append(self.sample_type_ratio[1])

        if not available_types:
            raise ValueError("No available sample types (all ratios are 0 or samplers are None)")

        # Normalize probabilities
        total_prob = sum(available_probs)
        normalized_probs = [p / total_prob for p in available_probs]

        # Select based on weighted random sampling
        prob = self._local_random.random()
        cumulative_prob = 0.0

        for sample_type, normalized_prob in zip(available_types, normalized_probs):
            cumulative_prob += normalized_prob
            if prob <= cumulative_prob:
                return sample_type

        # Fallback (should not reach here)
        return available_types[-1]

    def _select_strategy(self, sample_type: str) -> Optional[DataGenStrat]:
        """Select a strategy for the given sample type based on strategy ratios."""
        strategies = []
        strategy_ratios = []

        if sample_type == "pref":
            strategy_ratios = self.config.preference_strategy_ratio
            # Map ratios to strategies: [rewind, suboptimal_same_task, different_task, reverse_progress]
            if len(strategy_ratios) > 0 and strategy_ratios[0] > 0:
                strategies.append((DataGenStrat.REWIND, strategy_ratios[0]))
            if len(strategy_ratios) > 1 and strategy_ratios[1] > 0:
                strategies.append((DataGenStrat.SUBOPTIMAL, strategy_ratios[1]))
            if len(strategy_ratios) > 2 and strategy_ratios[2] > 0:
                strategies.append((DataGenStrat.DIFFERENT_TASK, strategy_ratios[2]))
            if len(strategy_ratios) > 3 and strategy_ratios[3] > 0:
                strategies.append((DataGenStrat.REVERSE_PROGRESS, strategy_ratios[3]))

        elif sample_type == "progress":
            strategy_ratios = self.config.progress_strategy_ratio
            # Map ratios to strategies: [different_task_instruction, forward_progress, reverse_progress, rewind]
            if len(strategy_ratios) > 0 and strategy_ratios[0] > 0:
                strategies.append((DataGenStrat.DIFFERENT_TASK_INSTRUCTION, strategy_ratios[0]))
            if len(strategy_ratios) > 1 and strategy_ratios[1] > 0:
                strategies.append((DataGenStrat.FORWARD_PROGRESS, strategy_ratios[1]))
            if len(strategy_ratios) > 2 and strategy_ratios[2] > 0:
                strategies.append((DataGenStrat.REVERSE_PROGRESS, strategy_ratios[2]))
            if len(strategy_ratios) > 3 and strategy_ratios[3] > 0:
                strategies.append((DataGenStrat.REWIND, strategy_ratios[3]))

        if not strategies:
            return None

        # Normalize probabilities
        total_prob = sum(prob for _, prob in strategies)
        if total_prob == 0:
            return None

        normalized_strategies = [(strat, prob / total_prob) for strat, prob in strategies]

        # Select based on weighted random sampling
        prob = self._local_random.random()
        cumulative_prob = 0.0

        for strategy, normalized_prob in normalized_strategies:
            cumulative_prob += normalized_prob
            if prob <= cumulative_prob:
                return strategy

        # Fallback
        return strategies[0][0]

    def _normalize_data_source_weights(self):
        """Normalize data source weights across all available sources."""
        available_sources = list(self.source_indices.keys())

        if not available_sources:
            self.normalized_weights = {}
            return

        # Get weights for available sources
        weights = {}
        total_weight = 0.0

        for source in available_sources:
            weight = self.data_source_weights.get(source, 1.0)
            weights[source] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            self.normalized_weights = {source: weight / total_weight for source, weight in weights.items()}
        else:
            # Equal weights fallback
            equal_weight = 1.0 / len(available_sources)
            self.normalized_weights = {source: equal_weight for source in available_sources}

    def _select_data_source(self, allowed_sources: Optional[List[str]] = None) -> str:
        """Select a data source from allowed data sources, using weights if available.

        Args:
            allowed_sources: Optional list of allowed data source names. If None, uses all available sources.

        Returns:
            Selected data source name
        """
        if allowed_sources is None:
            available_sources = list(self.source_indices.keys())
        else:
            # Filter to only include sources that exist in our source_indices
            available_sources = [source for source in allowed_sources if source in self.source_indices]

        if not available_sources:
            raise ValueError("No available data sources")

        if len(available_sources) == 1:
            return available_sources[0]

        # Use weighted selection if weights are provided
        if self.normalized_weights:
            # Re-normalize weights for filtered sources
            filtered_weights = {}
            total_weight = 0.0
            for source in available_sources:
                weight = self.normalized_weights.get(source, 0.0)
                if weight > 0:
                    filtered_weights[source] = weight
                    total_weight += weight

            if total_weight > 0:
                # Use re-normalized weights for filtered sources
                prob = self._local_random.random()
                cumulative_prob = 0.0

                for source in available_sources:
                    weight = filtered_weights.get(source, 0.0)
                    if weight <= 0:
                        continue
                    normalized_weight = weight / total_weight
                    cumulative_prob += normalized_weight
                    if prob <= cumulative_prob:
                        return source

                # Fallback: return last source with positive weight
                for source in reversed(available_sources):
                    if filtered_weights.get(source, 0.0) > 0:
                        return source

        # Uniform selection fallback
        return self._local_random.choice(available_sources)

    def _filter_data_sources_by_strategy(self, strategy: Optional[DataGenStrat]) -> List[str]:
        """Filter data sources based on strategy requirements.

        Args:
            strategy: The selected strategy

        Returns:
            List of viable data source names for the strategy
        """
        all_sources = list(self.source_indices.keys())

        if strategy is None:
            return all_sources

        # Filter based on strategy requirements
        if strategy == DataGenStrat.SUBOPTIMAL:
            # SUBOPTIMAL strategy needs data sources with suboptimal/failure trajectories
            filtered = [source for source in all_sources if is_suboptimal_fail_ds(source)]
            logger.trace(
                f"[StrategyFirstDataset] Filtered {len(filtered)}/{len(all_sources)} data sources for SUBOPTIMAL strategy"
            )
            return filtered if filtered else all_sources

        elif strategy == DataGenStrat.PAIRED_HUMAN_ROBOT:
            # PAIRED_HUMAN_ROBOT strategy needs paired data sources
            filtered = [source for source in all_sources if is_paired_ds(source)]
            logger.trace(
                f"[StrategyFirstDataset] Filtered {len(filtered)}/{len(all_sources)} data sources for PAIRED_HUMAN_ROBOT strategy"
            )
            return filtered if filtered else all_sources

        # Other strategies (REWIND, DIFFERENT_TASK, REVERSE_PROGRESS, etc.) can work with any data source
        return all_sources

    def _filter_indices_by_strategy(
        self, indices: List[int], data_source: str, sample_type: str, strategy: Optional[DataGenStrat]
    ) -> List[int]:
        """Filter indices based on strategy requirements.

        For SUBOPTIMAL strategy (preference), filters to only include indices from tasks
        that have optimal trajectories (unless RoboArena).

        Args:
            indices: List of trajectory indices to filter
            data_source: The data source name
            sample_type: The sample type (pref/progress)
            strategy: The selected strategy

        Returns:
            Filtered list of viable indices for the strategy
        """
        if strategy is None:
            return indices

        # For strategies that require successful trajectories only
        # REWIND: all sample types
        # REVERSE_PROGRESS: preference and progress samples
        # FORWARD_PROGRESS: progress samples only
        requires_successful = (
            strategy == DataGenStrat.REWIND
            or (strategy == DataGenStrat.REVERSE_PROGRESS and sample_type in ["pref", "progress"])
            or (strategy == DataGenStrat.FORWARD_PROGRESS and sample_type == "progress")
        )

        if requires_successful:
            indices_set = set(indices)
            filtered = indices_set & self.successful_indices

            if not filtered:
                logger.trace(
                    f"[StrategyFirstDataset] No successful trajectories available for {strategy.value} strategy in source {data_source}"
                )
                return []

            logger.trace(
                f"[StrategyFirstDataset] Filtered {len(filtered)}/{len(indices)} indices for {strategy.value} strategy "
                f"(keeping only successful trajectories)"
            )
            return list(filtered)

        # Check if this is RoboArena or RoboReward (skip task filtering for these datasets)
        is_roboarena = data_source and "roboarena" in str(data_source).lower()
        is_roboreward = data_source and "roboreward" in str(data_source).lower()

        # For SUBOPTIMAL strategy (preference), filter to tasks with both optimal and suboptimal trajectories
        if strategy == DataGenStrat.SUBOPTIMAL and sample_type == "pref":
            # Skip task filtering for RoboArena and RoboReward (they use partial_success logic)
            if is_roboarena or is_roboreward:
                return indices

            if not self.tasks_with_both:
                # No tasks with both optimal and suboptimal trajectories, return empty list
                logger.trace(
                    f"[StrategyFirstDataset] No tasks with both optimal and suboptimal trajectories available for SUBOPTIMAL strategy"
                )
                return []

            # Use pre-computed tasks_with_both_indices and intersect with our current indices
            indices_set = set(indices)
            filtered = self.tasks_with_both_indices & indices_set

            if not filtered:
                logger.trace(f"[StrategyFirstDataset] No viable indices after filtering for SUBOPTIMAL strategy")
                return []

            logger.trace(
                f"[StrategyFirstDataset] Filtered {len(filtered)}/{len(indices)} indices for SUBOPTIMAL strategy "
                f"(keeping only tasks with both optimal and suboptimal trajectories)"
            )
            return list(filtered)

        # Other strategies don't require task-level filtering
        return indices

    def _generate_sample_for_type(
        self, sample_type: str, item: Dict[str, Any], preferred_strategy: Optional[DataGenStrat] = None
    ):
        """Generate a sample using the appropriate sampler for the sample type.

        Args:
            sample_type: The sample type (pref/progress)
            item: The trajectory item
            preferred_strategy: Optional strategy to use (if None, sampler will select its own)
        """
        data_source = item["data_source"]
        quality_label = item["quality_label"]

        # Get the appropriate sampler
        if sample_type == "pref":
            sampler = self.pref_sampler
        elif sample_type == "progress":
            sampler = self.progress_sampler
        else:
            return None

        if sampler is None:
            return None

        # Handle non-successful trajectories: force preference-only
        if quality_label != "successful" and sample_type != "pref":
            if self.pref_sampler is not None:
                logger.trace(f"[StrategyFirstDataset] Non-successful quality detected, switching to preference sampler")
                return self.pref_sampler._generate_sample(item, preferred_strategy=preferred_strategy)
            else:
                return None

        # Handle preference-only data sources
        if is_preference_only(data_source) and sample_type != "pref":
            if self.pref_sampler is not None:
                logger.trace(
                    f"[StrategyFirstDataset] Preference-only data source detected, switching to preference sampler"
                )
                return self.pref_sampler._generate_sample(item, preferred_strategy=preferred_strategy)
            else:
                return None

        # Generate sample using the selected sampler with preferred strategy
        return sampler._generate_sample(item, preferred_strategy=preferred_strategy)

    def _try_generate_sample(
        self,
        sample_type: str,
        filtered_sources: Optional[List[str]] = None,
        strategy: Optional[DataGenStrat] = None,
        preferred_strategy: Optional[DataGenStrat] = None,
        max_attempts: int = 10,
    ):
        """Helper method to try generating a sample with retry logic.

        Args:
            sample_type: The sample type to generate (pref/progress)
            filtered_sources: Optional list of allowed data sources. If None, uses all sources.
            strategy: Optional strategy for filtering indices. If None, no index filtering.
            preferred_strategy: Optional strategy to pass to sampler. If None, sampler selects its own.
            max_attempts: Maximum number of attempts before giving up.

        Returns:
            Generated sample if successful, None otherwise.
        """
        for attempt in range(max_attempts):
            selected_source = self._select_data_source(filtered_sources)
            source_indices = self.source_indices.get(selected_source)

            if not source_indices:
                logger.trace(f"[StrategyFirstDataset] No indices for source {selected_source}, retrying...")
                continue

            # Filter indices based on strategy requirements if strategy is provided
            if strategy is not None:
                filtered_indices = self._filter_indices_by_strategy(
                    source_indices, selected_source, sample_type, strategy
                )
                if not filtered_indices:
                    logger.trace(
                        f"[StrategyFirstDataset] No viable indices after strategy filtering for source {selected_source}, retrying..."
                    )
                    continue
            else:
                filtered_indices = source_indices

            # Select a trajectory from filtered indices
            # For progress: trajectory for progress prediction; for preference: chosen trajectory
            selected_traj_idx = self._local_random.choice(filtered_indices)
            item = self.dataset[selected_traj_idx]

            traj_id = item["id"]
            data_source = item["data_source"]
            quality_label = item["quality_label"]

            strategy_str = strategy.value if strategy and hasattr(strategy, "value") else strategy
            logger.trace(
                f"[StrategyFirstDataset] Attempt {attempt + 1}/{max_attempts}: "
                f"Selected traj ID={traj_id}, source={data_source}, quality={quality_label}, "
                f"sample_type={sample_type}, strategy={strategy_str}"
            )

            # Generate sample using the selected sampler with the preferred strategy
            sample = self._generate_sample_for_type(sample_type, item, preferred_strategy=preferred_strategy)
            if sample is not None:
                # Check if the generated sample matches our preferred strategy (if available)
                generated_strategy = getattr(sample, "data_gen_strategy", None)
                if generated_strategy and preferred_strategy:
                    logger.trace(
                        f"[StrategyFirstDataset] Generated sample with strategy {generated_strategy} "
                        f"(preferred: {preferred_strategy.value if hasattr(preferred_strategy, 'value') else preferred_strategy})"
                    )
                logger.trace(f"[StrategyFirstDataset] Successfully generated {sample_type} sample for ID={traj_id}")
                return self._set_resample_attempts(sample, attempt + 1)

            logger.trace(f"[StrategyFirstDataset] Sampler returned None for ID={traj_id}, retrying...")

        return None

    def _try_other_samplers(self, failed_sample_type: str):
        """Try other available samplers when the selected one fails.

        Args:
            failed_sample_type: The sample type that failed

        Returns:
            A sample from one of the other samplers, or raises ValueError if all fail
        """
        # Get list of available samplers excluding the one that failed
        available_samplers = []
        if failed_sample_type != "pref" and self.pref_sampler is not None:
            available_samplers.append("pref")
        if failed_sample_type != "progress" and self.progress_sampler is not None:
            available_samplers.append("progress")

        if not available_samplers:
            logger.error(f"[StrategyFirstDataset] No other samplers available after {failed_sample_type} failed")
            raise ValueError(f"Failed to generate {failed_sample_type} sample and no other samplers available")

        # Try each available sampler
        for fallback_sample_type in available_samplers:
            logger.trace(f"[StrategyFirstDataset] Trying fallback sampler: {fallback_sample_type}")
            sample = self._try_generate_sample(
                sample_type=fallback_sample_type,
                filtered_sources=None,
                strategy=None,
                preferred_strategy=None,
            )
            if sample is not None:
                logger.trace(f"[StrategyFirstDataset] Fallback sampler {fallback_sample_type} succeeded")
                return sample

            logger.trace(f"[StrategyFirstDataset] Fallback sampler {fallback_sample_type} failed")

        # All fallback samplers failed
        logger.error(f"[StrategyFirstDataset] All samplers (including fallbacks) failed")
        raise ValueError(f"Failed to generate {failed_sample_type} sample and all fallback samplers also failed")

    def _generate_without_specific_strategy(self, sample_type: str):
        """Fallback method to generate sample without specific strategy selection."""
        sample = self._try_generate_sample(
            sample_type=sample_type,
            filtered_sources=None,
            strategy=None,
            preferred_strategy=None,
        )
        if sample is not None:
            return sample

        # If this also fails, try other samplers
        logger.trace(
            f"[StrategyFirstDataset] _generate_without_specific_strategy failed for {sample_type}, trying other samplers..."
        )
        return self._try_other_samplers(sample_type)

    def get_resample_attempt_stats(self):
        return self._resample_attempt_stats

    def get_resample_dataset_attempt_stats(self):
        return self._resample_dataset_attempt_stats
