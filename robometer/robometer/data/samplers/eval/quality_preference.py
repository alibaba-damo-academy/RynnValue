from typing import Dict, List, Any

from itertools import combinations
from tqdm import tqdm

from robometer.data.samplers.eval.base_pref import BaseQualityPreferenceSampler
from robometer.utils.distributed import rank_0_print


class QualityPreferenceSampler(BaseQualityPreferenceSampler):
    """Dataset that generates preference samples by pairing trajectories with different quality labels or partial_success values for the same task.

    For non-RoboArena: Pairs trajectories with different quality labels (failure, suboptimal, successful).
    For RoboArena: Pairs trajectories with different partial_success values (higher partial_success = chosen).
    """

    def __init__(
        self,
        comparisons_per_task=None,
        max_comparisons=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Set data_gen_strategy for this sampler
        self.data_gen_strategy = "quality_preference"
        self.comparisons_per_task = comparisons_per_task
        self.max_comparisons = max_comparisons

        # Generate all possible sample indices upfront (not the actual samples)
        self.sample_indices = self._generate_all_sample_indices()
        rank_0_print(f"Generated {len(self.sample_indices)} quality preference sample indices", verbose=self.verbose)

    def _generate_all_sample_indices(self) -> List[Dict[str, Any]]:
        """Generate all possible quality preference sample indices (not the actual samples).

        For non-RoboArena: Groups by task and quality_label, pairs trajectories with different quality labels.
        For RoboArena: Groups by task and partial_success values, pairs trajectories with different partial_success.
        """
        sample_indices = []

        # Check if this is RoboArena (has partial_success)
        is_roboarena = False
        if self.robot_trajectories:
            first_traj = self.dataset[self.robot_trajectories[0]]
            is_roboarena = first_traj.get("partial_success") is not None

        rank_0_print(
            f"Generating quality preference samples for {len(self.robot_trajectories)} trajectories "
            f"({'RoboArena (partial_success)' if is_roboarena else 'non-RoboArena (quality_label)'})",
            verbose=self.verbose,
        )

        if is_roboarena:
            # RoboArena: Group by task and partial_success (rounded to 2 decimals)
            task_to_partial_trajs = {}

            for traj_idx in self.robot_trajectories:
                traj = self.dataset[traj_idx]
                task = traj["task"]
                partial_success_val = traj.get("partial_success")

                if partial_success_val is None:
                    rank_0_print(
                        f"Warning: Trajectory {traj_idx} (task: {task}) missing partial_success, skipping",
                        verbose=self.verbose,
                    )
                    continue

                # Round partial_success to 2 decimals for grouping
                partial_success = round(float(partial_success_val), 2)

                if task not in task_to_partial_trajs:
                    task_to_partial_trajs[task] = {}

                if partial_success not in task_to_partial_trajs[task]:
                    task_to_partial_trajs[task][partial_success] = []

                task_to_partial_trajs[task][partial_success].append(traj_idx)

            # Generate pairs for each task
            for task in tqdm(task_to_partial_trajs, desc="Generating RoboArena quality preference samples"):
                partial_groups = task_to_partial_trajs[task]
                partial_values = list(partial_groups.keys())

                # Only create pairs if we have at least 2 different partial_success values
                if len(partial_values) < 2:
                    continue

                # Collect all pairs for this task
                task_pairs = []

                # Create pairs of different partial_success values
                for partial1, partial2 in combinations(partial_values, 2):
                    trajs1 = partial_groups[partial1]
                    trajs2 = partial_groups[partial2]

                    if not trajs1 or not trajs2:
                        continue

                    # Determine which partial_success is higher (chosen)
                    if partial1 > partial2:
                        chosen_partial = partial1
                        rejected_partial = partial2
                        chosen_indices = trajs1
                        rejected_indices = trajs2
                    elif partial2 > partial1:
                        chosen_partial = partial2
                        rejected_partial = partial1
                        chosen_indices = trajs2
                        rejected_indices = trajs1
                    else:
                        # Same value, skip this pair
                        continue

                    # Create all possible pairs for this partial_success combination
                    for chosen_idx in chosen_indices:
                        for rejected_idx in rejected_indices:
                            task_pairs.append({
                                "chosen_traj_idx": chosen_idx,
                                "rejected_traj_idx": rejected_idx,
                                "task": task,
                                "chosen_partial_success": chosen_partial,
                                "rejected_partial_success": rejected_partial,
                            })

                # Apply comparisons_per_task limit if set (sample uniformly across all pairs for this task)
                if self.comparisons_per_task is not None and len(task_pairs) > self.comparisons_per_task:
                    # Uniformly sample comparisons for this task
                    task_pairs = self._local_random.sample(task_pairs, self.comparisons_per_task)

                sample_indices.extend(task_pairs)

        else:
            # Non-RoboArena: Group by task and quality label
            task_to_quality_trajs = {}

            for traj_idx in self.robot_trajectories:
                traj = self.dataset[traj_idx]
                task = traj["task"]
                quality_label = traj["quality_label"]

                if task not in task_to_quality_trajs:
                    task_to_quality_trajs[task] = {}

                if quality_label not in task_to_quality_trajs[task]:
                    task_to_quality_trajs[task][quality_label] = []

                task_to_quality_trajs[task][quality_label].append(traj_idx)

            # Generate pairs for each task
            quality_order = {"failure": 1, "suboptimal": 2, "successful": 3}

            for task in tqdm(task_to_quality_trajs, desc="Generating quality preference samples"):
                quality_groups = task_to_quality_trajs[task]
                quality_labels = list(quality_groups.keys())

                # Only create pairs if we have at least 2 different quality labels
                if len(quality_labels) < 2:
                    continue

                # Collect all pairs for this task
                task_pairs = []

                # Create pairs of different quality labels
                for quality1, quality2 in combinations(quality_labels, 2):
                    trajs1 = quality_groups[quality1]
                    trajs2 = quality_groups[quality2]

                    if not trajs1 or not trajs2:
                        continue

                    # Determine which quality is better (chosen)
                    order1 = quality_order.get(quality1, 0)
                    order2 = quality_order.get(quality2, 0)

                    # Only create pairs if one quality is strictly better than the other
                    if order1 > order2:
                        chosen_quality = quality1
                        rejected_quality = quality2
                        chosen_indices = trajs1
                        rejected_indices = trajs2
                    elif order2 > order1:
                        chosen_quality = quality2
                        rejected_quality = quality1
                        chosen_indices = trajs2
                        rejected_indices = trajs1
                    else:
                        # Same order, skip this pair as we can't reliably compare them
                        continue

                    # Create all possible pairs for this quality combination
                    for chosen_idx in chosen_indices:
                        for rejected_idx in rejected_indices:
                            task_pairs.append({
                                "chosen_traj_idx": chosen_idx,
                                "rejected_traj_idx": rejected_idx,
                                "task": task,
                                "chosen_quality": chosen_quality,
                                "rejected_quality": rejected_quality,
                            })

                # Apply comparisons_per_task limit if set (sample uniformly across all pairs for this task)
                if self.comparisons_per_task is not None and len(task_pairs) > self.comparisons_per_task:
                    # Uniformly sample comparisons for this task
                    task_pairs = self._local_random.sample(task_pairs, self.comparisons_per_task)

                sample_indices.extend(task_pairs)

        # Apply max_comparisons limit if set (sample uniformly across all comparisons)
        original_count = len(sample_indices)
        if self.max_comparisons is not None and original_count > self.max_comparisons:
            sample_indices = self._local_random.sample(sample_indices, self.max_comparisons)
            rank_0_print(
                f"Limited total comparisons to {self.max_comparisons} (from {original_count} total comparisons)",
                verbose=self.verbose,
            )

        return sample_indices
