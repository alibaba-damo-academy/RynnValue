from typing import Dict, List, Any

from tqdm import tqdm

from robometer.data.samplers.eval.base_pref import BaseQualityPreferenceSampler
from robometer.utils.distributed import rank_0_print


class RoboArenaQualityPreferenceSampler(BaseQualityPreferenceSampler):
    """Dataset that generates preference samples by pairing trajectories with different partial_rewards for the same task.

    For RoboArena dataset, pairs trajectories from the same task where the chosen trajectory
    has a higher partial_reward (partial_success) than the rejected trajectory.
    """

    def __init__(
        self,
        comparisons_per_task=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Set data_gen_strategy for this sampler
        self.data_gen_strategy = "quality_preference_roboarena"

        self._cached_tasks = self.dataset["task"]
        self._cached_partial_success = self.dataset.get("partial_success")

        self.comparisons_per_task = comparisons_per_task

        # Generate all possible sample indices upfront (not the actual samples)
        self.sample_indices = self._generate_all_sample_indices()
        rank_0_print(
            f"Generated {len(self.sample_indices)} RoboArena quality preference sample indices", verbose=self.verbose
        )

    def _generate_all_sample_indices(self) -> List[Dict[str, Any]]:
        """Generate all possible quality preference sample indices based on partial_reward (partial_success)."""
        sample_indices = []

        # Group trajectories by task
        task_to_trajs = {}

        rank_0_print(
            f"Generating RoboArena quality preference samples for {len(self.robot_trajectories)} trajectories",
            verbose=self.verbose,
        )

        for traj_idx in self.robot_trajectories:
            # Use cached arrays for efficient access
            task = self._cached_tasks[traj_idx]
            partial_success = (
                self._cached_partial_success[traj_idx] if traj_idx < len(self._cached_partial_success) else None
            )

            # Ensure partial_success exists
            if partial_success is None:
                rank_0_print(
                    f"Warning: Trajectory {traj_idx} (task: {task}) missing partial_success, skipping",
                    verbose=self.verbose,
                )
                continue

            if task not in task_to_trajs:
                task_to_trajs[task] = []

            task_to_trajs[task].append({
                "traj_idx": traj_idx,
                "partial_success": float(partial_success),
            })

        # Generate pairs for each task
        for task in tqdm(task_to_trajs, desc="Generating RoboArena quality preference samples"):
            trajs = task_to_trajs[task]

            # Need at least 2 trajectories to create pairs
            if len(trajs) < 2:
                continue

            # Create all pairs of trajectories
            task_pairs = []
            for i, traj1 in enumerate(trajs):
                for j, traj2 in enumerate(trajs):
                    if i >= j:  # Avoid duplicates and self-pairs
                        continue

                    partial1 = traj1["partial_success"]
                    partial2 = traj2["partial_success"]

                    # Skip if partial_success values are equal (can't determine preference)
                    if partial1 == partial2:
                        continue

                    # Determine which trajectory is chosen (higher partial_success)
                    if partial1 > partial2:
                        chosen_traj_idx = traj1["traj_idx"]
                        rejected_traj_idx = traj2["traj_idx"]
                        chosen_partial = partial1
                        rejected_partial = partial2
                    else:
                        chosen_traj_idx = traj2["traj_idx"]
                        rejected_traj_idx = traj1["traj_idx"]
                        chosen_partial = partial2
                        rejected_partial = partial1

                    task_pairs.append({
                        "chosen_traj_idx": chosen_traj_idx,
                        "rejected_traj_idx": rejected_traj_idx,
                        "task": task,
                        "chosen_partial_success": chosen_partial,
                        "rejected_partial_success": rejected_partial,
                    })

            # Apply comparisons_per_task limit if set (sample uniformly across all pairs for this task)
            if self.comparisons_per_task is not None and len(task_pairs) > self.comparisons_per_task:
                # Uniformly sample comparisons for this task
                task_pairs = self._local_random.sample(task_pairs, self.comparisons_per_task)

            sample_indices.extend(task_pairs)

        return sample_indices
