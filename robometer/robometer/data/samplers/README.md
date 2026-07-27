# Sampler Strategies Documentation

This document summarizes the data generation strategies used by each sampler type in the RBM data pipeline.

## Overview

The codebase contains two main sampler types:
- **PrefSampler**: Generates preference prediction samples
- **ProgressSampler**: Generates progress prediction samples

Each sampler implements multiple strategies for generating training data, with automatic retry logic and strategy rebalancing on failure.

---

## PrefSampler (Preference Prediction)

The `PrefSampler` creates preference prediction samples with a chosen (preferred) trajectory and a rejected (suboptimal) trajectory.

### Strategies

#### 1. **REWIND**
- **Description**: Uses the same optimal trajectory for both chosen and rejected, but applies rewind subsampling to the rejected trajectory.
- **Purpose**: Learn that full trajectories are preferred over truncated/rewound versions.
- **Implementation**:
  - `chosen_trajectory`: Original optimal trajectory (forward subsampling)
  - `rejected_trajectory`: Same trajectory with `subsample_rewind` strategy

#### 2. **SUBOPTIMAL**
- **Description**: Uses an optimal trajectory as chosen and a suboptimal/failure trajectory from the same task as rejected.
- **Purpose**: Learn to prefer optimal trajectories over suboptimal ones from the same task.
- **Conditions**: Only available when suboptimal trajectories exist for the task
- **Implementation**:
  - `chosen_trajectory`: Optimal trajectory
  - `rejected_trajectory`: Suboptimal trajectory from same task (via `_get_same_task_suboptimal`)

#### 3. **DIFFERENT_TASK**
- **Description**: Uses an optimal trajectory as chosen and a trajectory from a completely different task as rejected.
- **Purpose**: Learn that trajectories from the same task are preferred over trajectories from different tasks.
- **Implementation**:
  - `chosen_trajectory`: Optimal trajectory
  - `rejected_trajectory`: Trajectory from different task (via `_get_different_video_traj`)
  - **Note**: Rejected trajectory's `target_progress` is set to `[0.0]` for all timesteps

#### 4. **REVERSE_PROGRESS**
- **Description**: Uses the same optimal trajectory for both chosen and rejected, but applies reverse uniform sampling to the rejected trajectory.
- **Purpose**: Learn that forward progress is preferred over reverse progress.
- **Implementation**:
  - `chosen_trajectory`: Original optimal trajectory (forward subsampling)
  - `rejected_trajectory`: Same trajectory with `subsample_reverse` strategy

#### 5. **ROBOARENA_PARTIAL_SUCCESS**
- **Description**: Uses two trajectories from the same task with different `partial_success` values. The trajectory with higher `partial_success` becomes chosen, and the one with lower `partial_success` becomes rejected.
- **Purpose**: Learn to prefer trajectories with higher partial success scores (RoboArena-specific).
- **Conditions**: Only available for RoboArena trajectories (has `partial_success` field and data_source contains "roboarena")
- **Implementation**:
  - Finds a different trajectory from same task (via `_get_different_partial_success_traj`)
  - Swaps trajectories if found trajectory has higher `partial_success`
  - `chosen_trajectory`: Trajectory with higher `partial_success`
  - `rejected_trajectory`: Trajectory with lower `partial_success`

### Special Handling
- **Non-successful trajectories**: If a trajectory has `quality_label != "successful"` (and is not RoboArena), it is automatically used as the rejected trajectory, with an optimal trajectory from the same task as the chosen trajectory.

### Strategy Selection
- Strategies are selected probabilistically based on `preference_strategy_ratio` configuration
- Probabilities are rebalanced when strategies fail
- Strategies are removed after 3 consecutive failures
- Maximum 10 total attempts per sample generation

---

## ProgressSampler (Progress Prediction)

The `ProgressSampler` creates progress prediction samples from a single trajectory, applying different subsampling strategies to create training data.

### Strategies

#### 1. **DIFFERENT_TASK_INSTRUCTION**
- **Description**: Uses a trajectory from a different task, but keeps the original task's embeddings and instruction.
- **Purpose**: Learn that progress should be 0.0 when the trajectory doesn't match the task instruction.
- **Implementation**:
  - Gets trajectory from different task (via `_get_different_task_instruction`)
  - Replaces embeddings with original task's embeddings
  - Sets `target_progress = [0.0]` for all timesteps
  - Uses forward subsampling

#### 2. **FORWARD_PROGRESS**
- **Description**: Samples the same trajectory with forward direction (start < middle < end).
- **Purpose**: Learn normal forward progress patterns.
- **Implementation**:
  - Uses same trajectory with `subsample_forward` strategy
  - Progress increases from start to end

#### 3. **REVERSE_PROGRESS**
- **Description**: Samples the same trajectory with reverse direction (end < middle < start).
- **Purpose**: Learn to handle reverse progress scenarios.
- **Implementation**:
  - Uses same trajectory with `subsample_reverse` strategy
  - Progress decreases from start to end

#### 4. **REWIND**
- **Description**: Samples the same trajectory with rewind direction (start < end < middle).
- **Purpose**: Learn to handle non-monotonic progress patterns.
- **Implementation**:
  - Uses same trajectory with `subsample_rewind` strategy
  - Progress pattern: increases, then decreases

### Strategy Selection
- Strategies are selected probabilistically based on `progress_strategy_ratio` configuration
- Probabilities are rebalanced when strategies fail
- Failed strategies are immediately removed (no retry count threshold)
- Maximum 10 total attempts per sample generation

---

## Common Features

### Retry Logic
All samplers implement retry logic with:
- Maximum attempt limits (typically 10 attempts)
- Strategy-specific retry counts (3-4 attempts per strategy)
- Automatic strategy removal after consecutive failures
- Probability rebalancing when strategies are removed

### Subsample Strategies
Common subsampling strategies used across samplers:
- `subsample_forward`: Normal forward sampling (start → end)
- `subsample_reverse`: Reverse sampling (end → start)
- `subsample_rewind`: Rewind sampling (start → end → start)

### Data Source Filtering
- Strategies may be filtered or boosted based on data source categories:
  - **Failure category**: Boosts SUBOPTIMAL strategy probability by 2x
  - **Paired category**: Boosts PAIRED_HUMAN_ROBOT strategy probability by 2x
  - **RoboArena**: Special handling for `partial_success` field
