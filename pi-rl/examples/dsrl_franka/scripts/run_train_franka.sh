#!/bin/bash
# =============================================================================
# DSRL Franka Online RL Training — launch script examples
# =============================================================================
# Usage: uncomment the desired example block, then run:
#   bash examples/dsrl_franka/scripts/run_train_franka.sh
# Or make it executable and run it directly:
#   chmod +x examples/dsrl_franka/scripts/run_train_franka.sh
#   ./examples/dsrl_franka/scripts/run_train_franka.sh
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Common environment variables
# -----------------------------------------------------------------------------
# Base path of local SFT weights (if using a local checkpoint instead of GCS)
export FRANKA_SFT_CKPT_BASE="/path/to/checkpoints/pi05_franka_dual_iql_optimized_v2/iql_v2_pick_up_the_box/10000"
# Task name subdirectory (leave empty if CKPT_BASE is already the full path, no concatenation)
export FRANKA_SFT_TASK=""
# GPU device selection
export CUDA_VISIBLE_DEVICES=7
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# Disable JAX memory preallocation to avoid OOM during cuBLAS/cuSolver initialization
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Project root directory (relative to this script's location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# =============================================================================
# Example 1: Dual-arm training (real environment)
# =============================================================================
# Standard dual-arm Franka online RL training, connecting to the real robot environment server.

# python "${PROJECT_ROOT}/examples/dsrl_franka/launch_train_franka.py" \
#     --arm_mode dual \
#     --update_type episode \
#     --utd_ratio 20 \
#     --publish_hz 10.0 \
#     --client_host 192.168.1.100 \
#     --client_port 8102 \
#     --pi0_action_horizon 10 \
#     --franka_norm_stats_asset_id libero \
#     --max_episodes 1000 \
#     --franka_max_timesteps 500 \
#     --eval_interval 10 \
#     --eval_episodes 3 \
#     --checkpoint_interval 50 \
#     --checkpoint_dir "${PROJECT_ROOT}/checkpoints/franka_dual" \
#     --noise_episodes 5 \
#     --noise_std 0.1 \
#     --task_description "pick up the object" \
#     --wandb_project dsrl_franka \
#     --wandb_run_name "dual_arm_train" \
#     --seed 42

# =============================================================================
# Example 2: Single-arm training (real environment)
# =============================================================================
# Single-arm mode, training with the left arm.

# python "${PROJECT_ROOT}/examples/dsrl_franka/launch_train_franka.py" \
#     --arm_mode single \
#     --side left \
#     --policy_config pi05_franka_single_optimized_v2 \
#     --update_type episode \
#     --utd_ratio 20 \
#     --publish_hz 10.0 \
#     --client_host localhost \
#     --client_port 8101 \
#     --pi0_action_horizon 16 \
#     --franka_norm_stats_asset_id pick_up_the_bread \
#     --max_episodes 500 \
#     --franka_max_timesteps 600 \
#     --eval_interval 1000 \
#     --eval_episodes 2 \
#     --checkpoint_interval 100 \
#     --checkpoint_dir "${PROJECT_ROOT}/checkpoints/franka_single_left" \
#     --noise_episodes 2 \
#     --noise_std 0.1 \
#     --log_interval 50 \
#     --task_description "Pick up the two breads from the table and put them in the basket." \
#     --wandb_project dsrl_franka \
#     --wandb_run_name "single_left_train" \
#     --seed 42

# =============================================================================
# Example 3: Fake environment test
# =============================================================================
# Uses a local fake environment (no WebSocket server needed) to quickly verify training script correctness.

# python "${PROJECT_ROOT}/examples/dsrl_franka/launch_train_franka.py" \
#     --arm_mode single \
#     --fake_env \
#     --policy_config pi05_franka_single_optimized_v2 \
#     --update_type episode \
#     --utd_ratio 10 \
#     --publish_hz 10.0 \
#     --pi0_action_horizon 16 \
#     --franka_norm_stats_asset_id pick_up_the_bread \
#     --max_episodes 10 \
#     --franka_max_timesteps 50 \
#     --eval_interval 5 \
#     --eval_episodes 2 \
#     --checkpoint_interval 1000 \
#     --checkpoint_dir "${PROJECT_ROOT}/checkpoints/franka_fake_test" \
#     --noise_episodes 2 \
#     --noise_std 0.1 \
#     --log_interval 10 \
#     --task_description "Pick up the two breads from the table and put them in the basket." \
#     --wandb_project dsrl_franka_debug \
#     --wandb_run_name "fake_env_test" \
#     --seed 0

# =============================================================================
# Example 4: Single-arm training + Reward Shaping (real environment + progress reward server)
# =============================================================================
# Single-arm mode + external reward model server for reward shaping; start reward_service first.

python "${PROJECT_ROOT}/examples/dsrl_franka/launch_train_franka.py" \
    --arm_mode dual \
    --policy_config pi05_franka_dual_iql_optimized_v2 \
    --update_type episode \
    --utd_ratio 100 \
    --publish_hz 10.0 \
    --client_host localhost \
    --client_port 8101 \
    --pi0_action_horizon 16 \
    --franka_norm_stats_asset_id pick_up_the_box \
    --max_episodes 500 \
    --franka_max_timesteps 600 \
    --eval_interval 1000 \
    --eval_episodes 2 \
    --start_online_updates 200 \
    --checkpoint_interval 200 \
    --checkpoint_dir "${PROJECT_ROOT}/experiments/dual_reward_shaping_pick_up_the_box_robometer" \
    --noise_episodes 2 \
    --noise_std 0.1 \
    --log_interval 100 \
    --score_server "http://localhost:8000" \
    --shaping_weight 1.0 \
    --shaping_gamma 0.999 \
    --task_description "Move the box from the right side to the left side." \
    --wandb_project dsrl_franka \
    --wandb_run_name "dual_reward_shaping_pick_up_the_box_robometer" \
    --seed 42

# =============================================================================
# Example 5: Dual-arm training + Reward Shaping (real environment + progress reward server)
# =============================================================================
# Enables an external reward model server for reward shaping; start reward_service first.

# python "${PROJECT_ROOT}/examples/dsrl_franka/launch_train_franka.py" \
#     --arm_mode dual \
#     --update_type episode \
#     --utd_ratio 20 \
#     --publish_hz 10.0 \
#     --client_host 192.168.1.100 \
#     --client_port 8102 \
#     --pi0_action_horizon 10 \
#     --franka_norm_stats_asset_id libero \
#     --max_episodes 1000 \
#     --franka_max_timesteps 500 \
#     --eval_interval 10 \
#     --eval_episodes 3 \
#     --checkpoint_interval 50 \
#     --checkpoint_dir "${PROJECT_ROOT}/checkpoints/franka_dual_shaped" \
#     --noise_episodes 5 \
#     --noise_std 0.1 \
#     --score_server "http://localhost:8001" \
#     --shaping_weight 1.0 \
#     --shaping_gamma 0.999 \
#     --task_description "pick up the object" \
#     --wandb_project dsrl_franka \
#     --wandb_run_name "dual_arm_reward_shaping" \
#     --seed 42

echo "Please uncomment one of the example blocks above before running this script."
