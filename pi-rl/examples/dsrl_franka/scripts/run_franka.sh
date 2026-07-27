#!/bin/bash
# DSRL on Franka dual-arm: SAC actor learns the pi0 latent noise that drives a
# frozen pi05_libero diffusion policy on a real Franka robot via ROS2.
# Sister of examples/dsrl_robotwin/scripts/run_robotwin.sh; here we reset/step
# through FrankaEnv (EXPO-FT) instead of SAPIEN.
set -e

proj_name=DSRL_pi0_Franka
device_id=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || (cd "$SCRIPT_DIR/../../.." && pwd))"

# Pi0 checkpoint — defaults to the public pi05_libero checkpoint, which the
# Franka client (franka_client_sync.py) was designed around.
PI0_CKPT=${PI0_CKPT:-$(python3 -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))")}
# Reward model server URL for progress-based reward shaping. Empty = sparse only.
SCORE_SERVER=${SCORE_SERVER:-}
# Language instruction sent to the pi0 model and FrankaEnv.
TASK_DESCRIPTION=${TASK_DESCRIPTION:-"pick up the object"}

export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$REPO_ROOT/openpi}
export EXP=$REPO_ROOT/logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 examples/dsrl_franka/launch_train_franka.py \
  --algorithm pixel_sac \
  --env franka \
  --pi0_checkpoint_dir $PI0_CKPT \
  --pi0_action_horizon 10 \
  --franka_state_dim 16 \
  --franka_max_timesteps 500 \
  --task_description "$TASK_DESCRIPTION" \
  --prefix dsrl_pi0_franka \
  --wandb_project ${proj_name} \
  --batch_size 256 \
  --discount 0.999 \
  --seed 0 \
  --max_steps 500000 \
  --eval_interval 10000 \
  --log_interval 500 \
  --eval_episodes 5 \
  --multi_grad_step 20 \
  --start_online_updates 500 \
  --resize_image 64 \
  --action_magnitude 1.0 \
  --query_freq 10 \
  --hidden_dims 128 \
  --score_server "$SCORE_SERVER"
