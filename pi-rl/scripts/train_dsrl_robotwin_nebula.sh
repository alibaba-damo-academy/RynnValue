#!/bin/bash
# DSRL training for pi0.5 on RoboTwin inside a Nebula container (no conda).
#
# Env vars (set by run_nebula_dsrl_robotwin.py):
#   TASK_NAME    - RoboTwin task name (required)
#   TASK_CONFIG  - task_config yml stem (default: demo_clean)
#   PI0_CKPT     - pi0/pi05 checkpoint path
#   SCORE_SERVER - reward model server URL (empty to disable)
#   EXP_NAME     - experiment name for wandb (auto-derived if empty)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "\033[32m[Info] Working directory: $PROJECT_ROOT\033[0m"

PYTHON="${PYTHON:-python3}"
TASK_NAME="${TASK_NAME:?TASK_NAME is required}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
PI0_CKPT="${PI0_CKPT:-/path/to/checkpoints/pi05_robotwin_all/all_50/30000}"
SCORE_SERVER="${SCORE_SERVER:-}"
EXP_NAME="${EXP_NAME:-dsrl_pi0_robotwin_${TASK_NAME}}"

echo "==> PYTHON       = ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "==> TASK_NAME    = ${TASK_NAME}"
echo "==> TASK_CONFIG  = ${TASK_CONFIG}"
echo "==> PI0_CKPT     = ${PI0_CKPT}"
echo "==> SCORE_SERVER = ${SCORE_SERVER}"
echo "==> EXP_NAME     = ${EXP_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/path/to/openpi_data}"
export EXP="${EXP:-/path/to/dsrl_logs/${EXP_NAME}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SCORE_ARGS=()
if [[ -n "$SCORE_SERVER" ]]; then
    SCORE_ARGS+=(--score_server "$SCORE_SERVER")
fi

"${PYTHON}" examples/dsrl_robotwin/launch_train_sim.py \
    --algorithm pixel_sac \
    --env robotwin \
    --task_name "$TASK_NAME" \
    --task_config "$TASK_CONFIG" \
    --pi0_checkpoint_dir "$PI0_CKPT" \
    --pi0_action_horizon 10 \
    --robotwin_state_dim 14 \
    --prefix dsrl_pi0_robotwin \
    --wandb_project DSRL_pi0_RoboTwin \
    --batch_size 256 \
    --discount 0.999 \
    --seed 0 \
    --max_steps 500000 \
    --eval_interval 10000 \
    --log_interval 500 \
    --eval_episodes 10 \
    --multi_grad_step 20 \
    --start_online_updates 500 \
    --resize_image 64 \
    --action_magnitude 1.0 \
    --query_freq 10 \
    "${SCORE_ARGS[@]}"
