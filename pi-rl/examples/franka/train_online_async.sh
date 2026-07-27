#!/usr/bin/env bash
# Launch EXPO-FT async online RL training for Franka (multi-GPU).
#
# Requires at least 2 GPUs: device[0] for inference, device[1:] for updates.
#
# Prerequisites:
#   1. convert_data.sh + calculate_norm.sh + finetune.sh all finished
#   2. On the robot machine: bash scripts/franka/run_server.sh
#
# Usage:
#   cd /path/to/pi-rl
#   bash scripts/franka/train_online_async.sh
#   ROBOT_IP=192.168.1.x bash scripts/franka/train_online_async.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ── Configuration (edit these) ──────────────────────────────────────────────
ROBOT_IP="${ROBOT_IP:-localhost}"
ROBOT_PORT="${ROBOT_PORT:-8101}"
RUN_NAME="${RUN_NAME:-franka_expo_ft_async}"

# SFT checkpoint to warm-start from
SFT_CKPT_DIR="${SFT_CKPT_DIR:-/path/to/checkpoints/franka_sft_abs/pi05_franka_single/pick_up_the_bread/5000}"
SFT_WEIGHT_LOADER_PATH="$SFT_CKPT_DIR/params"

ASSETS_DIR="${ASSETS_DIR:-$SFT_CKPT_DIR/assets}"
ASSET_ID="${ASSET_ID:-pick_up_the_bread}"

# Offline SFT data to seed the replay buffer (leave empty for pure online RL)
DATASET_PATH="${DATASET_PATH:-}"

# At least 2 GPUs required: device[0] for inference, device[1:] for training
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# ────────────────────────────────────────────────────────────────────────────

.venv/bin/python train_pi_robo_franka_async.py \
    --run_name="$RUN_NAME" \
    --client_host="$ROBOT_IP" \
    --client_port="$ROBOT_PORT" \
    --dataset_path="$DATASET_PATH" \
    --config=configs/model/expo_ft_pi_config.py \
    --config_task=configs/task/franka_single.py \
    --config.pi05_config_name=expo_pi05_franka_single_lora_sft \
    --config.pi05_weight_loader_path="$SFT_WEIGHT_LOADER_PATH" \
    --config.pi05_assets_dir="$ASSETS_DIR" \
    --config.pi05_asset_id="$ASSET_ID" \
    --max_steps=100000 \
    --batch_size=12 \
    --utd_ratio=20 \
    --replan_steps=8 \
    --fsdp_devices=6 \
    --checkpoint_model \
    --overwrite \
    --checkpoint_interval=5000 \
    "$@"
