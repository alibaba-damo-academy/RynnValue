#!/usr/bin/env bash
# Launch EXPO-FT online RL training for Franka single-arm (left arm).
#
# Prerequisites:
#   1. convert_data.sh + calculate_norm.sh + finetune.sh all finished
#   2. On the robot machine: bash scripts/franka/run_server.sh
#
# Usage:
#   cd /path/to/pi-rl
#   bash scripts/franka/train_online.sh
#   ROBOT_IP=192.168.1.x bash scripts/franka/train_online.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ── Configuration (edit these) ──────────────────────────────────────────────
ROBOT_IP="${ROBOT_IP:-localhost}"
ROBOT_PORT="${ROBOT_PORT:-8102}"
RUN_NAME="${RUN_NAME:-franka_expo_ft}"

# SFT checkpoint to warm-start from (produced by finetune.sh)
SFT_EXP_NAME="${SFT_EXP_NAME:-pick_up_the_bread_lora_sft}"
SFT_CKPT_STEP="${SFT_CKPT_STEP:-10000}"
SFT_WEIGHT_LOADER_PATH="/path/to/checkpoints/expo_pi05_franka_lora_sft/$SFT_EXP_NAME/$SFT_CKPT_STEP/params"

ASSETS_DIR="${ASSETS_DIR:-/path/to/assets/expo_pi05_franka_lora_sft}"
ASSET_ID="${ASSET_ID:-expo_ft/pick_up_the_bread}"

# Offline SFT data to seed the replay buffer (leave empty for pure online RL)
DATASET_PATH="${DATASET_PATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# ────────────────────────────────────────────────────────────────────────────

.venv/bin/python train_pi_robo_franka.py \
    --run_name="$RUN_NAME" \
    --client_host="$ROBOT_IP" \
    --client_port="$ROBOT_PORT" \
    --dataset_path="$DATASET_PATH" \
    --config=configs/model/expo_ft_pi_config.py \
    --config_task=configs/task/franka_single.py \
    --config.pi05_config_name=expo_pi05_franka_lora_sft \
    --config.pi05_weight_loader_path="$SFT_WEIGHT_LOADER_PATH" \
    --config.pi05_assets_dir="$ASSETS_DIR" \
    --config.pi05_asset_id="$ASSET_ID" \
    --max_steps=100000 \
    --batch_size=32 \
    --utd_ratio=20 \
    --replan_steps=8 \
    --update_type=episode \
    --num_updates=1 \
    --fsdp_devices=2 \
    --checkpoint_model \
    --overwrite \
    --checkpoint_interval=5000 \
    "$@"
