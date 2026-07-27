#!/usr/bin/env bash
# Franka async training with local fake environment (no robot/server needed).
# Uses the SAME model config as train_online_async.sh for consistency.
#
# Usage:
#   cd /path/to/pi-rl
#   bash scripts/franka/run_fake_env_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ── Model configuration ───────────────────────────────────────────────────────
SFT_CKPT_DIR="${SFT_CKPT_DIR:-/path/to/checkpoints/franka_sft_abs/pi05_franka_single/pick_up_the_bread/5000}"
SFT_WEIGHT_LOADER_PATH="$SFT_CKPT_DIR/params"

ASSETS_DIR="${ASSETS_DIR:-$SFT_CKPT_DIR/assets}"
ASSET_ID="${ASSET_ID:-pick_up_the_bread}"

# ── Environment ───────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# ── Launch ────────────────────────────────────────────────────────────────────
.venv/bin/python train_pi_robo_franka_async.py \
    --fake_env \
    --run_name=franka_fake_test \
    --config=configs/model/expo_ft_pi_config.py \
    --config_task=configs/task/franka_single_fake.py \
    --config.pi05_config_name=expo_pi05_franka_single_lora_sft \
    --config.pi05_weight_loader_path="$SFT_WEIGHT_LOADER_PATH" \
    --config.pi05_assets_dir="$ASSETS_DIR" \
    --config.pi05_asset_id="$ASSET_ID" \
    --config.N=8 \
    --config.n_edit_samples=8 \
    --config.edit_scale=0.2 \
    --offline_ratio=0 \
    --max_steps=500 \
    --batch_size=12 \
    --utd_ratio=4 \
    --replan_steps=8 \
    --fsdp_devices=2 \
    --overwrite \
    --checkpoint_model=false \
    --project_name=expo-ft-franka-test \
    --tqdm \
    "$@"
