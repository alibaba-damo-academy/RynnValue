#!/usr/bin/env bash
# SFT (supervised fine-tuning) of pi0.5 LoRA on Franka real-robot data.
#
# Run AFTER calculate_norm.sh has finished.
#
# Usage:
#   cd /path/to/pi-rl
#   bash scripts/franka/finetune.sh
#
# The checkpoint will be saved to:
#   /path/to/checkpoints/expo_pi05_franka_lora_sft/<EXP_NAME>/<STEP>/params

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ── Configuration (edit these) ──────────────────────────────────────────────
REPO_ID="${REPO_ID:-expo_ft/pick_up_the_bread}"
EXP_NAME="${EXP_NAME:-pick_up_the_bread_lora_sft}"
ASSETS_DIR="${ASSETS_DIR:-/path/to/assets/expo_pi05_franka_lora_sft}"
ASSET_ID="${ASSET_ID:-$REPO_ID}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
# Use the platform (cudaMalloc) allocator instead of JAX's BFC allocator.
# BFC grows to fill 32GB then fragments; platform allocator gives each alloc its
# own CUDA buffer, avoiding OOM caused by fragmentation on single-GPU training.
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
# ────────────────────────────────────────────────────────────────────────────

source .venv/bin/activate

uv run expo_ft/agents/vla/openpi/scripts/train.py expo_pi05_franka_lora_sft \
    --exp-name="$EXP_NAME" \
    --resume \
    --data.repo_id="$REPO_ID" \
    --data.assets.assets_dir="$ASSETS_DIR" \
    --data.assets.asset_id="$ASSET_ID" \
    --num_train_steps="$NUM_TRAIN_STEPS" \
    --save_interval="$SAVE_INTERVAL" \
    --batch_size="$BATCH_SIZE" \
    --fsdp_devices="$FSDP_DEVICES"

echo ""
echo "✓ SFT complete. Checkpoint saved at:"
echo "  /path/to/checkpoints/expo_pi05_franka_lora_sft/$EXP_NAME/$NUM_TRAIN_STEPS/params"
echo ""
echo "Next steps:"
echo "  Evaluate:     bash scripts/franka/eval_policy.sh"
echo "  Online RL:    bash scripts/franka/train_online.sh"

# CUDA_VISIBLE_DEVICES=0 FSDP_DEVICES=1 BATCH_SIZE=16 bash scripts/franka/finetune.sh