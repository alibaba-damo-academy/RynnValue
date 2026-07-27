#!/usr/bin/env bash
# Evaluate the SFT-finetuned pi0.5 policy on the real Franka robot.
#
# Prerequisites:
#   1. On the robot machine:  bash scripts/franka/run_server.sh
#   2. SFT finished:          bash scripts/franka/finetune.sh
#
# Usage (env-var style):
#   bash scripts/franka/eval_policy.sh
#   ROBOT_IP=192.168.1.x SFT_CKPT_STEP=2000 bash scripts/franka/eval_policy.sh
#
# Usage (flag style):
#   bash scripts/franka/eval_policy.sh --robot-ip 192.168.1.x --ckpt-step 2000
#
# All flags (each also overridable via the matching env-var):
#   --robot-ip          Robot machine IP              [ROBOT_IP, default: localhost]
#   --robot-port        Env server port               [ROBOT_PORT, default: 8102]
#   --exp-name          SFT experiment name           [EXP_NAME, default: pick_up_the_bread_lora_sft]
#   --ckpt-step         Checkpoint step to load       [SFT_CKPT_STEP, default: 10000]
#   --ckpt-dir          Full checkpoint dir override  [SFT_CKPT_DIR]
#   --asset-id          LeRobot dataset repo id       [ASSET_ID, default: expo_ft/pick_up_the_bread]
#   --num-episodes      Number of eval episodes       [NUM_EPISODES, default: 20]
#   --replan-steps      Action re-planning frequency  [REPLAN_STEPS, default: 8]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ── Defaults ────────────────────────────────────────────────────────────────
ROBOT_IP="${ROBOT_IP:-localhost}"
ROBOT_PORT="${ROBOT_PORT:-8102}"
EXP_NAME="${EXP_NAME:-pick_up_the_bread_lora_sft}"
SFT_CKPT_STEP="${SFT_CKPT_STEP:-10000}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-}"
ASSETS_DIR="${ASSETS_DIR:-/path/to/assets/expo_pi05_franka_lora_sft}"
ASSET_ID="${ASSET_ID:-expo_ft/pick_up_the_bread}"
NUM_EPISODES="${NUM_EPISODES:-20}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"

# ── Parse CLI flags ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot-ip)       ROBOT_IP="$2";        shift 2 ;;
        --robot-ip=*)     ROBOT_IP="${1#*=}";   shift   ;;
        --robot-port)     ROBOT_PORT="$2";      shift 2 ;;
        --robot-port=*)   ROBOT_PORT="${1#*=}"; shift   ;;
        --exp-name)       EXP_NAME="$2";        shift 2 ;;
        --exp-name=*)     EXP_NAME="${1#*=}";   shift   ;;
        --ckpt-step)      SFT_CKPT_STEP="$2";  shift 2 ;;
        --ckpt-step=*)    SFT_CKPT_STEP="${1#*=}"; shift ;;
        --ckpt-dir)       SFT_CKPT_DIR="$2";   shift 2 ;;
        --ckpt-dir=*)     SFT_CKPT_DIR="${1#*=}"; shift ;;
        --asset-id)       ASSET_ID="$2";        shift 2 ;;
        --asset-id=*)     ASSET_ID="${1#*=}";   shift   ;;
        --num-episodes)   NUM_EPISODES="$2";    shift 2 ;;
        --num-episodes=*) NUM_EPISODES="${1#*=}"; shift ;;
        --replan-steps)   REPLAN_STEPS="$2";    shift 2 ;;
        --replan-steps=*) REPLAN_STEPS="${1#*=}"; shift ;;
        *) break ;;  # pass remaining args to eval_droid_policy.py
    esac
done

# Resolve checkpoint dir
if [[ -z "$SFT_CKPT_DIR" ]]; then
    SFT_CKPT_DIR="/path/to/checkpoints/expo_pi05_franka_lora_sft/$EXP_NAME/$SFT_CKPT_STEP"
fi
# ────────────────────────────────────────────────────────────────────────────

echo "Robot:       $ROBOT_IP:$ROBOT_PORT"
echo "Checkpoint:  $SFT_CKPT_DIR"
echo "Asset:       $ASSET_ID"
echo "Episodes:    $NUM_EPISODES  (replan every $REPLAN_STEPS steps)"
echo ""

.venv/bin/python eval_droid_policy.py \
    --config=configs/model/expo_ft_pi_config.py \
    --config_task=configs/task/franka_single.py \
    --config.pi05_config_name=expo_pi05_franka_lora_sft \
    --config.pi05_weight_loader_path="$SFT_CKPT_DIR/params" \
    --config.pi05_assets_dir="$ASSETS_DIR" \
    --config.pi05_asset_id="$ASSET_ID" \
    --checkpoint_dir="$SFT_CKPT_DIR" \
    --client_host="$ROBOT_IP" \
    --client_port="$ROBOT_PORT" \
    --num_episodes="$NUM_EPISODES" \
    --replan_steps="$REPLAN_STEPS" \
    --fsdp_devices=1 \
    "$@"
