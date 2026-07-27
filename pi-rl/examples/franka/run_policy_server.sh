#!/usr/bin/env bash
# Launch the pi0.5 policy inference server for the Franka real robot.
#
# This starts a WebSocket policy server that the robot control machine
# connects to for action inference. It does NOT run the robot env itself.
#
# Architecture:
#   [GPU machine]   bash scripts/franka/run_policy_server.sh   → port 8000
#   [Robot machine] bash scripts/franka/run_server.sh          → port 8102
#
# Usage (env-var style):
#   bash scripts/franka/run_policy_server.sh
#   SFT_CKPT_STEP=2000 bash scripts/franka/run_policy_server.sh
#
# Usage (flag style):
#   bash scripts/franka/run_policy_server.sh --ckpt-step 2000 --lang "pick up bread"
#
# All flags (each also overridable via the matching env-var):
#   --exp-name          SFT experiment name          [SFT_EXP_NAME, default: pick_up_the_bread]
#   --ckpt-step         Checkpoint step to load      [SFT_CKPT_STEP, default: 19999]
#   --ckpt-dir          Full checkpoint dir override [SFT_CKPT_DIR]
#   --policy-config     OpenPI config name           [POLICY_CONFIG, default: pi05_franka_single]
#   --lang              Language instruction          [LANGUAGE_INSTRUCTION, default: "pick up the bread"]
#   --port              Policy server port            [POLICY_PORT, default: 8100]
#   --gpu               CUDA_VISIBLE_DEVICES          [CUDA_VISIBLE_DEVICES, default: 0]
#   --arm-mode          single or dual arm output     [ARM_MODE, default: single]
#   --dsrl-ckpt         DSRL SAC checkpoint directory  [DSRL_CKPT_DIR]
#   --dsrl-arm-mode     DSRL arm mode for obs process  [DSRL_ARM_MODE, default: same as --arm-mode]
#   --dsrl-resize       DSRL SAC encoder image size    [DSRL_RESIZE_IMAGE, default: 64]
#   --no-state-as-input Disable state input (zero out proprioception)  [FRANKA_NO_STATE_INPUT]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ── Defaults ────────────────────────────────────────────────────────────────
SFT_EXP_NAME="${SFT_EXP_NAME:-pick_up_the_bread}"
SFT_CKPT_STEP="${SFT_CKPT_STEP:-10000}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_franka_single_optimized_v2}"
POLICY_PORT="${POLICY_PORT:-8100}"
LANGUAGE_INSTRUCTION="${LANGUAGE_INSTRUCTION:-Pick up the two breads from the table and put it in the basket.}"
ARM_MODE="${ARM_MODE:-single}"
DSRL_CKPT_DIR="${DSRL_CKPT_DIR:-}"
DSRL_ARM_MODE="${DSRL_ARM_MODE:-}"
DSRL_RESIZE_IMAGE="${DSRL_RESIZE_IMAGE:-64}"
FRANKA_NO_STATE_INPUT="${FRANKA_NO_STATE_INPUT:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── Parse CLI flags ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name)   SFT_EXP_NAME="$2";        shift 2 ;;
        --exp-name=*) SFT_EXP_NAME="${1#*=}";   shift   ;;
        --ckpt-step)  SFT_CKPT_STEP="$2";       shift 2 ;;
        --ckpt-step=*)SFT_CKPT_STEP="${1#*=}";  shift   ;;
        --ckpt-dir)   SFT_CKPT_DIR="$2";        shift 2 ;;
        --ckpt-dir=*) SFT_CKPT_DIR="${1#*=}";   shift   ;;
        --policy-config) POLICY_CONFIG="$2";     shift 2 ;;
        --policy-config=*) POLICY_CONFIG="${1#*=}"; shift ;;
        --lang)       LANGUAGE_INSTRUCTION="$2"; shift 2 ;;
        --lang=*)     LANGUAGE_INSTRUCTION="${1#*=}"; shift ;;
        --port)       POLICY_PORT="$2";          shift 2 ;;
        --port=*)     POLICY_PORT="${1#*=}";     shift   ;;
        --gpu)        export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        --gpu=*)      export CUDA_VISIBLE_DEVICES="${1#*=}"; shift ;;
        --arm-mode)   ARM_MODE="$2";              shift 2 ;;
        --arm-mode=*) ARM_MODE="${1#*=}";         shift   ;;
        --dsrl-ckpt)  DSRL_CKPT_DIR="$2";         shift 2 ;;
        --dsrl-ckpt=*) DSRL_CKPT_DIR="${1#*=}";   shift   ;;
        --dsrl-arm-mode) DSRL_ARM_MODE="$2";      shift 2 ;;
        --dsrl-arm-mode=*) DSRL_ARM_MODE="${1#*=}"; shift ;;
        --dsrl-resize) DSRL_RESIZE_IMAGE="$2";    shift 2 ;;
        --dsrl-resize=*) DSRL_RESIZE_IMAGE="${1#*=}"; shift ;;
        --no-state-as-input) FRANKA_NO_STATE_INPUT="1"; shift ;;
        *) break ;;  # pass remaining args to serve_policy.py
    esac
done

# Resolve checkpoint dir (can be fully overridden by --ckpt-dir or SFT_CKPT_DIR)
if [[ -z "$SFT_CKPT_DIR" ]]; then
    SFT_CKPT_DIR="/path/to/checkpoints/franka_sft_optimized_v2/pi05_franka_single_optimized_v2/pick_up_the_bread_v2/10000"
fi

# Map arm mode to output action dimension
FRANKA_EXTRACT_DIM=8
case "$ARM_MODE" in
    dual)  FRANKA_EXTRACT_DIM=16 ;;
    single|*) FRANKA_EXTRACT_DIM=8 ;;
esac
# ────────────────────────────────────────────────────────────────────────────

echo "Loading SFT checkpoint: $SFT_CKPT_DIR"
echo "Policy config:          $POLICY_CONFIG"
echo "Language instruction:   $LANGUAGE_INSTRUCTION"
echo "Serving on port:        $POLICY_PORT  (GPU: $CUDA_VISIBLE_DEVICES)"
echo "Arm mode:               $ARM_MODE  (output action dim: $FRANKA_EXTRACT_DIM)"
if [[ -n "$DSRL_CKPT_DIR" ]]; then
    echo "DSRL checkpoint:        $DSRL_CKPT_DIR"
    echo "DSRL arm mode:          ${DSRL_ARM_MODE:-$ARM_MODE}"
    echo "DSRL resize:            $DSRL_RESIZE_IMAGE"
fi
if [[ "$FRANKA_NO_STATE_INPUT" == "1" ]]; then
    echo "State input:            DISABLED (no proprioception)"
fi
echo ""

source .venv/bin/activate

# Build DSRL args if checkpoint is specified
DSRL_ARGS=""
if [[ -n "$DSRL_CKPT_DIR" ]]; then
    DSRL_ARGS="--dsrl-ckpt-dir=$DSRL_CKPT_DIR --dsrl-arm-mode=${DSRL_ARM_MODE:-$ARM_MODE} --dsrl-resize-image=$DSRL_RESIZE_IMAGE"
fi

FRANKA_EXTRACT_DIM="$FRANKA_EXTRACT_DIM" \
FRANKA_NO_STATE_INPUT="$FRANKA_NO_STATE_INPUT" \
python scripts/serve_policy.py \
    --default_prompt="$LANGUAGE_INSTRUCTION" \
    --port="$POLICY_PORT" \
    $DSRL_ARGS \
    policy:checkpoint \
    --policy.config="$POLICY_CONFIG" \
    --policy.dir="$SFT_CKPT_DIR" \
    "$@"
