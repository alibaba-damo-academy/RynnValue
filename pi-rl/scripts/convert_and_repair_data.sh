#!/usr/bin/env bash
# Convert Franka demonstration data to LeRobot format with data-quality repair.
#
# Compared to convert_data.sh this script additionally:
#   - clips gripper channels (action + observation) to a ceiling (default 252)
#   - drops episodes whose total per-frame action jumps exceed a budget
#
# Repairs are applied in-memory during conversion; no cleaned copy is written.
#
# Usage:
#   bash scripts/convert_and_repair_data.sh
#
# Override config from shell:
#   DATA_DIR=/path/to/data \
#   REPO_NAME=my_repo \
#   LANGUAGE_INSTRUCTION="pick up the bread" \
#   CLIP_GRIPPER_MAX=252 \
#   JUMP_THRESHOLD=0.15 \
#   MAX_JUMPS_PER_EPISODE=0 \
#   bash scripts/convert_and_repair_data.sh
#
# Disable a repair step entirely:
#   CLIP_GRIPPER_MAX=None bash scripts/convert_and_repair_data.sh      # no gripper clip
#   JUMP_THRESHOLD=None    bash scripts/convert_and_repair_data.sh     # no jump filter
#
# Resume an interrupted run:
#   RESUME=true bash scripts/convert_and_repair_data.sh

set -euo pipefail

# Resolve project root from this script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Configuration (override via env vars) ───────────────────────────────────
FRANKA_HOME="${FRANKA_HOME:-/path/to/franka_lerobot_data}"
DATA_DIR="${DATA_DIR:-/path/to/raw_franka_data/close_the_drawer_new}"
REPO_NAME="${REPO_NAME:-close_the_drawer}"
LANGUAGE_INSTRUCTION="${LANGUAGE_INSTRUCTION:-Put the box on the table into the drawer, then close the drawer.}"
MAX_EPISODES="${MAX_EPISODES:-}"   # empty = all
FILTER_PATH="${FILTER_PATH:-${DATA_DIR}/filter.json}"

FPS="${FPS:-30}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-224}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-224}"

NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_EPISODES="${PREFETCH_EPISODES:-8}"
EPISODE_IMAGE_THREADS="${EPISODE_IMAGE_THREADS:-8}"

PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
PROFILE="${PROFILE:-false}"
RESUME="${RESUME:-false}"

GENERATE_SAMPLE_VIDEO="${GENERATE_SAMPLE_VIDEO:-true}"
SAMPLE_VIDEO_NAME="${SAMPLE_VIDEO_NAME:-sample_episode_preview}"

# ── Repair options ──────────────────────────────────────────────────────────
#   CLIP_GRIPPER_MAX          : ceiling applied to action.gripper / obs.state.gripper
#                               (use "None" to disable)
#   JUMP_THRESHOLD            : per-frame |action diff| above this counts as a jump
#                               (use "None" to disable)
#   MAX_JUMPS_PER_EPISODE     : drop any episode with more jumps than this
CLIP_GRIPPER_MAX="${CLIP_GRIPPER_MAX:-252}"
JUMP_THRESHOLD="${JUMP_THRESHOLD:-0.15}"
MAX_JUMPS_PER_EPISODE="${MAX_JUMPS_PER_EPISODE:-0}"
# ────────────────────────────────────────────────────────────────────────────

CONVERT_SCRIPT="$PROJECT_ROOT/scripts/convert_and_repair_franka_data_to_lerobot.py"

if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "[ERROR] Cannot find conversion script: $CONVERT_SCRIPT" >&2
    exit 1
fi

mkdir -p "$FRANKA_HOME"
export HF_LEROBOT_HOME="$FRANKA_HOME"

bool_flag() {
    local name="$1"
    local value="$2"
    if [ "$value" = "true" ]; then
        echo "--$name"
    elif [ "$value" = "false" ]; then
        echo "--no-$name"
    else
        echo "[ERROR] $name must be 'true' or 'false', got: $value" >&2
        exit 1
    fi
}

# Helper: build `--flag value` args; pass the literal string "None" to skip.
optional_arg() {
    local name="$1"
    local value="$2"
    if [ "$value" = "None" ] || [ "$value" = "none" ] || [ "$value" = "" ]; then
        echo "--$name" "None"
    else
        echo "--$name" "$value"
    fi
}

MAX_EPISODES_ARGS=()
if [ -n "$MAX_EPISODES" ]; then
    MAX_EPISODES_ARGS+=(--max_episodes "$MAX_EPISODES")
fi

read -r -a CLIP_GRIPPER_ARGS <<< "$(optional_arg clip_gripper_max "$CLIP_GRIPPER_MAX")"
read -r -a JUMP_THRESHOLD_ARGS <<< "$(optional_arg jump_threshold "$JUMP_THRESHOLD")"

PUSH_TO_HUB_FLAG="$(bool_flag push_to_hub "$PUSH_TO_HUB")"
PROFILE_FLAG="$(bool_flag profile "$PROFILE")"
GENERATE_SAMPLE_VIDEO_FLAG="$(bool_flag generate_sample_video "$GENERATE_SAMPLE_VIDEO")"
RESUME_FLAG="$(bool_flag resume "$RESUME")"

echo "============================================================"
echo " Franka -> LeRobot conversion (with repair)"
echo "============================================================"
echo "PROJECT_ROOT            = $PROJECT_ROOT"
echo "CONVERT_SCRIPT          = $CONVERT_SCRIPT"
echo "DATA_DIR                = $DATA_DIR"
echo "REPO_NAME               = $REPO_NAME"
echo "HF_LEROBOT_HOME         = $HF_LEROBOT_HOME"
echo "LANGUAGE_INSTRUCTION    = $LANGUAGE_INSTRUCTION"
echo "FILTER_PATH             = $FILTER_PATH"
echo "MAX_EPISODES            = ${MAX_EPISODES:-<all>}"
echo "FPS                     = $FPS"
echo "IMAGE_SIZE              = (${IMAGE_SIZE_H}, ${IMAGE_SIZE_W})"
echo "NUM_WORKERS             = $NUM_WORKERS"
echo "PREFETCH_EPISODES       = $PREFETCH_EPISODES"
echo "EPISODE_IMAGE_THREADS   = $EPISODE_IMAGE_THREADS"
echo "PUSH_TO_HUB             = $PUSH_TO_HUB"
echo "PROFILE                 = $PROFILE"
echo "RESUME                  = $RESUME"
echo "GENERATE_SAMPLE_VIDEO   = $GENERATE_SAMPLE_VIDEO"
echo "SAMPLE_VIDEO_NAME       = $SAMPLE_VIDEO_NAME"
echo "------------------------------------------------------------"
echo "CLIP_GRIPPER_MAX        = $CLIP_GRIPPER_MAX"
echo "JUMP_THRESHOLD          = $JUMP_THRESHOLD"
echo "MAX_JUMPS_PER_EPISODE   = $MAX_JUMPS_PER_EPISODE"
echo "============================================================"
echo ""

python "$CONVERT_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --repo_name "$REPO_NAME" \
    --language_instruction "$LANGUAGE_INSTRUCTION" \
    --fps "$FPS" \
    --image_size "$IMAGE_SIZE_H" "$IMAGE_SIZE_W" \
    --num_workers "$NUM_WORKERS" \
    --prefetch_episodes "$PREFETCH_EPISODES" \
    --episode_image_threads "$EPISODE_IMAGE_THREADS" \
    --episode_filter_path "$FILTER_PATH" \
    --sample_video_name "$SAMPLE_VIDEO_NAME" \
    --max_jumps_per_episode "$MAX_JUMPS_PER_EPISODE" \
    "${CLIP_GRIPPER_ARGS[@]}" \
    "${JUMP_THRESHOLD_ARGS[@]}" \
    "$PUSH_TO_HUB_FLAG" \
    "$PROFILE_FLAG" \
    "$GENERATE_SAMPLE_VIDEO_FLAG" \
    "$RESUME_FLAG" \
    "${MAX_EPISODES_ARGS[@]}" \
    "$@"

echo ""
echo "✓ Conversion (with repair) complete."
echo "✓ Dataset saved under: $HF_LEROBOT_HOME/$REPO_NAME"
echo "✓ Episode filter:      $FILTER_PATH"

if [ "$GENERATE_SAMPLE_VIDEO" = "true" ]; then
    echo "✓ Sample videos:       $SCRIPT_DIR/$SAMPLE_VIDEO_NAME/"
fi

echo ""
echo "Recommended next steps:"
echo "  1. Inspect the sample videos (check for episodes that should have been dropped)"
echo "  2. Edit $FILTER_PATH to exclude any remaining bad episodes"
echo "  3. Run calculate_norm.sh"
