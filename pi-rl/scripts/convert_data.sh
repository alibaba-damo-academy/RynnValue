#!/usr/bin/env bash
# Convert Franka demonstration data to LeRobot format (full conversion).
#
# Features:
# - Converts all episodes, including *_f failed episodes
# - Generates a RynnValue-compatible filter.json containing only non-_f episodes
# - Generates a sample preview video for the first successful episode
#
# Usage:
#   cd /path/to/pi-rl
#   bash scripts/convert_data.sh
#
# Override config from shell:
#   DATA_DIR=/path/to/data \
#   REPO_NAME=my_repo \
#   LANGUAGE_INSTRUCTION="pick up the bread" \
#   bash scripts/convert_data.sh
#
# Resume an interrupted run (skip already-converted episodes):
#   RESUME=true bash scripts/convert_data.sh
#
# After conversion:
#   1. inspect the sample video
#   2. edit filter.json if needed
#   3. run calculate_norm.sh

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
# ────────────────────────────────────────────────────────────────────────────

CONVERT_SCRIPT="$PROJECT_ROOT/scripts/convert_franka_data_to_lerobot.py"

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

MAX_EPISODES_ARGS=()
if [ -n "$MAX_EPISODES" ]; then
    MAX_EPISODES_ARGS+=(--max_episodes "$MAX_EPISODES")
fi

PUSH_TO_HUB_FLAG="$(bool_flag push_to_hub "$PUSH_TO_HUB")"
PROFILE_FLAG="$(bool_flag profile "$PROFILE")"
GENERATE_SAMPLE_VIDEO_FLAG="$(bool_flag generate_sample_video "$GENERATE_SAMPLE_VIDEO")"
RESUME_FLAG="$(bool_flag resume "$RESUME")"

echo "============================================================"
echo " Franka -> LeRobot conversion"
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
    "$PUSH_TO_HUB_FLAG" \
    "$PROFILE_FLAG" \
    "$GENERATE_SAMPLE_VIDEO_FLAG" \
    "$RESUME_FLAG" \
    "${MAX_EPISODES_ARGS[@]}" \
    "$@"

echo ""
echo "✓ Conversion complete."
echo "✓ Dataset saved under: $HF_LEROBOT_HOME/$REPO_NAME"
echo "✓ Episode filter:      $FILTER_PATH"

if [ "$GENERATE_SAMPLE_VIDEO" = "true" ]; then
    echo "✓ Sample videos:       $SCRIPT_DIR/$SAMPLE_VIDEO_NAME/"
fi

echo ""
echo "Recommended next steps:"
echo "  1. Inspect the sample videos"
echo "  2. Edit $FILTER_PATH to exclude bad episodes if needed"
echo "  3. Run calculate_norm.sh"
