#!/usr/bin/env bash
# Profile the Franka -> LeRobot conversion on a small subset of episodes.
#
# Usage:
#   bash scripts/profile_convert_data.sh
#
# Override from shell:
#   DATA_DIR=/path/to/data MAX_EPISODES=5 bash scripts/profile_convert_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Configuration ──────────────────────────────────────────────────────────
FRANKA_HOME="${FRANKA_HOME:-/path/to/franka_lerobot_data}"
DATA_DIR="${DATA_DIR:-/path/to/raw_franka_data/pick_up_the_banana}"
REPO_NAME="${REPO_NAME:-__profile_test__}"
LANGUAGE_INSTRUCTION="${LANGUAGE_INSTRUCTION:-Move the basket from the right side to the left, then place the banana held by the left gripper into the basket.}"
MAX_EPISODES="${MAX_EPISODES:-10}"

FPS="${FPS:-30}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-224}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-224}"

NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_EPISODES="${PREFETCH_EPISODES:-8}"
EPISODE_IMAGE_THREADS="${EPISODE_IMAGE_THREADS:-8}"
# ───────────────────────────────────────────────────────────────────────────

CONVERT_SCRIPT="$PROJECT_ROOT/scripts/convert_franka_data_to_lerobot.py"

if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "[ERROR] Cannot find conversion script: $CONVERT_SCRIPT" >&2
    exit 1
fi

mkdir -p "$FRANKA_HOME"
export HF_LEROBOT_HOME="$FRANKA_HOME"

echo "============================================================"
echo " Franka -> LeRobot PROFILE run"
echo "============================================================"
echo "DATA_DIR                = $DATA_DIR"
echo "REPO_NAME               = $REPO_NAME"
echo "MAX_EPISODES            = $MAX_EPISODES"
echo "IMAGE_SIZE              = (${IMAGE_SIZE_H}, ${IMAGE_SIZE_W})"
echo "NUM_WORKERS             = $NUM_WORKERS"
echo "PREFETCH_EPISODES       = $PREFETCH_EPISODES"
echo "EPISODE_IMAGE_THREADS   = $EPISODE_IMAGE_THREADS"
echo "============================================================"
echo ""

START_TIME=$SECONDS

python "$CONVERT_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --repo_name "$REPO_NAME" \
    --language_instruction "$LANGUAGE_INSTRUCTION" \
    --fps "$FPS" \
    --image_size "$IMAGE_SIZE_H" "$IMAGE_SIZE_W" \
    --max_episodes "$MAX_EPISODES" \
    --num_workers "$NUM_WORKERS" \
    --prefetch_episodes "$PREFETCH_EPISODES" \
    --episode_image_threads "$EPISODE_IMAGE_THREADS" \
    --profile \
    --no-push_to_hub \
    # --no-generate_sample_video \
    "$@"

ELAPSED=$(( SECONDS - START_TIME ))
echo ""
echo "============================================================"
echo " Profile complete: ${ELAPSED}s wall time for $MAX_EPISODES episodes"
echo "============================================================"
