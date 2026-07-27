#!/usr/bin/env bash
# Convert Franka dual-arm demonstration data to LeRobot format.
#
# Usage:
#   cd /path/to/pi-rl
#   bash scripts/franka/convert_data.sh
#
# After conversion, run calculate_norm.sh to compute normalisation statistics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ── Configuration (edit these) ─────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/path/to/raw_franka_data/pick_up_the_bread}"
REPO_NAME="${REPO_NAME:-expo_ft/pick_up_the_bread}"
LANGUAGE_INSTRUCTION="${LANGUAGE_INSTRUCTION:-Pick up the bread from the table and put it in the basket.}"
MAX_EPISODES="${MAX_EPISODES:-}"   # leave empty to convert all episodes
# ───────────────────────────────────────────────────────────────────────────

MAX_EPISODES_ARG=""
if [ -n "$MAX_EPISODES" ]; then
    MAX_EPISODES_ARG="--max_episodes $MAX_EPISODES"
fi

.venv/bin/python scripts/convert_franka_data_to_lerobot.py \
    --data_dir "$DATA_DIR" \
    --repo_name "$REPO_NAME" \
    --language_instruction "$LANGUAGE_INSTRUCTION" \
    --no-push_to_hub \
    $MAX_EPISODES_ARG \
    "$@"

echo ""
echo "✓ Conversion complete. Dataset saved under \$LEROBOT_HOME/$REPO_NAME"
echo ""
echo "Next step — compute normalisation stats:"
echo "  bash scripts/franka/calculate_norm.sh"
