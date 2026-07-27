#!/usr/bin/env bash
# Repair demonstration data.
#
# Features:
# - Clips gripper values to 252
# - Discards episodes with action jumps
#
# Usage:
#   cd /path/to/pi-rl
#   bash scripts/repair_data.sh
#
# Override config:
#   INPUT_DIR=/path/to/input \
#   OUTPUT_DIR=/path/to/output \
#   bash scripts/repair_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR="${INPUT_DIR:-/path/to/raw_franka_data}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/raw_franka_data_cleaned}"

REPAIR_SCRIPT="$SCRIPT_DIR/repair_dataset_parallel.py"

if [ ! -f "$REPAIR_SCRIPT" ]; then
    echo "[ERROR] Cannot find repair script: $REPAIR_SCRIPT" >&2
    exit 1
fi

echo "============================================================"
echo " Data Repair"
echo "============================================================"
echo "INPUT_DIR   = $INPUT_DIR"
echo "OUTPUT_DIR  = $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "Repair operations:"
echo "  1. Clip gripper values to 252"
echo "  2. Discard episodes with action jumps"
echo ""

python "$REPAIR_SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "✓ Data repair complete."
echo "✓ Cleaned dataset saved to: $OUTPUT_DIR"
