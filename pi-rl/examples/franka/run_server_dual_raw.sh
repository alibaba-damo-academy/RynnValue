#!/bin/bash
# Launch the EXPO-FT env server for Franka DUAL-arm robot (raw gripper mode).
# Run this on the robot control machine (with ROS2 installed).
#
# "raw" gripper mode: policy output is clipped to [0, GRIPPER_ACTION_MAX]
# and sent to the robot, without binary thresholding.
#
# Usage:
#   bash scripts/franka/run_server_dual_raw.sh
#   bash scripts/franka/run_server_dual_raw.sh --port 8202
#   bash scripts/franka/run_server_dual_raw.sh --instruction "pick up the cup"
#   bash scripts/franka/run_server_dual_raw.sh --port 8202 --instruction "flip the steak"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
PORT="8101"
INSTRUCTION="Move the box from the right side to the left side."

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"; shift 2 ;;
        --port=*)
            PORT="${1#*=}"; shift ;;
        --instruction|--inst)
            INSTRUCTION="$2"; shift 2 ;;
        --instruction=*|--inst=*)
            INSTRUCTION="${1#*=}"; shift ;;
        *)
            # Treat bare number as port.
            PORT="$1"; shift ;;
    esac
done

cd "$PROJECT_DIR"

# Build extra env vars for language instruction override
if [[ -n "$INSTRUCTION" ]]; then
    export FRANKA_LANGUAGE_INSTRUCTION="$INSTRUCTION"
    echo "[run_server_dual_raw] instruction=\"$INSTRUCTION\""
fi

python -m client.run_client \
    --config-task-path configs/task/franka.py \
    --server-port "$PORT"
