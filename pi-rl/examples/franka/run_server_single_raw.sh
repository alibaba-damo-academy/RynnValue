#!/bin/bash
# Launch the EXPO-FT env server for Franka SINGLE-arm robot (raw gripper mode).
# Run this on the robot control machine (with ROS2 installed).
#
# "raw" gripper mode: policy output is directly multiplied by GRIPPER_ACTION_MAX
# and sent to the robot, without binary thresholding.
#
# Usage:
#   bash scripts/franka/run_server_single_raw.sh
#   bash scripts/franka/run_server_single_raw.sh --port 8200
#   bash scripts/franka/run_server_single_raw.sh --instruction "pick up the cup"
#   bash scripts/franka/run_server_single_raw.sh --port 8200 --instruction "flip the steak"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
PORT="8101"
INSTRUCTION=""

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
    echo "[run_server_single_raw] instruction=\"$INSTRUCTION\""
fi

python -m client.run_client \
    --config-task-path configs/task/franka_single.py \
    --server-port "$PORT"
