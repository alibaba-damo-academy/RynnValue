#!/bin/bash
# Launch the EXPO-FT env server for Franka dual-arm robot.
# Run this on the robot control machine (with ROS2 installed).
#
# Usage (env-var style):
#   bash scripts/franka/run_server.sh
#   LANGUAGE_INSTRUCTION="pick up the bread" bash scripts/franka/run_server.sh
#
# Usage (flag style):
#   bash scripts/franka/run_server.sh --lang "pick up the bread" --port 8102
#
# All flags (each also overridable via the matching env-var):
#   --lang              Language instruction for the robot  [LANGUAGE_INSTRUCTION, default: "complete the task"]
#   --port              Env server port                     [SERVER_PORT, default: 8102]
#   --task-config       Task config file path               [TASK_CONFIG, default: configs/task/franka.py]
#   --reset-joints      14 joint angles (rad), comma-sep    [RESET_JOINTS, default: all zeros]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Defaults ────────────────────────────────────────────────────────────────
SERVER_PORT="${SERVER_PORT:-8102}"
LANGUAGE_INSTRUCTION="${LANGUAGE_INSTRUCTION:-complete the task}"
TASK_CONFIG="${TASK_CONFIG:-configs/task/franka.py}"
RESET_JOINTS="${RESET_JOINTS:-}"   # 14 comma-separated joint angles in radians

# ── Parse CLI flags ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)         SERVER_PORT="$2";           shift 2 ;;
        --port=*)       SERVER_PORT="${1#*=}";      shift   ;;
        --lang)         LANGUAGE_INSTRUCTION="$2";  shift 2 ;;
        --lang=*)       LANGUAGE_INSTRUCTION="${1#*=}"; shift ;;
        --task-config)  TASK_CONFIG="$2";           shift 2 ;;
        --task-config=*)TASK_CONFIG="${1#*=}";      shift   ;;
        --reset-joints) RESET_JOINTS="$2";          shift 2 ;;
        --reset-joints=*)RESET_JOINTS="${1#*=}";    shift   ;;
        *)
            # Legacy: bare number treated as port
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                SERVER_PORT="$1"; shift
            else
                echo "Unknown argument: $1" >&2; exit 1
            fi ;;
    esac
done
# ────────────────────────────────────────────────────────────────────────────

echo "Task config: $TASK_CONFIG"
echo "Language:    $LANGUAGE_INSTRUCTION"
echo "Server port: $SERVER_PORT"
echo ""

cd "$PROJECT_DIR"

LANGUAGE_INSTRUCTION="$LANGUAGE_INSTRUCTION" \
RESET_JOINTS="$RESET_JOINTS" \
python -m client.run_client \
    --config-task-path "$TASK_CONFIG" \
    --server-port "$SERVER_PORT"
