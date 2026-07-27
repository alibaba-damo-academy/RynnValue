#!/bin/bash
# Serve pi0.5 DROID policy for inference.
#
# Usage:
#   bash scripts/serve_pi05_droid.sh
#   bash scripts/serve_pi05_droid.sh /path/to/custom/checkpoint
#
# Env (override as needed):
#   PORT        - WebSocket server port (default: 8000).
#   PROMPT      - Default language prompt for the policy.

set -euo pipefail

# ----- workspace (cd to repo root so relative paths resolve) -----
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "\033[32m[Info] Working directory: $PROJECT_ROOT\033[0m"

# ----- config -----
CONFIG="pi05_droid"
CHECKPOINT_DIR="${1:-gs://openpi-assets/checkpoints/pi05_droid}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-}"

# ----- build command -----
CMD="uv run scripts/serve_policy.py policy:checkpoint --policy.config=${CONFIG} --policy.dir=${CHECKPOINT_DIR} --port=${PORT}"

if [ -n "$PROMPT" ]; then
    CMD="$CMD --default-prompt=\"${PROMPT}\""
fi

echo -e "\033[32m[Info] Config: ${CONFIG}\033[0m"
echo -e "\033[32m[Info] Checkpoint: ${CHECKPOINT_DIR}\033[0m"
echo -e "\033[32m[Info] Port: ${PORT}\033[0m"
if [ -n "$PROMPT" ]; then
    echo -e "\033[32m[Info] Default prompt: ${PROMPT}\033[0m"
fi

echo -e "\033[32m[Info] Starting policy server...\033[0m"
eval $CMD
