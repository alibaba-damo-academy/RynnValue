#!/usr/bin/env bash
# Compute pi0.5 normalisation statistics for the Franka real-robot dataset.
#
# Run AFTER convert_data.sh has finished.
#
# Usage:
#   cd /path/to/pi-rl
#   bash scripts/franka/calculate_norm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ── Configuration (edit these) ──────────────────────────────────────────────
REPO_ID="${REPO_ID:-expo_ft/pick_up_the_bread}"
# ────────────────────────────────────────────────────────────────────────────

source .venv/bin/activate

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

uv run expo_ft/agents/vla/openpi/scripts/compute_norm_stats.py \
    --config-name expo_pi05_franka_lora_sft \
    --repo-id "$REPO_ID"

echo ""
echo "✓ Norm stats computed for repo: $REPO_ID"
echo "  Saved under: /path/to/assets/expo_pi05_franka_lora_sft/$REPO_ID/norm_stats.json"
echo ""
echo "Next step — SFT fine-tuning:"
echo "  bash scripts/franka/finetune.sh"
