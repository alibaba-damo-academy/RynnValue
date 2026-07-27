#!/bin/bash
# pi0.5 SFT on every RoboTwin sub-task at once (multi-task fine-tuning).
#
# Usage:
#   bash scripts/train_pi05_robotwin_all.sh                       # default exp_name = "all_50"
#   bash scripts/train_pi05_robotwin_all.sh <exp_name> [extra train.py args...]
#
# Known CONFIG / ROBOTWIN_ROOT pairs:
#   CONFIG=pi05_robotwin_all          ROBOTWIN_ROOT=.../lerobot_robotwin_eef_clean_50  (default)
#   CONFIG=pi05_robotwin_all_lora     ROBOTWIN_ROOT=.../lerobot_robotwin_eef_clean_50  (LoRA)
#   CONFIG=pi05_robotwin_aug_500      ROBOTWIN_ROOT=.../lerobot_robotwin_eef_aug_500   (10x data)
#   CONFIG=pi05_robotwin_aug_500_lora ROBOTWIN_ROOT=.../lerobot_robotwin_eef_aug_500   (10x + LoRA)
#
# Env (override as needed):
#   CONDA_ENV    - conda env to activate (default: pi-rl).
#   CONDA_ROOT   - conda installation root (default: /opt/conda).
#   CONFIG       - registered TrainConfig (default: pi05_robotwin_all).
#   ROBOTWIN_ROOT- root containing per-task lerobot subdirs. If unset, defaults are inferred
#                  from CONFIG (clean_50 vs aug_500).
#   ASSET_ID     - sub-folder used for norm_stats. If unset, inferred from CONFIG.
#   GPUS         - CUDA devices (default: 0,1).
#   FSDP_DEVICES - if >1, shard the model across this many devices.
#   SKIP_NORM    - if set, skip the (slow) shared norm-stats compute.

# `set -u` would trip on conda's internal bookkeeping (PS1 etc.), so defer it until after activate.
set -eo pipefail

# ----- workspace (cd to repo root so the relative paths below resolve) -----
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "\033[32m[Info] Working directory: $PROJECT_ROOT\033[0m"

# ----- conda activation -----
CONDA_ROOT="${CONDA_ROOT:-/opt/conda}"
CONDA_ENV="${CONDA_ENV:-pi-rl}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
else
    echo "[error] conda.sh not found at ${CONDA_ROOT}/etc/profile.d/conda.sh; set CONDA_ROOT" >&2
    exit 1
fi
conda activate "${CONDA_ENV}"
echo "==> CONDA_ENV     = ${CONDA_ENV} ($(which python))"

set -u

CONFIG="${CONFIG:-pi05_robotwin_all}"
EXP_NAME="${1:-all_50}"
GPUS="${GPUS:-0,1}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"

# Default ROBOTWIN_ROOT / ASSET_ID inferred from CONFIG. Override either env var to point
# elsewhere (e.g. a subset of tasks) without editing the script.
case "${CONFIG}" in
    *aug_500*)
        DEFAULT_ROOT="/path/to/robotwin_lerobot_data/lerobot_robotwin_eef_aug_500"
        DEFAULT_ASSET_ID="robotwin_aug_500"
        ;;
    *)
        DEFAULT_ROOT="/path/to/robotwin_lerobot_data/lerobot_robotwin_eef_clean_50"
        DEFAULT_ASSET_ID="robotwin_all"
        ;;
esac
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${DEFAULT_ROOT}}"
ASSET_ID="${ASSET_ID:-${DEFAULT_ASSET_ID}}"

if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
    echo "[error] dataset root not found: ${ROBOTWIN_ROOT}" >&2
    exit 1
fi

echo "==> CONFIG        = ${CONFIG}"
echo "==> ROBOTWIN_ROOT = ${ROBOTWIN_ROOT}"
echo "==> ASSET_ID      = ${ASSET_ID}"
echo "==> EXP_NAME      = ${EXP_NAME}"
echo "==> GPUS          = ${GPUS}"
echo "==> FSDP_DEVICES  = ${FSDP_DEVICES}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export OPENPI_DATA_HOME=./openpi

# Multi-task norm stats are aggregated across every sub-task and saved to
# ./assets/<CONFIG>/<ASSET_ID>/norm_stats.json.
NORM_STATS_FILE="./assets/${CONFIG}/${ASSET_ID}/norm_stats.json"
if [[ -n "${SKIP_NORM:-}" || -f "${NORM_STATS_FILE}" ]]; then
    echo "==> norm stats present at ${NORM_STATS_FILE} (or SKIP_NORM set), skipping compute"
else
    echo "==> computing norm stats over all sub-tasks -> ${NORM_STATS_FILE}"
    # Force JAX to CPU + disable XLA preallocation: norm_stats is pure-numpy CPU work,
    # but importing openpi pulls in JAX which would otherwise grab 75% of every visible GPU
    # (per worker process under spawn-mode DataLoader). These two env vars only affect this
    # subshell, so the train step below still gets full GPU access.
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python scripts/compute_norm_stats_fast.py --config-name "${CONFIG}"
fi

# Drop the first positional arg (exp_name) and pass the rest through to train.py.
shift $(( $# >= 1 ? 1 : 0 ))

python scripts/train.py \
    "${CONFIG}" \
    --exp_name="${EXP_NAME}" \
    --fsdp_devices="${FSDP_DEVICES}" \
    --resume \
    "$@"
