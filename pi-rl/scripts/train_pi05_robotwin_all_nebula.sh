#!/bin/bash
# Nebula-side entry for pi0.5 RoboTwin (all sub-tasks) SFT.
#
# Called by run_nebula.py inside the worker container. Nebula's algo_name
# already provisions the python env, so we skip conda activation entirely and
# just call `python` directly.
#
# Override examples:
#   CONFIG=pi05_robotwin_aug_500 EXP_NAME=aug500_nebula bash scripts/train_pi05_robotwin_all_nebula.sh
#   GPUS=0,1,2,3 FSDP_DEVICES=4 bash scripts/train_pi05_robotwin_all_nebula.sh

set -eo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "\033[32m[Info] Working directory: $PROJECT_ROOT\033[0m"

# HuggingFace / tokenizer hygiene — mirror EasyVLA's nebula entry.
export HDF5_USE_FILE_LOCKING=FALSE
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CONFIG="${CONFIG:-pi05_robotwin_all}"
EXP_NAME="${EXP_NAME:-pi05_robotwin_all_50_nebula}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"

# ROBOTWIN_ROOT / ASSET_ID default by CONFIG; override via env to use a subset.
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
echo "==> EXP_NAME      = ${EXP_NAME}"
echo "==> ROBOTWIN_ROOT = ${ROBOTWIN_ROOT}"
echo "==> ASSET_ID      = ${ASSET_ID}"
echo "==> GPUS          = ${GPUS}"
echo "==> FSDP_DEVICES  = ${FSDP_DEVICES}"
echo "==> python        = $(which python)"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export OPENPI_DATA_HOME=./openpi

# Aggregated norm stats over all sub-tasks live at this path; recompute only if missing.
NORM_STATS_FILE="./assets/${CONFIG}/${ASSET_ID}/norm_stats.json"
if [[ -n "${SKIP_NORM:-}" || -f "${NORM_STATS_FILE}" ]]; then
    echo "==> norm stats present at ${NORM_STATS_FILE} (or SKIP_NORM set), skipping compute"
else
    echo "==> computing norm stats -> ${NORM_STATS_FILE}"
    # Pin JAX to CPU so the norm_stats import doesn't grab all GPU memory.
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python scripts/compute_norm_stats_fast.py --config-name "${CONFIG}"
fi

exec python scripts/train.py \
    "${CONFIG}" \
    --exp_name="${EXP_NAME}" \
    --fsdp_devices="${FSDP_DEVICES}" \
    --overwrite \
    "$@"
