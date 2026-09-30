#!/bin/bash
# pi0.5 filter BC on mixed single-arm + dual-arm Franka optimized-v2 repos.
#
# Frames whose W-window advantage A_t does not exceed the per-repo threshold
# (A_t > 0 by default, criterion "zero") are dropped; the kept frames train as
# plain BC (task prompt only). Filter whitelists + advantage thresholds are
# computed offline during the norm-stats pass and cached under assets/${CONFIG}.
#
# Every repo must carry a per-frame ``progress`` feature (the RynnValue-labeled
# dumps); the filter reads it straight from the lerobot episode parquet files.
#
# Usage:
#   bash scripts/train_pi05_franka_mixed_filter_bc.sh
#   REPO_IDS="/path/to/close_the_drawer,/path/to/pick_up_the_box" \
#       bash scripts/train_pi05_franka_mixed_filter_bc.sh
#   ADVANTAGE_HORIZON=128 ADVANTAGE_LAMBDA=0.98 \
#       bash scripts/train_pi05_franka_mixed_filter_bc.sh
#
# Env (override as needed):
#   CONFIG            - registered TrainConfig (default: pi05_franka_mixed_filter_bc).
#   EXP_NAME          - experiment / checkpoint sub-folder name.
#   GPUS              - CUDA devices (default: 0,1).
#   FSDP_DEVICES      - if >1, shard the model across this many devices.
#   CKPT_BASE_DIR     - checkpoint root (default: the TrainConfig's own).
#   REPO_IDS          - comma-separated repo list overriding the config's ``repo_ids``.
#   ADVANTAGE_CRITERION / ADVANTAGE_HORIZON / ADVANTAGE_QUANTILE / ADVANTAGE_LAMBDA
#                     - filter overrides; empty means "keep the config default". They are
#                       passed to BOTH the norm-stats pass and train.py so the cached
#                       whitelist matches the run that consumes it.
#   SKIP_NORM         - if set, skip the norm-stats / whitelist compute.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo -e "\033[32m[Info] Working directory: $PROJECT_ROOT\033[0m"

CONFIG="${CONFIG:-pi05_franka_mixed_filter_bc}"
EXP_NAME="${EXP_NAME:-${CONFIG#pi05_}}"
GPUS="${GPUS:-0,1}"
FSDP_DEVICES="${FSDP_DEVICES:-2}"
CKPT_BASE_DIR="${CKPT_BASE_DIR:-}"
REPO_IDS="${REPO_IDS:-}"
ADVANTAGE_CRITERION="${ADVANTAGE_CRITERION:-}"
ADVANTAGE_HORIZON="${ADVANTAGE_HORIZON:-}"
ADVANTAGE_QUANTILE="${ADVANTAGE_QUANTILE:-}"
ADVANTAGE_LAMBDA="${ADVANTAGE_LAMBDA:-}"

NORM_STATS_FILE="${NORM_STATS_FILE:-assets/${CONFIG}/${CONFIG#pi05_}/norm_stats.json}"

echo "==> CONFIG        = ${CONFIG}"
echo "==> EXP_NAME      = ${EXP_NAME}"
echo "==> GPUS          = ${GPUS}"
echo "==> FSDP_DEVICES  = ${FSDP_DEVICES}"
echo "==> NORM_STATS    = ${NORM_STATS_FILE}"
echo "==> ADV_CRITERION = ${ADVANTAGE_CRITERION:-<config default>}"
echo "==> ADV_WINDOW    = ${ADVANTAGE_HORIZON:-<config default>}"
echo "==> ADV_QUANTILE  = ${ADVANTAGE_QUANTILE:-<config default>}"
echo "==> ADV_LAMBDA    = ${ADVANTAGE_LAMBDA:-<config default>}"
if [[ -n "${CKPT_BASE_DIR}" ]]; then
    echo "==> CKPT_BASE     = ${CKPT_BASE_DIR}"
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
export OPENPI_DATA_HOME=/path/to/openpi_data
# HF parquet generation forks num_proc workers; forking from the per-repo loader
# threads after JAX init can deadlock (os.fork + multithreaded JAX). Repos are built
# concurrently by a ThreadPoolExecutor, so single-proc per repo is fast enough.
export LEROBOT_LOAD_NUM_PROC="${LEROBOT_LOAD_NUM_PROC:-1}"

repo_args=()
if [[ -n "${REPO_IDS}" ]]; then
    # Space-separated values for the tyro tuple field --data.repo_ids (kept last on the
    # command line because variadic tuples swallow trailing values).
    read -r -a _repos <<< "${REPO_IDS//,/ }"
    repo_args+=(--data.repo_ids "${_repos[@]}")
fi

# Filter overrides shared by the norm-stats pass and train.py: all four feed the
# advantage-threshold / filter-index cache-key suffix, so a mismatch would silently
# leave the training run to recompute the whitelist from scratch.
adv_args=()
norm_adv_args=()
if [[ -n "${ADVANTAGE_CRITERION}" ]]; then
    adv_args+=(--data.advantage_criterion="${ADVANTAGE_CRITERION}")
    norm_adv_args+=(--advantage-criterion "${ADVANTAGE_CRITERION}")
fi
if [[ -n "${ADVANTAGE_HORIZON}" ]]; then
    adv_args+=(--data.advantage_horizon="${ADVANTAGE_HORIZON}")
    norm_adv_args+=(--advantage-horizon "${ADVANTAGE_HORIZON}")
fi
if [[ -n "${ADVANTAGE_QUANTILE}" ]]; then
    adv_args+=(--data.advantage_quantile="${ADVANTAGE_QUANTILE}")
    norm_adv_args+=(--advantage-quantile "${ADVANTAGE_QUANTILE}")
fi
if [[ -n "${ADVANTAGE_LAMBDA}" ]]; then
    adv_args+=(--data.advantage_lambda="${ADVANTAGE_LAMBDA}")
    norm_adv_args+=(--advantage-lambda "${ADVANTAGE_LAMBDA}")
fi

if [[ -n "${SKIP_NORM:-}" || -f "${NORM_STATS_FILE}" ]]; then
    echo "==> norm stats present at ${NORM_STATS_FILE} (or SKIP_NORM set), skipping compute"
else
    if [[ -n "${REPO_IDS}" ]]; then
        echo "[error] norm stats missing and REPO_IDS override is set: compute_norm_stats_fast.py" >&2
        echo "        does not accept --data.repo_ids. Fill repo_ids in config.py" >&2
        echo "        (LeRobotFrankaMixedFilterBCOptimizedV2DataConfig) and rerun." >&2
        exit 1
    fi
    echo "==> computing norm stats + advantage thresholds + filter indices -> assets/${CONFIG}"
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python scripts/compute_norm_stats_fast.py \
            --config-name "${CONFIG}" \
            "${norm_adv_args[@]}"
fi

extra_args=()
if [[ -n "${CKPT_BASE_DIR}" ]]; then
    extra_args+=(--checkpoint_base_dir="${CKPT_BASE_DIR}")
fi

python scripts/train.py \
    "${CONFIG}" \
    --exp_name="${EXP_NAME}" \
    --fsdp_devices="${FSDP_DEVICES}" \
    --overwrite \
    "${extra_args[@]}" \
    "$@" \
    "${adv_args[@]}" \
    "${repo_args[@]}"
