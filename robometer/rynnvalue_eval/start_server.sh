#!/usr/bin/env bash
# Launch the rynnvalue baseline eval server.
#
# Usage:
#   bash rynnvalue_eval/start_server.sh                                      # default port 8001, hf model
#   bash rynnvalue_eval/start_server.sh --model-path /path/to/hf_model_dir
#   bash rynnvalue_eval/start_server.sh --checkpoint-path /path/to/ckpt_dir
#   bash rynnvalue_eval/start_server.sh --port 8010 --gpu 1
#   bash rynnvalue_eval/start_server.sh --stride 2 --num-frames 16
#   bash rynnvalue_eval/start_server.sh --debug                              # attach debugpy on :5678
#
# --checkpoint-path expects a directory containing model.pt; the matching
# huggingface processor/config dir must live at <parent>/huggingface/ (see
# robometer/evals/baselines/rynnvalue.py: load_model_from_checkpoint).
set -euo pipefail

PORT=8001
GPU=0
MODEL_PATH=""
CHECKPOINT_PATH=""
MODE="absolute"
STRIDE=1
NUM_FRAMES=8
USE_FRAME_STEPS=true
BATCH_SIZE=16
DEBUG=false
DEBUG_PORT=5678

usage() {
    sed -n '2,14p' "$0" >&2
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --stride) STRIDE="$2"; shift 2 ;;
        --num-frames) NUM_FRAMES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --use-frame-steps) USE_FRAME_STEPS=true; shift ;;
        --debug) DEBUG=true; shift ;;
        --debug-port) DEBUG_PORT="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown arg: $1" >&2; usage 1 ;;
    esac
done

if [[ "${DEBUG}" == "true" ]]; then
    cmd=(python -Xfrozen_modules=off -m debugpy --listen "0.0.0.0:${DEBUG_PORT}" --wait-for-client
         robometer/evals/baseline_eval_server.py)
else
    cmd=(python robometer/evals/baseline_eval_server.py)
fi

cmd+=(reward_model=rynnvalue
      model_path="${MODEL_PATH}"
      model_config.mode="${MODE}"
      model_config.stride="${STRIDE}"
      model_config.num_frames="${NUM_FRAMES}"
      server_port="${PORT}"
      use_frame_steps="${USE_FRAME_STEPS}"
      batch_size="${BATCH_SIZE}")
if [[ -n "${CHECKPOINT_PATH}" ]]; then
    cmd+=(model_config.checkpoint_path="${CHECKPOINT_PATH}")
fi

echo "+ CUDA_VISIBLE_DEVICES=${GPU} HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} ${cmd[*]}" >&2

CUDA_VISIBLE_DEVICES="${GPU}" \
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
"${cmd[@]}"
