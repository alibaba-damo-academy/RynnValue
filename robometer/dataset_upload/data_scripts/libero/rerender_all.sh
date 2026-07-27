#!/usr/bin/env bash
set -euo pipefail

# Rerender LIBERO datasets across suites using uv + rerender_libero.py
# Usage:
#   ./libero/rerender_all.sh
#
# Optional env vars:
#   UV_BIN=uv                # override uv binary
#   SUITES="a b c"           # override suites list (space-separated)
#   DEBUG_MODE=1             # pass --debug_mode to the python script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(pwd)"

UV_BIN=${UV_BIN:-uv}
PY_SCRIPT="${PROJECT_ROOT}/dataset_upload/data_scripts/libero/rerender_libero.py"
REQS_FILE="${PROJECT_ROOT}/deps/libero/LIBERO/requirements.txt"
DATASETS_DIR="${PROJECT_ROOT}/deps/libero/LIBERO/libero/datasets"

# Default suites to process if not overridden
DEFAULT_SUITES=(
  "libero_spatial"
  "libero_object"
  "libero_goal"
  # "libero_10"
  "libero_90"
)

if [[ -n "${SUITES:-}" ]]; then
  # shellcheck disable=SC2206
  SUITES_ARR=( ${SUITES} )
else
  SUITES_ARR=( "${DEFAULT_SUITES[@]}" )
fi

DEBUG_FLAG=""
if [[ "${DEBUG_MODE:-0}" == "1" ]]; then
  DEBUG_FLAG="--debug_mode"
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Datasets dir: ${DATASETS_DIR}"
echo "Suites: ${SUITES_ARR[*]}"

for SUITE in "${SUITES_ARR[@]}"; do
  RAW_DIR="${DATASETS_DIR}/${SUITE}"
  TARGET_DIR="${DATASETS_DIR}/${SUITE}_256"

  if [[ ! -d "${RAW_DIR}" ]]; then
    echo "[skip] ${SUITE}: raw data dir not found: ${RAW_DIR}" >&2
    continue
  fi

  echo "[run] ${SUITE} → ${TARGET_DIR}"
  "${UV_BIN}" run --with-requirements "${REQS_FILE}" \
    "${PY_SCRIPT}" \
    --libero_task_suite "${SUITE}" \
    --libero_raw_data_dir "${RAW_DIR}" \
    --libero_target_dir "${TARGET_DIR}" \
    ${DEBUG_FLAG}
done

echo "All requested suites processed."


