#!/usr/bin/env bash
# Merge-convert the pick_up_the_pen source dirs into ONE LeRobot repo.
#
#   pick_up_the_pen        episode_000001 .. episode_000110   (99 eps, incl _f)
#   pick_up_the_pen_new_3  episode_000001 .. episode_000100   (100 eps, incl _f)
#
# They share the SAME task (pick up the pen) but live in separate dirs. NOTE
# their episode_* directory names OVERLAP (80 shared names). This script feeds
# them, in order, into a single repo `pick_up_the_pen` by reusing
# convert_and_repair_data.sh's RESUME mechanism:
#
#   segment 1  (pick_up_the_pen)        RESUME=false  -> create the repo fresh
#   segment 2  (pick_up_the_pen_new_3)  RESUME=true   -> append
#
# Why this is safe to merge via RESUME:
#   - The resume path de-duplicates source episodes by a SOURCE-NAMESPACED key
#     ("<src_dir>/<episode>") in processed_sources.txt, so two dirs reusing the
#     same episode_* name (as pen and new_3 do) are kept distinct and no
#     episode from a later segment is wrongly skipped.
#   - RESUME also asserts the on-disk feature layout matches the freshly
#     detected active-arm mode. If any pen dir has a different arm config the
#     append will abort loudly rather than mix incompatible episodes -- that
#     is the desired behaviour.
#
# The episode filter (list of successful ep_idx) is accumulated across all
# three segments into ONE shared FILTER_PATH: segment 1 writes it fresh, and
# segments 2/3 merge their new successes into it.
#
# Usage:
#   bash scripts/convert_and_repair_pen_merged.sh
#
# Resume an interrupted merge (do NOT rebuild from scratch; make ALL three
# segments resume so already-saved episodes are skipped):
#   RESUME_ALL=true bash scripts/convert_and_repair_pen_merged.sh
#
# Repair knobs are forwarded to the child script via the environment, e.g.:
#   CLIP_GRIPPER_MAX=250 JUMP_THRESHOLD=0.2 MAX_JUMPS_PER_EPISODE=1 \
#       bash scripts/convert_and_repair_pen_merged.sh

set -euo pipefail

# Resolve project root from this script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Configuration (override via env vars) ───────────────────────────────────
FRANKA_BASE="${FRANKA_BASE:-/path/to/raw_franka_data}"
FRANKA_HOME="${FRANKA_HOME:-/path/to/franka_lerobot_data_merged}"
REPO_NAME="${REPO_NAME:-pick_up_the_pen}"
LANGUAGE_INSTRUCTION="${LANGUAGE_INSTRUCTION:-Put the pen in the pen holder.}"

# One shared filter for the merged repo, aligned with convert_and_repair_all_merged.sh
# so both pipelines read/write the SAME pen filter. Kept OUTSIDE the output repo
# (which segment 1 wipes) and off the OSS mount, so it is fast and never clobbered.
FILTER_PATH="${FILTER_PATH:-${PROJECT_ROOT}/logs/convert_and_repair_all_merged/filters/${REPO_NAME}.json}"

# RESUME_ALL=true -> every segment resumes (safe for continuing a broken run).
# RESUME_ALL=false (default) -> segment 1 rebuilds the repo from scratch.
RESUME_ALL="${RESUME_ALL:-false}"
# ────────────────────────────────────────────────────────────────────────────

# Source dirs, in merge order. Segment 1 seeds the repo; the rest append.
SRC_DIRS=(
    "pick_up_the_pen"
    "pick_up_the_pen_new_3"
)

CONVERT_ONE="$SCRIPT_DIR/convert_and_repair_data.sh"
if [ ! -f "$CONVERT_ONE" ]; then
    echo "[ERROR] Cannot find child script: $CONVERT_ONE" >&2
    exit 1
fi

mkdir -p "$(dirname "$FILTER_PATH")"

echo "============================================================"
echo " Merge 3 pen dirs -> single LeRobot repo '$REPO_NAME'"
echo "============================================================"
echo "PROJECT_ROOT         = $PROJECT_ROOT"
echo "FRANKA_BASE          = $FRANKA_BASE"
echo "FRANKA_HOME (output) = $FRANKA_HOME"
echo "REPO_NAME            = $REPO_NAME"
echo "LANGUAGE_INSTRUCTION = $LANGUAGE_INSTRUCTION"
echo "FILTER_PATH (shared) = $FILTER_PATH"
echo "RESUME_ALL           = $RESUME_ALL"
echo "SEGMENTS             = ${SRC_DIRS[*]}"
echo "CLIP_GRIPPER_MAX     = ${CLIP_GRIPPER_MAX:-<child default>}"
echo "JUMP_THRESHOLD       = ${JUMP_THRESHOLD:-<child default>}"
echo "MAX_JUMPS_PER_EPISODE= ${MAX_JUMPS_PER_EPISODE:-<child default>}"
echo "============================================================"
echo ""

wall_start="$(date +%s)"
n_total=${#SRC_DIRS[@]}

for i in "${!SRC_DIRS[@]}"; do
    src="${SRC_DIRS[$i]}"
    seg=$(( i + 1 ))

    # First segment builds fresh (unless resuming a broken run); the rest append.
    if [ "$i" -eq 0 ] && [ "$RESUME_ALL" != "true" ]; then
        seg_resume="false"
        seg_video="true"    # sample preview only needed once
    else
        seg_resume="true"
        seg_video="false"   # avoid overwriting segment 1's preview
    fi

    src_path="$FRANKA_BASE/$src"
    if [ ! -d "$src_path" ]; then
        echo "[ERROR] Segment $seg source dir missing: $src_path" >&2
        exit 1
    fi

    echo "------------------------------------------------------------"
    echo "[segment $seg/$n_total] $src  (RESUME=$seg_resume)"
    echo "------------------------------------------------------------"

    # Repair knobs (CLIP_GRIPPER_MAX, JUMP_THRESHOLD, MAX_JUMPS_PER_EPISODE)
    # set in this script's environment are inherited by the child bash.
    DATA_DIR="$src_path" \
    REPO_NAME="$REPO_NAME" \
    LANGUAGE_INSTRUCTION="$LANGUAGE_INSTRUCTION" \
    FILTER_PATH="$FILTER_PATH" \
    FRANKA_HOME="$FRANKA_HOME" \
    RESUME="$seg_resume" \
    GENERATE_SAMPLE_VIDEO="$seg_video" \
    SAMPLE_VIDEO_NAME="sample_${REPO_NAME}" \
        bash "$CONVERT_ONE"

    echo "[segment $seg/$n_total] $src done."
    echo ""
done

wall_elapsed=$(( $(date +%s) - wall_start ))

echo "============================================================"
echo " Merge complete"
echo "============================================================"
echo "  Merged repo   = $FRANKA_HOME/$REPO_NAME"
echo "  Episode filter= $FILTER_PATH"
echo "  Segments      = $n_total  (${SRC_DIRS[*]})"
echo "  Wall time     = ${wall_elapsed}s"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Inspect the sample videos under $SCRIPT_DIR/sample_${REPO_NAME}/"
echo "  2. Edit $FILTER_PATH to exclude any remaining bad episodes"
echo "  3. Run calculate_norm.sh against repo '$REPO_NAME'"
