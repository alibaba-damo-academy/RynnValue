#!/usr/bin/env bash
# Parallel launcher that converts ALL RynnValue Franka datasets to LeRobot
# format via convert_and_repair_data.sh, with support for MERGING several
# source dirs into a single repo.
#
# This is a superset of convert_and_repair_all.sh: each task now maps ONE
# repo to ONE-OR-MORE source dirs. Multi-source repos (e.g. pick_up_the_pen)
# are merged by feeding their dirs sequentially into the same repo using the
# child script's RESUME mechanism (segment 1 builds fresh, later segments
# append). Different repos still run concurrently under a job semaphore.
#
# Task registry format:  repo_name | language_instruction | src1[,src2,...]
#
# Layout of side outputs (each repo isolated, so parallel repos never clash):
#   logs/<repo>.{out,err}                 per-repo stdout/stderr
#   logs/.../filters/<repo>.json          per-repo episode filter (accumulated
#                                          across that repo's source segments)
#   scripts/sample_<repo>/                per-repo sample preview videos
#
# Concurrency is capped by PARALLEL_JOBS (default 2) because each task itself
# fans out to NUM_WORKERS Python processes + EPISODE_IMAGE_THREADS image
# threads. A merged repo also runs its source dirs one after another, so its
# wall time is the sum of its segments.
#
# Usage:
#   bash scripts/convert_and_repair_all_merged.sh
#
# Override knobs (all env-driven):
#   PARALLEL_JOBS=4 bash scripts/convert_and_repair_all_merged.sh
#   FRANKA_BASE=/alt/path bash scripts/convert_and_repair_all_merged.sh
#   CLIP_GRIPPER_MAX=250 JUMP_THRESHOLD=0.2 bash scripts/convert_and_repair_all_merged.sh
#
# Skip repos whose output already exists (coarse repo-level resume):
#   RERUN_FAILED=true bash scripts/convert_and_repair_all_merged.sh
#
# Continue an interrupted run WITHOUT rebuilding (every source segment of a
# processed repo resumes instead of wiping segment 1):
#   RESUME_ALL=true bash scripts/convert_and_repair_all_merged.sh

set -uo pipefail
set +e  # never exit on a child failure; we collect results ourselves

# Resolve project root from this script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Configuration ───────────────────────────────────────────────────────────
FRANKA_BASE="${FRANKA_BASE:-/path/to/raw_franka_data}"
FRANKA_HOME="${FRANKA_HOME:-/path/to/franka_lerobot_data_merged}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/convert_and_repair_all_merged}"
FILTER_DIR="${FILTER_DIR:-${LOG_DIR}/filters}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RERUN_FAILED="${RERUN_FAILED:-false}"   # skip repos whose output already exists
RESUME_ALL="${RESUME_ALL:-true}"       # every segment resumes (continue a broken run)
# ────────────────────────────────────────────────────────────────────────────

CONVERT_ONE="$SCRIPT_DIR/convert_and_repair_data.sh"
if [ ! -f "$CONVERT_ONE" ]; then
    echo "[ERROR] Cannot find child script: $CONVERT_ONE" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$FILTER_DIR"

# Task registry: repo_name | language_instruction | src1[,src2,...]
#
# pick_up_the_pen merges 3 source dirs (verified non-overlapping episode
# names, so RESUME-based append never wrongly skips). All other repos are
# single-source and match convert_and_repair_all.sh's mapping.
TASK_LIST=(
    "close_the_drawer|Put the box in the drawer and close it.|close_the_drawer_new"
    "pick_up_the_banana|Put the banana in the basket and move the basket left.|pick_up_the_banana"
    "pick_up_the_book|Put the book on the bookshelf.|pick_up_the_book"
    "pick_up_the_box|Move the box to the other side.|pick_up_the_box"
    "pick_up_the_bread|Put the two breads in the basket.|pick_up_the_bread"
    "pick_up_the_pen|Put the pen in the pen holder.|pick_up_the_pen"
    "pick_up_the_steak|Move the steak from the pan to the plate.|pick_up_the_steak"
)


echo "============================================================"
echo " Parallel Franka -> LeRobot conversion (merged, ${#TASK_LIST[@]} repos)"
echo "============================================================"
echo "PROJECT_ROOT   = $PROJECT_ROOT"
echo "FRANKA_BASE    = $FRANKA_BASE"
echo "FRANKA_HOME    = $FRANKA_HOME"
echo "PARALLEL_JOBS  = $PARALLEL_JOBS"
echo "LOG_DIR        = $LOG_DIR"
echo "FILTER_DIR     = $FILTER_DIR"
echo "TIMESTAMP      = $TIMESTAMP"
echo "RERUN_FAILED   = $RERUN_FAILED"
echo "RESUME_ALL     = $RESUME_ALL"
echo "CLIP_GRIPPER_MAX      = ${CLIP_GRIPPER_MAX:-<default>}"
echo "JUMP_THRESHOLD        = ${JUMP_THRESHOLD:-<default>}"
echo "MAX_JUMPS_PER_EPISODE = ${MAX_JUMPS_PER_EPISODE:-<default>}"
echo "------------------------------------------------------------"
echo " Task groups (repo <- sources):"
for entry in "${TASK_LIST[@]}"; do
    IFS='|' read -r repo_name _ srcs_csv <<< "$entry"
    printf "   %-22s <- %s\n" "$repo_name" "$srcs_csv"
done
echo "============================================================"
echo ""

# Job-slot semaphore: a FIFO whose line count equals PARALLEL_JOBS.
# Acquire = read a line; release = write a line back.
SEMAPHORE_FIFO="$(mktemp -u)"
mkfifo "$SEMAPHORE_FIFO"
exec 3<>"$SEMAPHORE_FIFO"
rm -f "$SEMAPHORE_FIFO"
for (( i = 0; i < PARALLEL_JOBS; i++ )); do
    echo >&3
done

# Associative arrays to track per-task state
declare -A TASK_PIDS
declare -A TASK_STATUS
declare -A TASK_OUT_LOG
declare -A TASK_ERR_LOG
declare -A TASK_START_TIME

# Convert one repo, merging its (comma-separated) source dirs in order.
# Segment 1 builds fresh (unless RESUME_ALL); later segments RESUME=true so
# their episodes are appended to the same repo.
run_task() {
    local repo_name="$1"
    local lang="$2"
    local srcs_csv="$3"
    local out_log="$LOG_DIR/${repo_name}.out"
    local err_log="$LOG_DIR/${repo_name}.err"
    local filter_path="$FILTER_DIR/${repo_name}.json"

    : > "$out_log"
    : > "$err_log"

    echo "[task:${repo_name}] start" | tee -a "$out_log" >&2

    local -a srcs
    IFS=',' read -ra srcs <<< "$srcs_csv"

    local rc=0
    {
        local idx=0
        local n_seg=${#srcs[@]}
        for src in "${srcs[@]}"; do
            local seg_resume seg_video
            if [ "$idx" -eq 0 ] && [ "$RESUME_ALL" != "true" ]; then
                seg_resume="false"   # build the repo fresh
                seg_video="true"     # one sample preview per repo
            else
                seg_resume="true"    # append to existing repo
                seg_video="false"
            fi

            local src_path="$FRANKA_BASE/$src"
            if [ ! -d "$src_path" ]; then
                echo "[task:${repo_name}] ERROR: missing source dir: $src_path" >&2
                rc=1
                break
            fi

            echo "[task:${repo_name}] segment $((idx + 1))/${n_seg}: $src (RESUME=$seg_resume)"

            # Repair knobs (CLIP_GRIPPER_MAX, JUMP_THRESHOLD, MAX_JUMPS_PER_EPISODE)
            # present in this script's env are inherited by the child bash.
            DATA_DIR="$src_path" \
            REPO_NAME="$repo_name" \
            LANGUAGE_INSTRUCTION="$lang" \
            FILTER_PATH="$filter_path" \
            FRANKA_HOME="$FRANKA_HOME" \
            RESUME="$seg_resume" \
            GENERATE_SAMPLE_VIDEO="$seg_video" \
            SAMPLE_VIDEO_NAME="sample_${repo_name}" \
                bash "$CONVERT_ONE"
            local seg_rc=$?

            if [ "$seg_rc" -ne 0 ]; then
                echo "[task:${repo_name}] segment '$src' FAILED (rc=$seg_rc)" >&2
                rc=$seg_rc
                break
            fi
            idx=$((idx + 1))
        done
    } >>"$out_log" 2>>"$err_log"

    if [ $rc -eq 0 ]; then
        echo "[task:${repo_name}] done (rc=0)" >> "$out_log"
    else
        echo "[task:${repo_name}] FAILED (rc=$rc)" >> "$err_log"
    fi
    return $rc
}

# Ctrl+C: kill every child and drain the semaphore
cleanup() {
    echo ""
    echo "[CTRL+C] killing child tasks..."
    for pid in "${!TASK_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid in "${!TASK_PIDS[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    exec 3>&-
    exit 130
}
trap cleanup INT TERM

# ── Submit tasks ────────────────────────────────────────────────────────────
n_total=${#TASK_LIST[@]}
n_skipped=0
n_submitted=0
wall_start="$(date +%s)"

for entry in "${TASK_LIST[@]}"; do
    IFS='|' read -r repo_name lang srcs_csv <<< "$entry"

    # Coarse repo-level skip: if the repo output already exists and
    # RERUN_FAILED=true, mark it "skipped". Useful to only fill in repos that
    # were never produced by a prior run.
    if [ "$RERUN_FAILED" = "true" ]; then
        repo_out="$FRANKA_HOME/$repo_name"
        if [ -d "$repo_out" ]; then
            echo "[skip] $repo_name: $repo_out already exists (RERUN_FAILED=true)"
            TASK_STATUS["$repo_name"]="skipped"
            n_skipped=$((n_skipped + 1))
            continue
        fi
    fi

    TASK_OUT_LOG["$repo_name"]="$LOG_DIR/${repo_name}.out"
    TASK_ERR_LOG["$repo_name"]="$LOG_DIR/${repo_name}.err"

    # Acquire a slot (blocks when all slots are taken)
    read -r <&3

    (
        run_task "$repo_name" "$lang" "$srcs_csv"
        rc=$?
        # Release the slot (the trailing echo wakes `read -r <&3` above)
        echo >&3
        exit $rc
    ) &
    pid=$!
    TASK_PIDS["$repo_name"]=$pid
    TASK_START_TIME["$repo_name"]="$(date +%s)"
    echo "[submit] $repo_name (pid=$pid)  sources: $srcs_csv"
    n_submitted=$((n_submitted + 1))
done

# ── Wait for every submitted task and report ────────────────────────────────
echo ""
echo "[wait] $n_submitted task(s) running, cap=$PARALLEL_JOBS"
echo ""

n_ok=0
n_fail=0
for repo_name in "${!TASK_PIDS[@]}"; do
    pid="${TASK_PIDS[$repo_name]}"
    wait "$pid"
    rc=$?
    elapsed=$(( $(date +%s) - ${TASK_START_TIME[$repo_name]} ))
    if [ $rc -eq 0 ]; then
        echo "[ok]   $repo_name  (${elapsed}s)"
        TASK_STATUS["$repo_name"]="ok"
        n_ok=$((n_ok + 1))
    else
        echo "[FAIL] $repo_name  (rc=$rc, ${elapsed}s)"
        echo "       stdout tail:"
        tail -n 5 "${TASK_OUT_LOG[$repo_name]}" | sed 's/^/         /'
        echo "       stderr tail:"
        tail -n 5 "${TASK_ERR_LOG[$repo_name]}" | sed 's/^/         /'
        TASK_STATUS["$repo_name"]="failed (rc=$rc)"
        n_fail=$((n_fail + 1))
    fi
done

exec 3>&-

wall_elapsed=$(( $(date +%s) - wall_start ))

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Summary"
echo "============================================================"
for entry in "${TASK_LIST[@]}"; do
    IFS='|' read -r repo_name _ srcs_csv <<< "$entry"
    status="${TASK_STATUS[$repo_name]:-not run}"
    printf "  %-22s %-16s (%s)\n" "$repo_name" "$status" "$srcs_csv"
done
echo "------------------------------------------------------------"
echo "  total=${n_total}  ok=${n_ok}  failed=${n_fail}  skipped=${n_skipped}"
echo "  wall time = ${wall_elapsed}s"
echo "  outputs   = $FRANKA_HOME"
echo "  logs      = $LOG_DIR"
echo "  filters   = $FILTER_DIR"
echo "============================================================"

if [ "$n_fail" -gt 0 ]; then
    echo ""
    echo "Re-run: continue without rebuilding finished repos:"
    echo "  RESUME_ALL=true bash scripts/convert_and_repair_all_merged.sh"
    echo "Or skip repos whose output already exists:"
    echo "  RERUN_FAILED=true bash scripts/convert_and_repair_all_merged.sh"
    exit 1
fi
exit 0
