#!/usr/bin/env bash
# Parallel launcher: runs all 7 RynnValue Franka tasks concurrently via
# convert_and_repair_data.sh.
#
# Each task writes its own stdout/stderr to logs/<repo>.{out,err}. The main
# script shows a live, interleaved prefix-log stream plus a final summary.
#
# Concurrency is capped by PARALLEL_JOBS (default 2) because each task itself
# fans out to NUM_WORKERS Python processes + EPISODE_IMAGE_THREADS image
# threads. Running all 7 at once easily saturates the CPU / parquet I/O,
# so keep PARALLEL_JOBS low unless you have a very big box.
#
# Usage:
#   bash scripts/convert_and_repair_all.sh
#
# Override knobs (all env-driven):
#   PARALLEL_JOBS=4 bash scripts/convert_and_repair_all.sh
#   FRANKA_BASE=/alt/path bash scripts/convert_and_repair_all.sh
#   CLIP_GRIPPER_MAX=250 JUMP_THRESHOLD=0.2 bash scripts/convert_and_repair_all.sh
#
# Re-run only tasks that failed last time:
#   RERUN_FAILED=true bash scripts/convert_and_repair_all.sh

set -uo pipefail
set +e  # never exit on a child failure; we collect results ourselves

# Resolve project root from this script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Configuration ───────────────────────────────────────────────────────────
FRANKA_BASE="${FRANKA_BASE:-/path/to/raw_franka_data}"
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/convert_and_repair_all}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RERUN_FAILED="${RERUN_FAILED:-false}"   # skip tasks whose repo output already exists
# ────────────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " Parallel Franka -> LeRobot conversion (7 tasks)"
echo "============================================================"
echo "PROJECT_ROOT   = $PROJECT_ROOT"
echo "FRANKA_BASE    = $FRANKA_BASE"
echo "PARALLEL_JOBS  = $PARALLEL_JOBS"
echo "LOG_DIR        = $LOG_DIR"
echo "TIMESTAMP      = $TIMESTAMP"
echo "RERUN_FAILED   = $RERUN_FAILED"
echo "CLIP_GRIPPER_MAX      = ${CLIP_GRIPPER_MAX:-<default>}"
echo "JUMP_THRESHOLD        = ${JUMP_THRESHOLD:-<default>}"
echo "MAX_JUMPS_PER_EPISODE = ${MAX_JUMPS_PER_EPISODE:-<default>}"
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

# Task registry: src_dir | repo_name | language_instruction
#
# Order here is the submission order; they run in whatever order the
# semaphore allows.
TASK_LIST=(
    "close_the_drawer_new|close_the_drawer|Put the box on the table into the drawer, then close the drawer."
    "pick_up_the_banana|pick_up_the_banana|Move the basket from the right side to the left, then place the banana held by the left gripper into the basket."
    "pick_up_the_book|pick_up_the_book|Place the book onto the bookshelf."
    "pick_up_the_box|pick_up_the_box|Move the box to the other side."
    "pick_up_the_bread|pick_up_the_bread|Pick up the two breads from the table and put them in the basket."
    "pick_up_the_pen|pick_up_the_pen|Pick up the pen from the desk, then put it into the pen holder."
    "pick_up_the_steak|pick_up_the_steak|Move the steak from the pan to the plate."
)

run_task() {
    local src_dir="$1"
    local repo_name="$2"
    local lang="$3"
    local out_log="$LOG_DIR/${repo_name}.out"
    local err_log="$LOG_DIR/${repo_name}.err"

    : > "$out_log"
    : > "$err_log"

    echo "[task:${repo_name}] start" | tee -a "$out_log" >&2

    (
        export DATA_DIR="$FRANKA_BASE/$src_dir"
        export REPO_NAME="$repo_name"
        export LANGUAGE_INSTRUCTION="$lang"
        export FRANKA_HOME="${FRANKA_HOME:-/path/to/franka_lerobot_data_clean}"
        exec bash "$SCRIPT_DIR/convert_and_repair_data.sh"
    ) >"$out_log" 2>"$err_log"
    local rc=$?

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
    IFS='|' read -r src_dir repo_name lang <<< "$entry"

    # Optional skip: if the repo output already exists and RERUN_FAILED=false,
    # mark it "skipped" and move on. Useful when resuming after a partial run.
    if [ "$RERUN_FAILED" = "true" ]; then
        repo_out="$FRANKA_HOME/$repo_name"
        if [ -d "$repo_out" ]; then
            echo "[skip] $repo_name: $repo_out already exists (RERUN_FAILED=true)"
            TASK_STATUS["$repo_name"]="skipped"
            n_skipped=$((n_skipped + 1))
            continue
        fi
    fi

    out_log="$LOG_DIR/${repo_name}.out"
    err_log="$LOG_DIR/${repo_name}.err"
    TASK_OUT_LOG["$repo_name"]="$out_log"
    TASK_ERR_LOG["$repo_name"]="$err_log"

    # Acquire a slot (blocks when all slots are taken)
    read -r <&3

    (
        run_task "$src_dir" "$repo_name" "$lang"
        rc=$?
        # Release the slot (the trailing echo wakes `read -r <&3` above)
        echo >&3
        exit $rc
    ) &
    pid=$!
    TASK_PIDS["$repo_name"]=$pid
    TASK_START_TIME["$repo_name"]="$(date +%s)"
    echo "[submit] $repo_name (pid=$pid)"
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
    IFS='|' read -r _ repo_name _ <<< "$entry"
    status="${TASK_STATUS[$repo_name]:-not run}"
    printf "  %-28s %s\n" "$repo_name" "$status"
done
echo "------------------------------------------------------------"
echo "  total=${n_total}  ok=${n_ok}  failed=${n_fail}  skipped=${n_skipped}"
echo "  wall time = ${wall_elapsed}s"
echo "  logs      = $LOG_DIR"
echo "============================================================"

if [ "$n_fail" -gt 0 ]; then
    echo ""
    echo "Re-run failed tasks only:"
    echo "  RERUN_FAILED=true bash scripts/convert_and_repair_all.sh"
    exit 1
fi
exit 0
