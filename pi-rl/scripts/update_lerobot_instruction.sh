#!/bin/bash
# Update the language instruction stored in a LeRobot dataset's meta files.
#
# Edits both `meta/tasks.jsonl` (task catalog) and `meta/episodes.jsonl`
# (per-episode task list) so the dataset reads the new prompt at training
# and inference time. Existing action/state/image data is left untouched.
#
# Default usage: fix the `pick_up_the_box` instruction
#   "Move the box from the left side to the right side."  ->  "Move the box to the other side."
#
# Usage:
#   bash scripts/update_lerobot_instruction.sh
#   bash scripts/update_lerobot_instruction.sh \
#        --task-dir /path/to/pick_up_the_box \
#        --old "Move the box from the left side to the right side." \
#        --new "Move the box to the other side."
#   bash scripts/update_lerobot_instruction.sh --dry-run    # preview only

set -euo pipefail

TASK_DIR="${TASK_DIR:-/path/to/franka_lerobot_data/pick_up_the_box}"
OLD_INSTRUCTION="${OLD_INSTRUCTION:-Move the box from the left side to the right side.}"
NEW_INSTRUCTION="${NEW_INSTRUCTION:-Move the box to the other side.}"
DRY_RUN=0
NO_BACKUP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-dir)     TASK_DIR="$2";        shift 2 ;;
        --old)          OLD_INSTRUCTION="$2"; shift 2 ;;
        --new)          NEW_INSTRUCTION="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=1;            shift   ;;
        --no-backup)    NO_BACKUP=1;          shift   ;;
        -h|--help)
            sed -n '2,17p' "$0"
            exit 0
            ;;
        *)
            echo "[error] unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

TASKS_JSONL="${TASK_DIR}/meta/tasks.jsonl"
EPISODES_JSONL="${TASK_DIR}/meta/episodes.jsonl"

if [[ ! -f "$TASKS_JSONL" ]]; then
    echo "[error] tasks.jsonl not found: $TASKS_JSONL" >&2
    exit 1
fi
if [[ ! -f "$EPISODES_JSONL" ]]; then
    echo "[error] episodes.jsonl not found: $EPISODES_JSONL" >&2
    exit 1
fi

OLD_COUNT_TASKS=$(grep -cF "$OLD_INSTRUCTION" "$TASKS_JSONL" || true)
OLD_COUNT_EPS=$(grep -cF "$OLD_INSTRUCTION" "$EPISODES_JSONL" || true)

echo "=========================================="
echo " LeRobot instruction updater"
echo "=========================================="
echo " TASK_DIR       : $TASK_DIR"
echo " OLD instruction: \"$OLD_INSTRUCTION\""
echo " NEW instruction: \"$NEW_INSTRUCTION\""
echo " tasks.jsonl    : $OLD_COUNT_TASKS occurrence(s)"
echo " episodes.jsonl : $OLD_COUNT_EPS occurrence(s)"
echo " DRY_RUN        : $DRY_RUN"
echo " NO_BACKUP      : $NO_BACKUP"
echo "=========================================="

if [[ "$OLD_COUNT_TASKS" -eq 0 && "$OLD_COUNT_EPS" -eq 0 ]]; then
    echo "[info] nothing to update - old instruction not found."
    exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo ""
    echo "[dry-run] would edit:"
    echo "  $TASKS_JSONL"
    echo "  $EPISODES_JSONL"
    echo ""
    echo "[dry-run] preview of tasks.jsonl after change:"
    sed "s|${OLD_INSTRUCTION}|${NEW_INSTRUCTION}|g" "$TASKS_JSONL"
    exit 0
fi

if [[ "$NO_BACKUP" != "1" ]]; then
    ts=$(date +%Y%m%d_%H%M%S)
    cp "$TASKS_JSONL"    "${TASKS_JSONL}.bak.${ts}"
    cp "$EPISODES_JSONL" "${EPISODES_JSONL}.bak.${ts}"
    echo "[info] backups:"
    echo "  ${TASKS_JSONL}.bak.${ts}"
    echo "  ${EPISODES_JSONL}.bak.${ts}"
fi

# Safe literal-string substitution via python (avoids sed escaping issues
# with `/`, `&`, etc. in the prompt text).
python3 -c "
import sys, pathlib
tasks, eps, old, new = [pathlib.Path(p) if i < 2 else p for i, p in enumerate(sys.argv[1:])]
for p in (tasks, eps):
    txt = p.read_text(encoding='utf-8')
    new_txt = txt.replace(old, new)
    if new_txt == txt:
        print(f'[info] {p}: no change')
    else:
        p.write_text(new_txt, encoding='utf-8')
        print(f'[ok]   {p}: updated')
" "$TASKS_JSONL" "$EPISODES_JSONL" "$OLD_INSTRUCTION" "$NEW_INSTRUCTION"

echo ""
echo "=========================================="
echo " Verification"
echo "=========================================="
echo "-- tasks.jsonl --"
cat "$TASKS_JSONL"
echo ""
echo "-- episodes.jsonl (head 3) --"
head -n 3 "$EPISODES_JSONL"
echo ""
echo "-- grep new instruction --"
echo "  tasks.jsonl    : $(grep -cF "$NEW_INSTRUCTION" "$TASKS_JSONL") hit(s)"
echo "  episodes.jsonl : $(grep -cF "$NEW_INSTRUCTION" "$EPISODES_JSONL") hit(s)"
echo "  (old residue)  : tasks=$(grep -cF "$OLD_INSTRUCTION" "$TASKS_JSONL" || true)  eps=$(grep -cF "$OLD_INSTRUCTION" "$EPISODES_JSONL" || true)"
echo "=========================================="
echo " Done."
