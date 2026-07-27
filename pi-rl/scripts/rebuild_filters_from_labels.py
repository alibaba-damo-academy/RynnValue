#!/usr/bin/env python3
# adapted from openpi
"""Rebuild per-task episode filters from the authoritative `_f` labels WITHOUT
reconverting the datasets.

Why this works
--------------
Conversion preserves each trajectory's frame count 1:1: the LeRobot dataset's
`meta/episodes.jsonl` `length` equals the raw episode's `timeseries.parquet`
row count, in the same order. The success/failure label lives only in the raw
directory name (`episode_*_f` == failure). So we can fingerprint-align each
dataset episode to its raw source episode by frame length (order-preserving
LCS), read the `_f` label from the raw name, and emit a correct filter --
purely from metadata.

The dataset stores NO success flag and `processed_sources.txt` is unreliable
(cross-run accumulation / legacy bare names), so length fingerprinting is the
robust bridge.

Alignment lets B (raw) skip episodes that were jump-dropped during conversion,
and lets A (dataset) have a small number of unmatched "orphan" episodes (from
cross-run rebuilds) which default to SUCCESS (keep) so we never drop good data.

Validation gate: for tasks whose current filter is already correct
(book/bread/pen), the rebuilt success set must match exactly, proving the
aligner. Only tasks that pass validation are written under --apply.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib
import sys

import pyarrow.parquet as pq

DS_ROOT = pathlib.Path("/path/to/franka_lerobot_data_merged")
SRC_BASE = pathlib.Path("/path/to/raw_franka_data")
# Co-locate the filter with the LeRobot dataset: config's _load_episode_ids_from
# now prefers <repo_id>/filter.json, i.e. DS_ROOT/<task>/filter.json.
READ_DIR = DS_ROOT

# repo -> raw source dirs, in merge order (from convert_and_repair_all_merged.sh /
# convert_and_repair_pen_merged.sh). Pen was built from two dirs.
REGISTRY = {
    "close_the_drawer": ["close_the_drawer_new"],
    "pick_up_the_banana": ["pick_up_the_banana"],
    "pick_up_the_book": ["pick_up_the_book"],
    "pick_up_the_box": ["pick_up_the_box"],
    "pick_up_the_bread": ["pick_up_the_bread"],
    "pick_up_the_pen": ["pick_up_the_pen", "pick_up_the_pen_new_3"],
    "pick_up_the_steak": ["pick_up_the_steak"],
}


def raw_num_rows(ep_dir: str) -> int | None:
    f = os.path.join(ep_dir, "timeseries.parquet")
    try:
        return pq.ParquetFile(f).metadata.num_rows
    except Exception:
        return None


def raw_episodes(srcs: list[str]) -> list[tuple[str, int, bool]]:
    """Ordered (name, length, is_failed) across source dirs, converter order."""
    out: list[tuple[str, int, bool]] = []
    for s in srcs:
        for p in sorted(glob.glob(str(SRC_BASE / s / "episode_*"))):
            if not os.path.isdir(p):
                continue
            n = raw_num_rows(p)
            if n is None:
                continue
            out.append((os.path.basename(p), n, os.path.basename(p).endswith("_f")))
    return out


def dataset_lengths(task: str) -> list[int]:
    rows = [
        json.loads(l)
        for l in (DS_ROOT / task / "meta" / "episodes.jsonl").read_text().splitlines()
        if l.strip()
    ]
    rows.sort(key=lambda r: r["episode_index"])
    return [int(r["length"]) for r in rows]


def align(a_len: list[int], b: list[tuple[str, int, bool]]) -> list[int]:
    """Order-preserving LCS by frame length. Returns, for each A index, the
    matched B index or -1 (orphan). Maximizes matched pairs; B may skip freely
    (jump-dropped), A orphans are the unmatched remainder."""
    n, m = len(a_len), len(b)
    blen = [x[1] for x in b]
    # dp[i][j] = LCS length of a_len[:i], blen[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a_len[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            if ai == blen[j - 1]:
                row[j] = prev[j - 1] + 1
            elif prev[j] >= row[j - 1]:
                row[j] = prev[j]
            else:
                row[j] = row[j - 1]
    # backtrack
    match = [-1] * n
    i, j = n, m
    while i > 0 and j > 0:
        if a_len[i - 1] == blen[j - 1] and dp[i][j] == dp[i - 1][j - 1] + 1:
            match[i - 1] = j - 1
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return match


def candidate_sources(task: str) -> list[list[str]]:
    """Plausible raw source-dir sets for a task, most-specific first. The same task lives under
    several dataset roots built from different raw subsets, so we try candidates and pick the one
    that covers the dataset (see resolve_sources)."""
    cands: list[list[str]] = []

    def add(c: list[str]) -> None:
        if c and c not in cands:
            cands.append(c)

    if task in REGISTRY:
        add(REGISTRY[task])
    add([task])  # same-name single source
    if task == "close_the_drawer":
        add(["close_the_drawer_new"])
    if task == "pick_up_the_pen":
        add(["pick_up_the_pen"])
        add(["pick_up_the_pen", "pick_up_the_pen_new_3"])
    return cands


def resolve_sources(task: str, a_len: list[int]) -> tuple[list[str], list[tuple[str, int, bool]], int]:
    """Choose the raw source set whose frame-length multiset best covers the dataset. Returns
    (chosen_sources, raw_episodes, extra) where extra is the number of dataset episodes whose
    length is NOT covered by the chosen raw pool (0 == fully covered). Prefers extra==0 then the
    smallest raw pool (least ambiguity)."""
    a_counts = collections.Counter(a_len)
    best = None
    for cand in candidate_sources(task):
        b = raw_episodes(cand)
        if not b:
            continue
        b_counts = collections.Counter(l for _, l, _ in b)
        extra = sum(max(0, c - b_counts.get(L, 0)) for L, c in a_counts.items())
        key = (extra, len(b))
        if best is None or key < best[0]:
            best = (key, cand, b, extra)
    if best is None:
        return [task], [], len(a_len)
    return best[1], best[2], best[3]


def rebuild(task: str) -> dict:
    a_len = dataset_lengths(task)
    srcs, b, extra = resolve_sources(task, a_len)
    match = align(a_len, b)

    failed = [False] * len(a_len)
    orphans = []
    for i, j in enumerate(match):
        if j < 0:
            orphans.append(i)  # unmatched -> default SUCCESS (keep)
        else:
            failed[i] = b[j][2]
    success_idx = [i for i in range(len(a_len)) if not failed[i]]
    failed_idx = [i for i in range(len(a_len)) if failed[i]]

    raw_f = sum(1 for _, _, f in b if f)
    return {
        "task": task,
        "ds_n": len(a_len),
        "raw_n": len(b),
        "raw_f": raw_f,
        "matched": sum(1 for j in match if j >= 0),
        "orphans": orphans,
        "failed_found": sum(failed),
        "success_idx": success_idx,
        "failed_idx": failed_idx,
        "srcs": srcs,
        "extra": extra,
    }


def write_filter(path: pathlib.Path, task: str, success_idx: list[int]) -> None:
    data = {"episodes": [{"metadata": {"repo_id": task, "ep_idx": i}} for i in success_idx]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_failed(path: pathlib.Path, task: str, failed_idx: list[int]) -> None:
    data = {"repo_id": task, "failed_ep_idx": sorted(failed_idx)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def current_ids(path: pathlib.Path, emit: str) -> set[int] | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if emit == "failed":
        return {int(i) for i in d.get("failed_ep_idx", [])}
    return {int((e.get("metadata") or {}).get("ep_idx")) for e in d.get("episodes", [])
            if (e.get("metadata") or {}).get("ep_idx") is not None}


# Tasks whose current filter is trusted-correct; used as a proof of the aligner
# (only valid for --emit filter on the franka_0711 SFT datasets).
KNOWN_GOOD = {"pick_up_the_book", "pick_up_the_bread", "pick_up_the_pen"}
_FRANKA_0711 = pathlib.Path("/path/to/franka_lerobot_data_merged")


def main() -> int:
    global DS_ROOT, READ_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ds-root", default=str(DS_ROOT),
                    help="LeRobot dataset root (repo dirs live under here). Default: franka_0711.")
    ap.add_argument("--tasks", default="", help="comma-separated subset; default all in REGISTRY")
    ap.add_argument("--emit", choices=["filter", "failed"], default="filter",
                    help="filter=success whitelist filter.json (SFT); "
                         "failed=failed_episodes.json (IQL keeps failures, withholds terminal reward).")
    ap.add_argument("--apply", action="store_true", help="write co-located files (backs up any existing).")
    args = ap.parse_args()

    DS_ROOT = pathlib.Path(args.ds_root)
    READ_DIR = DS_ROOT
    out_name = "failed_episodes.json" if args.emit == "failed" else "filter.json"
    proof_ok = args.emit == "filter" and DS_ROOT == _FRANKA_0711

    tasks = [t for t in args.tasks.split(",") if t] or list(REGISTRY)

    label = "failed" if args.emit == "failed" else "succ"
    print(f"emit={args.emit}  ds_root={DS_ROOT}")
    print(f"{'task':<20}{'ds':>5}{'raw':>5}{'raw_f':>6}{'orph':>5}{'new_'+label:>9}{'cur':>5}{'Δadd':>6}{'Δrm':>5}  validation")
    print("-" * 92)
    results = []
    validation_failed = []
    for t in tasks:
        r = rebuild(t)
        out_path = READ_DIR / t / out_name
        new_ids = set(r["failed_idx"] if args.emit == "failed" else r["success_idx"])
        cur = current_ids(out_path, args.emit)
        add = len(new_ids - cur) if cur is not None else len(new_ids)
        rm = len(cur - new_ids) if cur is not None else 0

        note = ""
        if proof_ok and t in KNOWN_GOOD:
            good = current_ids(
                pathlib.Path("/path/to/pi-rl/logs/convert_and_repair_all_merged/filters") / f"{t}.json",
                "filter",
            )
            if good is not None and good == set(r["success_idx"]):
                note = "PROOF ok (==known-good)"
            else:
                note = f"PROOF FAIL (diff {len(good ^ set(r['success_idx'])) if good else '?'})"
                validation_failed.append(t)
        if len(r["orphans"]) > 2:
            note = (note + f" ; MANY orphans={len(r['orphans'])}").strip(" ;")
            validation_failed.append(t)
        if r["extra"] > 0:
            note = (note + f" ; {r['extra']} uncovered->success").strip(" ;")
        note = (note + f"  src={'+'.join(r['srcs'])}").strip()

        print(f"{t:<20}{r['ds_n']:>5}{r['raw_n']:>5}{r['raw_f']:>6}{len(r['orphans']):>5}"
              f"{len(new_ids):>9}{(len(cur) if cur is not None else -1):>5}{add:>6}{rm:>5}  {note}")
        results.append((t, r, out_path, new_ids))

    if not args.apply:
        print(f"\n[dry-run] no files written. Re-run with --apply to write co-located {out_name} (backs up existing).")
        return 0

    if validation_failed:
        print(f"\n[abort] validation failed for: {sorted(set(validation_failed))}. Not writing anything.")
        return 1

    print()
    for t, r, out_path, new_ids in results:
        if out_path.exists():
            bak = out_path.with_suffix(".json.prealign.bak")
            if not bak.exists():
                bak.write_text(out_path.read_text())
                print(f"[backup] {out_path} -> {bak.name}")
        if args.emit == "failed":
            write_failed(out_path, t, r["failed_idx"])
        else:
            write_filter(out_path, t, r["success_idx"])
        print(f"[write ] {out_path}  ({len(new_ids)} {label})")
    print(f"\nDone. Wrote co-located {out_name} for {len(results)} task(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
