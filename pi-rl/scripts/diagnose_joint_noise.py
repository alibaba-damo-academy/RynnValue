# adapted from openpi
"""
Scan every episode in a Franka source dataset and score joint-position noise.

For each `observation.state.arm` and `action.arm` (active arm dims only), we
compute:
  - max |diff|             (worst single-frame jump)
  - p99 |diff|
  - jerk RMS               (second-difference RMS; smooth trajectories keep
                            this near zero even with high velocity)
  - #frames with |diff|>thr for a few thresholds

Results are written to `noisy_report.csv` (sorted by jerk RMS desc) so you can
pick a cut and feed a delete list to a follow-up script.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

ARM_ACTIVITY_THRESH = 0.1


def detect_mode(data_dir: Path, scan: int = 5) -> str:
    ep_dirs = sorted(data_dir.glob("episode_*"))[:scan]
    arm_std_l, arm_std_r = [], []
    for ep in ep_dirs:
        ts = ep / "timeseries.parquet"
        if not ts.exists():
            continue
        df = pd.read_parquet(ts, columns=["action.arm"])
        if len(df) == 0:
            continue
        arm = np.stack(df["action.arm"].values).astype(np.float32)
        arm_std_l.append(arm[:, :7].std())
        arm_std_r.append(arm[:, 7:].std())
    l = float(np.mean(arm_std_l)) if arm_std_l else 0.0
    r = float(np.mean(arm_std_r)) if arm_std_r else 0.0
    if l > ARM_ACTIVITY_THRESH and r > ARM_ACTIVITY_THRESH:
        return "dual-arm"
    if l > ARM_ACTIVITY_THRESH:
        return "single-arm-left"
    return "single-arm-right"


def slice_active(arm_all: np.ndarray, mode: str) -> np.ndarray:
    if mode == "dual-arm":
        return arm_all
    if mode == "single-arm-left":
        return arm_all[:, :7]
    return arm_all[:, 7:]


def score_episode(ts_path: Path, mode: str, thresholds):
    df = pd.read_parquet(
        ts_path,
        columns=["frame_index", "action.arm", "observation.state.arm"],
    )
    if len(df) < 3:
        return None

    act = slice_active(np.stack(df["action.arm"].values).astype(np.float32), mode)
    obs = slice_active(np.stack(df["observation.state.arm"].values).astype(np.float32), mode)

    def _metrics(arr):
        d1 = np.abs(np.diff(arr, axis=0))
        d2 = np.diff(arr, n=2, axis=0)
        max_j = float(d1.max())
        p99 = float(np.percentile(d1, 99))
        p999 = float(np.percentile(d1, 99.9))
        jerk_rms = float(np.sqrt((d2 ** 2).mean()))
        counts = {f"n_gt_{thr}": int((d1 > thr).sum()) for thr in thresholds}
        return {"max": max_j, "p99": p99, "p999": p999, "jerk_rms": jerk_rms, **counts}

    m_act = _metrics(act)
    m_obs = _metrics(obs)
    return {
        "T": len(df),
        **{f"act_{k}": v for k, v in m_act.items()},
        **{f"obs_{k}": v for k, v in m_obs.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/path/to/raw_franka_data/pick_up_the_bread")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.15])
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_csv = Path(args.out_csv) if args.out_csv else data_dir / "noisy_report.csv"

    mode = detect_mode(data_dir)
    print(f"[MODE] {mode}")

    rows = []
    ep_dirs = sorted(data_dir.glob("episode_*"))
    for ep in ep_dirs:
        ts = ep / "timeseries.parquet"
        if not ts.exists():
            continue
        m = score_episode(ts, mode, args.thresholds)
        if m is None:
            continue
        m["ep"] = ep.name
        m["failed"] = ep.name.endswith("_f")
        rows.append(m)

    if not rows:
        print("[WARN] no episodes scored")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("act_jerk_rms", ascending=False).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}  ({len(df)} episodes)")

    print(f"\n[TOP {args.top_k}] noisiest by act_jerk_rms:")
    cols = ["ep", "T", "failed", "act_jerk_rms", "act_max", "act_p99", "act_p999"] + [
        f"act_n_gt_{t}" for t in args.thresholds
    ]
    print(df[cols].head(args.top_k).to_string(index=False))

    print("\n[DIST] act_jerk_rms percentiles:")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  p{p:>3}: {np.percentile(df['act_jerk_rms'], p):.6f}")

    for thr in args.thresholds:
        col = f"act_n_gt_{thr}"
        n_ep = int((df[col] > 0).sum())
        print(f"[COUNT] episodes with any action.arm |diff|>{thr}: {n_ep}/{len(df)}")


if __name__ == "__main__":
    main()
