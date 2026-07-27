#!/usr/bin/env python3
# adapted from openpi
"""Preprocess RynnValue "raw" Franka episodes into the "timeseries" layout.

Some pick_up_the_pen source dirs (e.g. pick_up_the_pen_new_1/_new_2) ship the
un-aligned "raw" stage: each channel is a separate parquet sampled at its own
rate (arm cmd ~50Hz, joint states ~54Hz, cameras ~29Hz) plus per-camera JPEG
folders. The LeRobot converter instead expects the aligned "timeseries" stage
(same layout as pick_up_the_pen's main dir):

    <episode>/
      timeseries.parquet          # one row per frame, all channels aligned
      raw_images/
        observation.images.left_side/000000.jpg ...
        observation.images.left_wrist/000000.jpg ...
        observation.images.right_side/000000.jpg ...
        observation.images.right_wrist/000000.jpg ...

This script converts raw -> timeseries by:
  1. Reading every channel's (timestamp, values) / (timestamp, image rows).
  2. Taking the COMMON time span (latest start .. earliest end) so no channel
     is missing data at the edges.
  3. Building a uniform FPS timeline (default 30) over that span.
  4. NEAREST-NEIGHBOR sampling each channel onto the timeline (no interpolation,
     so gripper / quaternion values are never distorted).
  5. Emitting timeseries.parquet + raw_images/<cam>/<frame>.jpg (symlinks by
     default to avoid copying hundreds of thousands of JPEGs back to OSS).

The arm stays 14-dim and gripper 2-dim; the downstream converter's
detect_active_arms decides single/dual-arm and slices accordingly.

Usage:
    python scripts/preprocess_raw_to_timeseries.py \
        --src_dir /path/to/raw_franka_data/pick_up_the_pen_new_1 \
        --out_dir /path/to/raw_franka_data/pick_up_the_pen_new_1_ts

    # quick check on a couple of episodes:
    python scripts/preprocess_raw_to_timeseries.py --src_dir ... --out_dir ... \
        --max_episodes 2 --image_mode copy
"""

import glob
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

# Numeric channels -> raw_data parquet filename. Order matches the main dir's
# timeseries.parquet column order (after frame_index / timestamp).
NUMERIC_CHANNELS = {
    "action.arm": "action.arm.parquet",
    "action.gripper": "action.gripper.parquet",
    "observation.state.arm": "observation.state.arm.parquet",
    "observation.state.gripper": "observation.state.gripper.parquet",
    "observation.state.left_ee_pose": "observation.state.left_ee_pose.parquet",
    "observation.state.left_ee_ftsensor": "observation.state.left_ee_ftsensor.parquet",
    "observation.state.right_ee_pose": "observation.state.right_ee_pose.parquet",
    "observation.state.right_ee_ftsensor": "observation.state.right_ee_ftsensor.parquet",
}

# Camera key -> (channel parquet, raw jpg dir). The converter reads
# raw_images/<camera key>/<frame>.jpg for exactly these four keys.
CAM_KEYS = [
    "observation.images.left_side",
    "observation.images.left_wrist",
    "observation.images.right_side",
    "observation.images.right_wrist",
]


def _cam_parquet(cam_key: str) -> str:
    return f"{cam_key}.parquet"


def _cam_raw_dir(cam_key: str) -> str:
    # "observation.images.left_side" -> "observation_images_left_side"
    suffix = cam_key.split("observation.images.")[1]
    return f"observation_images_{suffix}"


def _read_numeric(path: Path):
    """Return (timestamps (N,), values (N, D)) sorted by timestamp."""
    df = pd.read_parquet(path)
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    vals = np.stack(df["values"].to_numpy()).astype(np.float32)
    order = np.argsort(ts, kind="stable")
    return ts[order], vals[order]


def _read_camera_ts(path: Path):
    """Return (timestamps (N,), row_order (N,)) sorted by timestamp.

    row_order[k] is the original parquet row index (== jpg filename number) for
    the k-th chronologically-ordered frame.
    """
    df = pd.read_parquet(path)
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    order = np.argsort(ts, kind="stable")
    return ts[order], order


def _nearest_indices(src_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """For each target timestamp, index of the nearest src timestamp.

    src_ts must be sorted ascending.
    """
    n = len(src_ts)
    if n == 1:
        return np.zeros(len(target_ts), dtype=np.int64)
    pos = np.searchsorted(src_ts, target_ts)
    pos = np.clip(pos, 1, n - 1)
    left = src_ts[pos - 1]
    right = src_ts[pos]
    choose_left = (target_ts - left) <= (right - target_ts)
    return np.where(choose_left, pos - 1, pos)


def process_episode(src_ep: str, out_ep: str, fps: int, image_mode: str) -> dict:
    src_ep_p = Path(src_ep)
    out_ep_p = Path(out_ep)
    rd = src_ep_p / "raw_data"

    if not rd.is_dir():
        return {"status": "skip", "reason": "no raw_data dir", "ep": src_ep_p.name}

    try:
        # ── read numeric channels ──────────────────────────────────────────
        num_ts, num_vals = {}, {}
        for key, fname in NUMERIC_CHANNELS.items():
            p = rd / fname
            if not p.exists():
                return {"status": "skip", "reason": f"missing {fname}", "ep": src_ep_p.name}
            ts, vals = _read_numeric(p)
            if len(ts) == 0:
                return {"status": "skip", "reason": f"empty {fname}", "ep": src_ep_p.name}
            num_ts[key], num_vals[key] = ts, vals

        # ── read camera channels ───────────────────────────────────────────
        cam_ts, cam_row = {}, {}
        for cam in CAM_KEYS:
            p = rd / _cam_parquet(cam)
            if not p.exists():
                return {"status": "skip", "reason": f"missing {_cam_parquet(cam)}", "ep": src_ep_p.name}
            ts, order = _read_camera_ts(p)
            if len(ts) == 0:
                return {"status": "skip", "reason": f"empty {_cam_parquet(cam)}", "ep": src_ep_p.name}
            cam_ts[cam], cam_row[cam] = ts, order

        # ── common time span ───────────────────────────────────────────────
        starts = [ts[0] for ts in num_ts.values()] + [ts[0] for ts in cam_ts.values()]
        ends = [ts[-1] for ts in num_ts.values()] + [ts[-1] for ts in cam_ts.values()]
        t_start = max(starts)
        t_end = min(ends)
        duration = t_end - t_start
        if duration <= 0:
            return {"status": "skip", "reason": "no overlapping time span", "ep": src_ep_p.name}

        K = int(np.floor(duration * fps)) + 1
        if K < 2:
            return {"status": "skip", "reason": f"too few frames ({K})", "ep": src_ep_p.name}
        target_ts = t_start + np.arange(K, dtype=np.float64) / fps

        # ── resample numeric channels (nearest) ────────────────────────────
        resampled = {}
        for key in NUMERIC_CHANNELS:
            idx = _nearest_indices(num_ts[key], target_ts)
            resampled[key] = num_vals[key][idx]  # (K, D)

        # ── prepare output dirs (idempotent) ───────────────────────────────
        out_ep_p.mkdir(parents=True, exist_ok=True)
        img_root = out_ep_p / "raw_images"
        if img_root.exists():
            shutil.rmtree(img_root)

        # ── materialize per-frame images (nearest jpg) ─────────────────────
        for cam in CAM_KEYS:
            idx = _nearest_indices(cam_ts[cam], target_ts)   # into sorted order
            src_rows = cam_row[cam][idx]                     # original jpg numbers
            cam_out = img_root / cam
            cam_out.mkdir(parents=True, exist_ok=True)
            # Precompute once per camera; avoid per-frame OSS stat over FUSE.
            raw_cam_dir_abs = os.path.abspath(rd / _cam_raw_dir(cam))
            cam_out_str = str(cam_out)
            for k in range(K):
                src_jpg = f"{raw_cam_dir_abs}/{int(src_rows[k]):06d}.jpg"
                dst_jpg = f"{cam_out_str}/{k:06d}.jpg"
                if image_mode == "symlink":
                    os.symlink(src_jpg, dst_jpg)
                else:
                    shutil.copy(src_jpg, dst_jpg)

        # ── assemble timeseries.parquet ────────────────────────────────────
        data = {
            "frame_index": np.arange(K, dtype=np.int64),
            "timestamp": (np.arange(K, dtype=np.float64) / fps),
        }
        for key in NUMERIC_CHANNELS:
            data[key] = list(resampled[key])  # K rows, each a (D,) float32 array
        df_out = pd.DataFrame(data)
        df_out.to_parquet(out_ep_p / "timeseries.parquet", engine="pyarrow", compression="snappy")

        # ── metadata.json (carry task info, mark stage=timeseries) ─────────
        meta = {}
        src_meta = src_ep_p / "metadata.json"
        if src_meta.exists():
            try:
                meta = json.loads(src_meta.read_text())
            except Exception:
                meta = {}
        meta.update({
            "total_frames": int(K),
            "fps": int(fps),
            "timeseries": {"file": "timeseries.parquet"},
            "stage": "timeseries",
            "preprocessed_from": "raw",
        })
        (out_ep_p / "metadata.json").write_text(json.dumps(meta, indent=2))

        return {"status": "ok", "ep": src_ep_p.name, "frames": int(K)}

    except Exception as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}", "ep": src_ep_p.name}


def main(
    src_dir: str,
    *,
    out_dir: str,
    fps: int = 30,
    image_mode: str = "symlink",
    num_workers: int = 8,
    max_episodes: int | None = None,
):
    """Convert a raw-stage Franka dataset dir into the timeseries layout.

    Args:
        src_dir: raw dataset dir containing episode_*/raw_data/.
        out_dir: output dir; episode_*/timeseries.parquet + raw_images/ written here.
        fps: target frame rate for the uniform timeline.
        image_mode: 'symlink' (fast, points at source jpgs) or 'copy'.
        num_workers: parallel episode workers.
        max_episodes: only process the first N episodes (debug).
    """
    if image_mode not in ("symlink", "copy"):
        raise ValueError(f"image_mode must be 'symlink' or 'copy', got {image_mode}")

    src = Path(src_dir).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    if out == src:
        raise ValueError("out_dir must differ from src_dir (refusing in-place overwrite)")
    out.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted(p for p in src.glob("episode_*") if p.is_dir())
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* dirs found in {src}")
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]

    print(f"[preprocess] src        = {src}")
    print(f"[preprocess] out        = {out}")
    print(f"[preprocess] episodes   = {len(episode_dirs)}")
    print(f"[preprocess] fps        = {fps}")
    print(f"[preprocess] image_mode = {image_mode}")
    print(f"[preprocess] num_workers= {num_workers}")
    print()

    jobs = [(str(ep), str(out / ep.name)) for ep in episode_dirs]

    n_ok = n_skip = n_err = 0
    total_frames = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futs = {
            pool.submit(process_episode, s, o, fps, image_mode): Path(s).name
            for s, o in jobs
        }
        for fut in as_completed(futs):
            r = fut.result()
            if r["status"] == "ok":
                n_ok += 1
                total_frames += r.get("frames", 0)
                print(f"[ok]   {r['ep']}  ({r.get('frames', 0)} frames)")
            elif r["status"] == "skip":
                n_skip += 1
                print(f"[skip] {r['ep']}: {r['reason']}")
            else:
                n_err += 1
                print(f"[ERR]  {r['ep']}: {r['reason']}")

    print()
    print("============================================================")
    print(f" Preprocess done: ok={n_ok}  skip={n_skip}  error={n_err}")
    print(f" total frames written = {total_frames}")
    print(f" output dir = {out}")
    print("============================================================")

    if n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    tyro.cli(main)
