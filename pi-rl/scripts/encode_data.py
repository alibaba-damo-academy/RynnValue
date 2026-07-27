#!/usr/bin/env python3
# adapted from openpi
# -*- coding: utf-8 -*-

"""
Stage 2: offline alignment + video encoding

Pipeline:
  1. Read per-out_key parquet files under raw_data/
  2. Hold-last alignment to generate evenly spaced frames
  3. Write timeseries.parquet (numeric columns)
  4. Copy aligned images to raw_images/ and encode them into mp4
  5. Apply compose rules (optional)
  6. Clean up raw_data/
"""

import os
import sys
import glob
import json
import time
import shutil
import logging
import argparse
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import av
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add src/ to sys.path so that `import yaml_config_loader` works
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


# ===========================================================================
# Hold-last lookup
# ===========================================================================

def hold_last_lookup_numeric(
    timestamps: np.ndarray,
    values: np.ndarray,
    frame_times: np.ndarray,
) -> np.ndarray:
    """
    Hold-last alignment for numeric data.

    :param timestamps: original timestamps, sorted (N,)
    :param values: original numeric data (N, D)
    :param frame_times: target frame times (M,)
    :return: aligned values (M, D)
    """
    # searchsorted finds the insertion position of each frame_time in timestamps
    # side='right' makes idx[i] the first position > frame_times[i]
    idx = np.searchsorted(timestamps, frame_times, side="right") - 1
    # If frame_time < first record, fill with the first record
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return values[idx]


def hold_last_lookup_paths(
    timestamps: np.ndarray,
    paths: List[str],
    frame_times: np.ndarray,
) -> List[str]:
    """
    Hold-last alignment for image paths.

    :param timestamps: original timestamps (N,)
    :param paths: original image path list (N,)
    :param frame_times: target frame times (M,)
    :return: aligned image path list (M,)
    """
    idx = np.searchsorted(timestamps, frame_times, side="right") - 1
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return [paths[i] for i in idx]


# ===========================================================================
# ANSI bright colors
# ===========================================================================
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_RESET  = "\033[0m"


def _detect_data_gaps(
    all_data: Dict[str, pd.DataFrame],
    t_start: float,
    t_end: float,
    fps: int,
    logger: logging.Logger,
    gap_threshold: float = 1.0,
):
    """Detect data gaps within the alignment window for each key and print bright warnings"""
    duration = t_end - t_start
    has_issue = False

    for out_key, df in all_data.items():
        ts = df["timestamp"].values
        # Only look at data within the alignment window
        mask = (ts >= t_start) & (ts <= t_end)
        ts_in = ts[mask]

        if len(ts_in) == 0:
            logger.warning(
                f"{_RED}[Data gap] \"{out_key}\" "
                f"has no data within alignment window [{t_start:.1f}s, {t_end:.1f}s]!{_RESET}"
            )
            has_issue = True
            continue

        actual_hz = len(ts_in) / duration if duration > 0 else 0

        # Detect gaps in consecutive intervals
        gap_warned = False
        if len(ts_in) >= 2:
            diffs = np.diff(ts_in)
            gap_mask = diffs > gap_threshold
            if np.any(gap_mask):
                gap_indices = np.where(gap_mask)[0]
                max_gap = float(np.max(diffs))
                max_gap_idx = int(np.argmax(diffs))
                gap_start_rel = ts_in[max_gap_idx] - t_start
                gap_end_rel = ts_in[max_gap_idx + 1] - t_start
                n_filled = int(max_gap * fps)

                logger.warning(
                    f"{_YELLOW}[Data gap] \"{out_key}\" "
                    f"max gap {max_gap:.2f}s "
                    f"(relative time {gap_start_rel:.1f}s~{gap_end_rel:.1f}s), "
                    f"{n_filled} frames hold-last filled, "
                    f"{len(gap_indices)} gaps > {gap_threshold}s in total, "
                    f"actual Hz={actual_hz:.1f}{_RESET}"
                )
                has_issue = True
                gap_warned = True

        # Overall frequency too low (below 50% of target fps)
        if not gap_warned and actual_hz < fps * 0.5:
            logger.warning(
                f"{_YELLOW}[Low frequency] \"{out_key}\" "
                f"actual Hz={actual_hz:.1f}, target fps={fps}, "
                f"many frames will be duplicated by hold-last filling{_RESET}"
            )
            has_issue = True

    if not has_issue:
        logger.info("  Data integrity check passed, no significant gaps")


# ===========================================================================
# Alignment
# ===========================================================================

def resolve_image_path(
    episode_dir: str,
    out_key: str,
    recorded_path: str,
) -> Optional[str]:
    """Resolve an image path, tolerating old manifests after dataset moves/renames."""
    if os.path.exists(recorded_path):
        return recorded_path

    episode_relative = os.path.join(episode_dir, recorded_path)
    if os.path.exists(episode_relative):
        return episode_relative

    safe_name = out_key.replace("/", "_").replace(".", "_")
    packaged_path = os.path.join(
        episode_dir,
        "raw_data",
        safe_name,
        os.path.basename(recorded_path),
    )
    return packaged_path if os.path.exists(packaged_path) else None

def align_episode(
    episode_dir: str,
    fps: int,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    Perform offline hold-last alignment on an episode's raw_data/.

    Produces:
      - timeseries.parquet (aligned numeric data)
      - raw_images/<cam_key>/ (aligned image sequences for later encoding)
    """
    if logger is None:
        logger = logging.getLogger("align")

    raw_dir = os.path.join(episode_dir, "raw_data")
    if not os.path.isdir(raw_dir):
        logger.warning(f"[Skip] raw_data not found: {episode_dir}")
        return False, "no_raw_data"

    # ---------------------------------------------------------------
    # 1. Read all per-out_key parquet files
    # ---------------------------------------------------------------
    pq_files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    if not pq_files:
        logger.warning(f"[Skip] No parquet files under raw_data/: {episode_dir}")
        return False, "no_parquet"

    numeric_data: Dict[str, pd.DataFrame] = {}  # out_key → df
    image_data: Dict[str, pd.DataFrame] = {}    # out_key → df

    logger.info(f"[1/5] Reading raw_data: {len(pq_files)} parquet files...")
    for pf in pq_files:
        out_key = os.path.basename(pf).replace(".parquet", "")
        df = pd.read_parquet(pf)
        if "timestamp" not in df.columns:
            logger.warning(f"[Skip] {pf} is missing the timestamp column")
            continue
        df = df.sort_values("timestamp").reset_index(drop=True)
        if "image_path" in df.columns:
            image_data[out_key] = df
        else:
            numeric_data[out_key] = df

    all_data = {**numeric_data, **image_data}
    if not all_data:
        logger.warning(f"[Skip] No valid data: {episode_dir}")
        return False, "no_valid_data"

    # ---------------------------------------------------------------
    # 2. Determine the alignment time range
    # ---------------------------------------------------------------
    t_mins = [df["timestamp"].iloc[0] for df in all_data.values()]
    t_maxs = [df["timestamp"].iloc[-1] for df in all_data.values()]

    t_start = max(t_mins)   # earliest time at which all topics have data
    t_end = min(t_maxs)     # latest time at which all topics have data

    if t_end <= t_start:
        logger.warning(f"[Skip] Invalid alignment time range: t_start={t_start}, t_end={t_end}")
        return False, "invalid_time_range"

    n_frames = int((t_end - t_start) * fps)
    if n_frames <= 0:
        logger.warning(f"[Skip] Frame count is 0: duration={t_end - t_start:.3f}s, fps={fps}")
        return False, "zero_frames"

    frame_times = np.array([t_start + i / fps for i in range(n_frames)])
    logger.info(
        f"Alignment: {len(all_data)} keys, "
        f"t_start={t_start:.3f}, t_end={t_end:.3f}, "
        f"duration={t_end - t_start:.1f}s, n_frames={n_frames}"
    )

    # ---------------------------------------------------------------
    # 2.5 Detect data gaps
    # ---------------------------------------------------------------
    _detect_data_gaps(all_data, t_start, t_end, fps, logger)

    # ---------------------------------------------------------------
    # 3. Hold-last alignment — numeric data
    # ---------------------------------------------------------------
    aligned_records = {
        "frame_index": list(range(n_frames)),
        "timestamp": [i / fps for i in range(n_frames)],  # evenly spaced time axis
    }

    logger.info(f"[2/5] Aligning numeric data: {len(numeric_data)} keys...")
    for out_key, df in numeric_data.items():
        ts = df["timestamp"].values
        val_cols = [c for c in df.columns if c != "timestamp"]

        if "raw_bytes" in val_cols:
            idx = np.searchsorted(ts, frame_times, side="right") - 1
            idx = np.clip(idx, 0, len(ts) - 1)
            aligned_records[out_key] = [df["raw_bytes"].iloc[i] for i in idx]
        elif "values" in val_cols:
            values = np.array(df["values"].tolist(), dtype=np.float64)
            aligned = hold_last_lookup_numeric(ts, values, frame_times)
            aligned_records[out_key] = aligned.tolist()
        else:
            values = df[val_cols].values
            aligned = hold_last_lookup_numeric(ts, values, frame_times)
            aligned_records[out_key] = aligned.tolist()

    # ---------------------------------------------------------------
    # 4. Hold-last alignment — image data → raw_images/
    # ---------------------------------------------------------------
    if image_data:
        logger.info(f"[3/5] Aligning image data: {len(image_data)} cameras...")
    for out_key, df in image_data.items():
        ts = df["timestamp"].values
        paths = df["image_path"].tolist()

        aligned_paths = hold_last_lookup_paths(ts, paths, frame_times)

        dst_dir = os.path.join(episode_dir, "raw_images", out_key)
        os.makedirs(dst_dir, exist_ok=True)

        missing_count = 0
        relocated_count = 0
        missing_examples = []
        for fi, recorded_path in enumerate(aligned_paths):
            src_path = resolve_image_path(episode_dir, out_key, recorded_path)
            if src_path is None:
                missing_count += 1
                if len(missing_examples) < 3:
                    missing_examples.append(recorded_path)
                continue
            if src_path != recorded_path:
                relocated_count += 1
            _, ext = os.path.splitext(src_path)
            if not ext:
                ext = ".jpg"
            dst_path = os.path.join(dst_dir, f"{fi:06d}{ext}")
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                logger.error(f"Copy failed: {src_path} → {dst_path}: {e}")
                missing_count += 1

        if relocated_count:
            logger.info(
                f'[Path compat] "{out_key}" {relocated_count}/{n_frames} frames '
                "relocated from the current episode's raw_data"
            )

        if missing_count > 0:
            reason = (
                f'Camera "{out_key}" is missing {missing_count}/{n_frames} frames; '
                f"example paths: {missing_examples}"
            )
            logger.error(f"{_RED}[Missing images] {reason}{_RESET}")
            return False, reason

    # ---------------------------------------------------------------
    # 5. Apply compose rules (concatenate before writing to disk to avoid a second read/write pass)
    # ---------------------------------------------------------------
    compose_rules = []
    meta_path = os.path.join(episode_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        compose_rules = meta.get("compose", [])

    if compose_rules:
        logger.info(f"[4/5] Applying compose rules: {len(compose_rules)} rules...")
        for rule in compose_rules:
            out_key = rule["out_key"] if isinstance(rule, dict) else rule.out_key
            from_keys = (rule.get("from", rule.get("from_keys", []))
                         if isinstance(rule, dict) else rule.from_keys)
            from_cols = [k for k in from_keys if k in aligned_records]
            if not from_cols:
                continue
            composed = []
            for fi in range(n_frames):
                merged = []
                for c in from_cols:
                    val = aligned_records[c][fi]
                    if isinstance(val, np.ndarray):
                        merged.extend(val.tolist())
                    elif isinstance(val, (list, tuple)):
                        merged.extend(val)
                    elif isinstance(val, (int, float)):
                        merged.append(val)
                composed.append(merged)
            aligned_records[out_key] = composed
            logger.info(f"  Compose: {out_key} = concat({from_cols}) → dim={len(composed[0]) if composed else 0}")
    else:
        logger.info("[4/5] No compose rules, skipping")

    # ---------------------------------------------------------------
    # 6. Write timeseries.parquet (single write)
    # ---------------------------------------------------------------
    logger.info(f"[5/5] Writing timeseries.parquet...")
    ts_path = os.path.join(episode_dir, "timeseries.parquet")
    ts_df = pd.DataFrame(aligned_records)
    img_cols = [c for c in ts_df.columns if c.startswith("observation.images.")]
    ts_df = ts_df.drop(columns=img_cols, errors="ignore")
    table = pa.Table.from_pandas(ts_df)
    tmp_ts_path = f"{ts_path}.tmp"
    pq.write_table(table, tmp_ts_path)
    os.replace(tmp_ts_path, ts_path)
    logger.info(f"  timeseries: {len(ts_df)} frames × {len(ts_df.columns)} columns")

    return True, "ok"


# ===========================================================================
# Video encoding (kept compatible with the original encode_data.py)
# ===========================================================================

def encode_video_frames(
    image_paths,
    output_path,
    fps,
    codec="libx264",
    pix_fmt="yuv420p",
    crf=23,
    g=30,
    preset: str = "ultrafast",
    show_progress: bool = True,
    encode_logger: logging.Logger = None,
):
    if not image_paths:
        raise ValueError("Image list is empty")

    _log = encode_logger or logging.getLogger("encode_video")
    total_frames = len(image_paths)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with Image.open(image_paths[0]) as img:
        width, height = img.size

    video_options = {k: str(v) for k, v in {
        "g": g,
        "crf": crf,
        "preset": preset,
        "tune": "zerolatency",
    }.items() if v is not None}

    t_start = time.time()
    _last_log = t_start
    _LOG_INTERVAL = 3.0  # print progress every 3 seconds

    with av.open(output_path, mode="w") as output:
        st = output.add_stream(codec, int(fps), options=video_options)
        st.pix_fmt = pix_fmt
        st.width = width
        st.height = height

        for i, path in enumerate(image_paths, 1):
            with Image.open(path) as img:
                frame = av.VideoFrame.from_image(img)
            for packet in st.encode(frame):
                output.mux(packet)

            # Print encoding progress periodically
            if show_progress:
                now = time.time()
                if now - _last_log >= _LOG_INTERVAL or i == total_frames:
                    elapsed = now - t_start
                    speed = i / elapsed if elapsed > 0 else 0
                    remaining = total_frames - i
                    eta = remaining / speed if speed > 0 else 0
                    _log.info(f"Encoding progress: {i}/{total_frames} frames ({i*100//total_frames}%), {remaining} frames left, speed {speed:.1f}fps, ETA {eta:.1f}s")
                    _last_log = now

        for packet in st.encode():
            output.mux(packet)

    # Encoding summary
    t_total = time.time() - t_start
    file_size_mb = os.path.getsize(output_path) / 1048576
    avg_speed = total_frames / t_total if t_total > 0 else 0
    _log.info(f"Encoding done: {width}×{height} {fps}fps {total_frames} frames, took {t_total:.1f}s ({avg_speed:.1f}fps), output {file_size_mb:.2f}MB → {os.path.basename(output_path)}")


def list_images_sorted(img_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(img_dir, e)))
    return sorted(paths)


def count_video_frames(video_path: str) -> int:
    """Count by actually decoding; cannot rely on stream.frames which may be 0 for some MP4s."""
    with av.open(video_path) as container:
        return sum(1 for _ in container.decode(video=0))


def zip_successful_episodes(
    root: str,
    episode_dirs: List[str],
    report_path: str,
    zip_path: str,
):
    """Export only successfully verified episodes, then atomically replace the official ZIP."""
    root = os.path.abspath(root)
    base = os.path.basename(root.rstrip("/"))
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    tmp_zip = f"{zip_path}.tmp"
    if os.path.exists(tmp_zip):
        os.remove(tmp_zip)

    all_files = []
    for episode_dir in episode_dirs:
        for current_root, dirs, files in os.walk(episode_dir):
            dirs[:] = [d for d in dirs if d not in ("raw_images", "raw_data")]
            for fn in files:
                if fn.endswith(".tmp") or ".tmp." in fn:
                    continue
                all_files.append(os.path.join(current_root, fn))
    all_files.append(report_path)

    try:
        with zipfile.ZipFile(
            tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            for full in tqdm(
                all_files,
                desc=f"Compressing {base}.zip",
                unit="file",
                dynamic_ncols=True,
            ):
                rel = os.path.relpath(full, root)
                zf.write(full, arcname=os.path.join(base, rel))
        os.replace(tmp_zip, zip_path)
    except Exception:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
        raise


def _failure(reason: str, solution: str) -> Dict[str, str]:
    return {"reason": reason, "solution": solution}


# ===========================================================================
# Episode processing: align → encode → cleanup
# ===========================================================================

def process_episode(
    episode_dir, fps, codec, pix_fmt, crf, g, preset,
    show_progress, logger,
):
    """Process a single episode: align → encode → cleanup

    All required information (compose rules, fps, etc.) is read from the
    episode's metadata.json; no external config file is needed. The fps
    argument is only a fallback when metadata.json does not exist.
    """
    raw_dir = os.path.join(episode_dir, "raw_data")
    raw_images_dir = os.path.join(episode_dir, "raw_images")

    # Try to read the capture-time target_fps from metadata.json
    meta_path = os.path.join(episode_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        saved_fps = meta.get("target_fps")
        if saved_fps and saved_fps > 0:
            fps = int(saved_fps)
            logger.info(f"Using fps={fps} from metadata.json")

    # Step 1: alignment (if raw_data/ exists)
    if os.path.isdir(raw_dir):
        logger.info(f"=== Step 1/3: Alignment ===")
        ok, reason = align_episode(episode_dir, fps, logger=logger)
        if not ok:
            return False, _failure(
                f"Data alignment failed: {reason}",
                "Check raw_data and image paths, restore missing data, then re-run encoding",
            )

    # Step 2: encode videos (if raw_images/ exists)
    if os.path.isdir(raw_images_dir):
        cam_dirs = sorted([
            d for d in glob.glob(os.path.join(raw_images_dir, "observation.images.*"))
            if os.path.isdir(d)
        ])

        if not cam_dirs:
            return False, _failure(
                "No observation.images.* camera directories under raw_images",
                "Check camera directory naming, or restore original images from backup and re-run encoding",
            )

        ts_path = os.path.join(episode_dir, "timeseries.parquet")
        if not os.path.exists(ts_path):
            return False, _failure(
                "Missing timeseries.parquet",
                "Restore raw_data, then re-run data alignment and encoding",
            )
        expected_frames = len(pd.read_parquet(ts_path))
        staged_videos = []
        logger.info(f"=== Step 2/3: Encoding videos ({len(cam_dirs)} cameras) ===")
        for ci, d in enumerate(cam_dirs, 1):
            key = os.path.basename(d)
            imgs = list_images_sorted(d)
            if not imgs:
                for staged, _, _ in staged_videos:
                    if os.path.exists(staged):
                        os.remove(staged)
                return False, _failure(
                    f"Image directory for camera {key} is empty",
                    "Restore this camera's original images, then re-run encoding",
                )
            if len(imgs) != expected_frames:
                for staged, _, _ in staged_videos:
                    if os.path.exists(staged):
                        os.remove(staged)
                return False, _failure(
                    f"Camera {key} image count={len(imgs)}, timeseries frame count={expected_frames}",
                    "Check and restore missing images so the image count matches the timeseries frame count, then retry",
                )

            out_mp4 = os.path.join(episode_dir, f"{key}.mp4")
            tmp_mp4 = os.path.join(episode_dir, f".{key}.tmp.mp4")
            if os.path.exists(tmp_mp4):
                os.remove(tmp_mp4)
            logger.info(f"  [{ci}/{len(cam_dirs)}] {key}: {len(imgs)} frames → mp4")
            try:
                encode_video_frames(
                    image_paths=imgs,
                    output_path=tmp_mp4,
                    fps=fps,
                    codec=codec,
                    pix_fmt=pix_fmt,
                    crf=crf,
                    g=g,
                    preset=preset,
                    show_progress=show_progress,
                    encode_logger=logger,
                )
                video_frames = count_video_frames(tmp_mp4)
            except Exception as e:
                if os.path.exists(tmp_mp4):
                    os.remove(tmp_mp4)
                for staged, _, _ in staged_videos:
                    if os.path.exists(staged):
                        os.remove(staged)
                return False, _failure(
                    f"Camera {key} video encoding or decode verification failed: {e}",
                    "Check whether this camera has corrupted images; original data is kept, retry after fixing",
                )
            if video_frames != expected_frames:
                os.remove(tmp_mp4)
                for staged, _, _ in staged_videos:
                    if os.path.exists(staged):
                        os.remove(staged)
                return False, _failure(
                    f"Camera {key} video frame count={video_frames}, timeseries frame count={expected_frames}",
                    "Check image integrity and encoder output, then re-run encoding",
                )
            staged_videos.append((tmp_mp4, out_mp4, key))

        # Only replace the official MP4s after all cameras pass verification;
        # on failure, old videos and original images are both kept.
        for tmp_mp4, out_mp4, _ in staged_videos:
            os.replace(tmp_mp4, out_mp4)

    # Episodes already encoded without original images must also be verified;
    # they cannot be counted as successful directly.
    ok, reason = _verify_frame_alignment(episode_dir, logger)
    if not ok:
        return False, _failure(
            reason,
            "Restore original images or raw_data, then re-run encoding",
        )

    # Step 5: update metadata
    _update_metadata(episode_dir, fps, logger)

    logger.info(f"[Done] {episode_dir}")
    return True, {}


def _verify_frame_alignment(episode_dir: str, logger):
    """Verify frame-count consistency between timeseries.parquet and all MP4s"""
    ts_path = os.path.join(episode_dir, "timeseries.parquet")
    if not os.path.exists(ts_path):
        return False, "Missing timeseries.parquet"

    ts_frames = len(pd.read_parquet(ts_path))

    mp4_files = sorted(glob.glob(os.path.join(episode_dir, "observation.images.*.mp4")))
    if not mp4_files:
        return False, "No observation.images.*.mp4 videos were generated"

    all_match = True
    for mp4_path in mp4_files:
        key = os.path.basename(mp4_path)
        try:
            with av.open(mp4_path) as container:
                video_frames = sum(1 for _ in container.decode(video=0))
        except Exception as e:
            return False, f"Cannot read video {key}: {e}"

        if video_frames != ts_frames:
            logger.warning(
                f"{_RED}[Frame count mismatch] {key}: "
                f"MP4={video_frames} frames, timeseries={ts_frames} frames{_RESET}"
            )
            all_match = False
        else:
            logger.info(f"  ✓ {key}: {video_frames} frames = timeseries {ts_frames} frames")

    if all_match:
        logger.info(f"  Frame consistency check passed: all {len(mp4_files)} videos aligned with timeseries ({ts_frames} frames)")
        return True, ""
    return False, "One or more videos have a frame count inconsistent with timeseries"


def _update_metadata(episode_dir: str, fps: int, logger):
    """Update result fields after encoding while keeping the original metadata needed for re-encoding."""
    meta_path = os.path.join(episode_dir, "metadata.json")
    old_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            old_meta = json.load(f)

    # Count total_frames
    total_frames = 0
    ts_path = os.path.join(episode_dir, "timeseries.parquet")
    if os.path.exists(ts_path):
        df = pd.read_parquet(ts_path)
        total_frames = len(df)

    old_meta.update({
        "total_frames": total_frames,
        "fps": fps,
        "encode_status": "success",
    })

    tmp_path = f"{meta_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(old_meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, meta_path)


def _process_one_worker(ep, fps, codec, pix_fmt, crf, g, preset, show_progress):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("encode_data.worker")
    try:
        ok, result = process_episode(
            episode_dir=ep, fps=fps, codec=codec, pix_fmt=pix_fmt,
            crf=crf, g=g, preset=preset, show_progress=show_progress,
            logger=logger,
        )
        return ep, ok, result
    except Exception as e:
        logger.exception("Uncaught exception while processing episode")
        return ep, False, _failure(
            f"Uncaught exception: {type(e).__name__}: {e}",
            "Check the full log to locate the exception; original data is kept, retry after fixing",
        )


def _write_report(report_path: str, report: dict):
    tmp_path = f"{report_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, report_path)


def _cleanup_successful_episode(episode_dir: str, logger):
    for name in ("raw_images", "raw_data"):
        path = os.path.join(episode_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
            logger.info(f"Cleaned up {path}")


def _log_summary(
    logger, total, encoded, exported, failed, zip_path, export_error=None
):
    logger.info("\n" + "=" * 64)
    logger.info("Encode/export results")
    logger.info(f"Encoded successfully ({len(encoded)}):")
    for name in encoded:
        logger.info(f"  ✓ {name}")

    logger.info(f"Exported successfully ({len(exported)}):")
    for name in exported:
        logger.info(f"  ✓ {name}")
    if zip_path:
        logger.info(f"  ZIP: {zip_path}")
    elif export_error:
        logger.error(f"  ZIP export failed: {export_error}")

    logger.info(f"Failed to encode ({len(failed)}):")
    for item in failed:
        logger.error(f"  ✗ {item['episode']}")
        logger.error(f"    Reason: {item['reason']}")
        logger.error(f"    Solution: {item['solution']}")

    logger.info(
        f"Summary: total {total}, encoded {len(encoded)}, "
        f"exported {len(exported)}, failed {len(failed)}"
    )
    logger.info("=" * 64)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Stage 2: offline alignment + video encoding")
    ap.add_argument("-r", "--root", required=True, help="Task directory, e.g.: output/tianji_sharpa_teleop")
    ap.add_argument("--fps", type=int, default=30,
                    help="Target frame rate (target_fps in metadata.json takes precedence if present)")

    ap.add_argument("--codec", default="libx264")
    ap.add_argument("--pix-fmt", default="yuv420p")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--g", type=int, default=30)
    ap.add_argument("--preset", default="ultrafast")

    ap.add_argument("--jobs", type=int, default=1, help="Number of processes for parallel episode processing")
    ap.add_argument("--show-progress", action="store_true")
    ap.add_argument("--zip-task", action="store_true",
                    help="Export successfully encoded episodes as a zip")
    ap.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up raw_images/raw_data after successful encoding and export; kept by default",
    )

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("encode_data")

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise FileNotFoundError(root)

    episodes = sorted(
        ep for ep in glob.glob(os.path.join(root, "episode_*"))
        if os.path.isdir(ep)
    )
    if not episodes:
        logger.warning(f"[Warning] No episode_* found under task directory: {root}")
        return

    encoded_dirs = []
    failed = []
    show_progress = args.show_progress
    if args.jobs > 1 and len(episodes) > 1:
        logger.warning("[Note] Encoding progress logs may interleave when processing multiple episodes in parallel.")

    if args.jobs <= 1:
        for ei, ep in enumerate(episodes, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing episode {ei}/{len(episodes)}: {os.path.basename(ep)}")
            logger.info(f"{'='*60}")
            ep, ok, result = _process_one_worker(
                ep, args.fps, args.codec, args.pix_fmt,
                args.crf, args.g, args.preset, show_progress,
            )
            if ok:
                encoded_dirs.append(ep)
            else:
                failed.append({"episode": os.path.basename(ep), **result})
    else:
        logger.info(f"Starting parallel processing: jobs={args.jobs}, episodes={len(episodes)}")
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [
                ex.submit(
                    _process_one_worker,
                    ep, args.fps, args.codec, args.pix_fmt,
                    args.crf, args.g, args.preset, show_progress,
                )
                for ep in episodes
            ]
            for fut in as_completed(futs):
                ep, ok, result = fut.result()
                if ok:
                    encoded_dirs.append(ep)
                else:
                    failed.append({"episode": os.path.basename(ep), **result})

    encoded_dirs.sort()
    failed.sort(key=lambda item: item["episode"])
    encoded = [os.path.basename(ep) for ep in encoded_dirs]
    exported = []
    zip_path = None
    export_error = None
    report_path = os.path.join(root, "encode_report.json")
    report = {
        "total": len(episodes),
        "encoded": encoded,
        "exported": exported,
        "failed": failed,
        "zip_path": zip_path,
        "export_error": None,
    }
    _write_report(report_path, report)

    if args.zip_task and encoded_dirs:
        candidate_zip = os.path.join(
            os.path.dirname(root), f"{os.path.basename(root)}.zip"
        )
        try:
            # The report also goes into the ZIP; exported is only confirmed
            # after the ZIP atomic replace succeeds.
            report["exported"] = encoded
            report["zip_path"] = candidate_zip
            _write_report(report_path, report)
            zip_successful_episodes(
                root, encoded_dirs, report_path, candidate_zip
            )
            exported = encoded.copy()
            zip_path = candidate_zip
            logger.info(f"[Done] Archive created: {zip_path}")
        except Exception as e:
            logger.exception(f"[Failed] Compression failed: {e}")
            export_error = f"{type(e).__name__}: {e}"
            report["exported"] = []
            report["zip_path"] = None
            report["export_error"] = export_error
            _write_report(report_path, report)

    if args.cleanup:
        cleanup_dirs = encoded_dirs if not args.zip_task else [
            ep for ep in encoded_dirs if os.path.basename(ep) in exported
        ]
        for ep in cleanup_dirs:
            _cleanup_successful_episode(ep, logger)

    _log_summary(
        logger,
        total=len(episodes),
        encoded=encoded,
        exported=exported,
        failed=failed,
        zip_path=zip_path,
        export_error=export_error,
    )
    logger.info(f"Result report: {report_path}")


if __name__ == "__main__":
    main()