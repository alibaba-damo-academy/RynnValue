#!/usr/bin/env python3
# adapted from openpi
"""Render raw Franka demo episodes as grid videos (2x2 camera layout).

Reads the raw per-frame JPEG layout produced by the Franka collection stack
(see scripts/convert_franka_data_to_lerobot.py for the format) and writes one
MP4 (H.264) per episode showing all four camera views side-by-side.

Expected layout::

    <dataset_dir>/
      episode_XXXXXX[/_f]/
        metadata.json
        timeseries.parquet
        raw_images/
          observation.images.left_side/000000.jpg ...
          observation.images.left_wrist/000000.jpg ...
          observation.images.right_side/000000.jpg ...
          observation.images.right_wrist/000000.jpg ...

Output defaults to scripts/<dataset_dir.name>/.

Usage::

    python scripts/render_raw_franka_videos.py
    python scripts/render_raw_franka_videos.py --episodes 3 5 10
    python scripts/render_raw_franka_videos.py --dataset_dir /path/to/dataset
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


CAMERA_KEYS = [
    "observation.images.left_side",
    "observation.images.left_wrist",
    "observation.images.right_side",
    "observation.images.right_wrist",
]

CAMERA_LABELS = [
    "Left Side",
    "Left Wrist",
    "Right Side",
    "Right Wrist",
]

SCRIPT_DIR = Path(__file__).resolve().parent


def make_grid(frames: list[np.ndarray], labels: list[str], pad: int = 2) -> np.ndarray:
    """Combine 4 camera frames into a 2x2 grid with labels."""
    h, w = frames[0].shape[:2]

    label_h = 24
    labeled = []
    for frame, label in zip(frames, labels):
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        bar = np.zeros((label_h, w, 3), dtype=np.uint8)
        cv2.putText(bar, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        labeled.append(np.vstack([bar, frame]))

    top = np.hstack([labeled[0], np.full((h + label_h, pad, 3), 128, dtype=np.uint8), labeled[1]])
    bot = np.hstack([labeled[2], np.full((h + label_h, pad, 3), 128, dtype=np.uint8), labeled[3]])
    grid = np.vstack([top, np.full((pad, top.shape[1], 3), 128, dtype=np.uint8), bot])
    return grid


def load_episode_frames(ep_dir: Path) -> list[dict]:
    """Load all frames for one raw episode directory.

    Returns a list of dicts mapping camera_key -> BGR ndarray, indexed by frame.
    Frames missing any camera image are skipped with a warning.
    """
    img_root = ep_dir / "raw_images"
    cam_dirs = {k: img_root / k for k in CAMERA_KEYS}
    for k, d in cam_dirs.items():
        if not d.is_dir():
            print(f"  [SKIP] {ep_dir.name}: missing camera dir {d}")
            return []

    frame_indices = sorted(
        int(p.stem) for p in cam_dirs[CAMERA_KEYS[0]].glob("*.jpg") if p.stem.isdigit()
    )
    if not frame_indices:
        print(f"  [SKIP] {ep_dir.name}: no jpg frames found")
        return []

    frames = []
    for fi in frame_indices:
        images = {}
        ok = True
        for key in CAMERA_KEYS:
            img_path = cam_dirs[key] / f"{fi:06d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [WARN] {ep_dir.name}: failed to read {img_path}")
                ok = False
                break
            images[key] = img
        if ok:
            frames.append(images)

    return frames


def _encode_to_tmpfile(frames, camera_keys, labels, grid_w, grid_h, fps) -> Path | None:
    """Encode frames to a local temp mp4 via ffmpeg pipe."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="render_")
    os.close(tmp_fd)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{grid_w}x{grid_h}", "-pix_fmt", "bgr24",
        "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        tmp_path,
    ]

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("  [ERROR] ffmpeg not found on PATH.")
        os.unlink(tmp_path)
        return None

    for frame_dict in frames:
        cams = [frame_dict[k] for k in camera_keys]
        grid = make_grid(cams, labels)
        gh, gw = grid.shape[:2]
        if gh != grid_h or gw != grid_w:
            padded = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            padded[:gh, :gw] = grid
            grid = padded
        proc.stdin.write(grid.tobytes())

    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        os.unlink(tmp_path)
        return None

    return Path(tmp_path)


def render_episode_video(frames: list[dict], output_path: Path, fps: int = 30) -> bool:
    """Write a single episode grid video via a local temp file, then copy out.

    Using a local temp file avoids NFS write corruption (moov atom not flushed).
    """
    if not frames:
        return False

    h, w = frames[0][CAMERA_KEYS[0]].shape[:2]
    grid_w = w * 2 + 2
    grid_h = (h + 24) * 2 + 2
    if grid_w % 2 != 0:
        grid_w += 1
    if grid_h % 2 != 0:
        grid_h += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _encode_to_tmpfile(frames, CAMERA_KEYS, CAMERA_LABELS, grid_w, grid_h, fps)
    if tmp is None:
        return False

    shutil.copy2(str(tmp), str(output_path))
    tmp.unlink()
    return True


def render_per_camera_video(frames: list[dict], key: str, output_path: Path, fps: int) -> bool:
    """Optional per-camera video for one camera key."""
    if not frames:
        return False

    h, w = frames[0][key].shape[:2]
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="cam_")
    os.close(tmp_fd)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
        "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        tmp_path,
    ]

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for fd in frames:
            img = fd[key]
            ih, iw = img.shape[:2]
            if ih != h or iw != w:
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                padded[:ih, :iw] = img
                img = padded
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        proc.wait()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_path, str(output_path))
        return True
    except Exception as e:
        print(f"  [WARN] Failed to write per-camera video {output_path}: {e}")
        return False
    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


def discover_episodes(dataset_dir: Path, include_failed: bool) -> list[Path]:
    """Return sorted list of episode_* directories under dataset_dir."""
    eps = []
    for p in sorted(dataset_dir.glob("episode_*")):
        if not p.is_dir():
            continue
        if not include_failed and p.name.endswith("_f"):
            continue
        eps.append(p)
    return eps


def parse_episode_index(name: str) -> int | None:
    """Parse the integer index from an episode dir name like 'episode_000003' or 'episode_000003_f'."""
    stem = name[len("episode_"):] if name.startswith("episode_") else name
    if stem.endswith("_f"):
        stem = stem[:-2]
    try:
        return int(stem)
    except ValueError:
        return None


def read_episode_fps(ep_dir: Path, default_fps: int) -> int:
    """Read fps from metadata.json if present, else fall back to default."""
    meta_path = ep_dir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            fps = int(meta.get("fps", default_fps))
            if fps > 0:
                return fps
        except Exception as e:
            print(f"  [WARN] Could not parse {meta_path}: {e}")
    return default_fps


def main():
    parser = argparse.ArgumentParser(description="Render raw Franka episodes as 2x2 grid videos.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/path/to/raw_franka_data/close_the_drawer",
        help="Path to the raw Franka dataset directory containing episode_* dirs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to scripts/<dataset_dir.name>/.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Specific episode indices to render. If omitted, renders all discovered episodes.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Fallback FPS when metadata.json is missing or invalid.",
    )
    parser.add_argument(
        "--include_failed",
        action="store_true",
        default=False,
        help="Also render episodes whose dir name ends with '_f' (failed runs).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Only render episodes with strictly fewer than this many frames. Disabled if omitted.",
    )
    parser.add_argument(
        "--per_camera",
        action="store_true",
        default=False,
        help="Also render individual per-camera videos under output_dir/per_camera/.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        print(f"[ERROR] dataset_dir does not exist: {dataset_dir}")
        sys.exit(1)

    repo_id = dataset_dir.name
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / repo_id
    output_dir.mkdir(parents=True, exist_ok=True)

    all_eps = discover_episodes(dataset_dir, include_failed=args.include_failed)
    if not all_eps:
        print(f"[ERROR] No episode_* directories found under {dataset_dir}.")
        sys.exit(1)

    if args.episodes is not None:
        wanted = set(args.episodes)
        episodes = [ep for ep in all_eps if parse_episode_index(ep.name) in wanted]
        missing = wanted - {parse_episode_index(ep.name) for ep in episodes}
        if missing:
            print(f"[WARN] Requested episodes not found: {sorted(missing)}")
    else:
        episodes = all_eps

    if not episodes:
        print("[ERROR] No episodes selected.")
        sys.exit(1)

    print(f"Dataset:   {dataset_dir}")
    print(f"Repo ID:   {repo_id}")
    print(f"Output:    {output_dir}")
    print(f"Episodes:  {len(episodes)} (include_failed={args.include_failed})")
    print(f"Default FPS: {args.fps}")
    print()

    ok_count = 0
    fail_count = 0

    for i, ep_dir in enumerate(episodes):
        ep_idx = parse_episode_index(ep_dir.name)
        ep_label = f"{ep_idx:06d}" if ep_idx is not None else ep_dir.name
        print(f"[{i+1}/{len(episodes)}] Rendering {ep_dir.name}...")

        frames = load_episode_frames(ep_dir)
        if not frames:
            fail_count += 1
            continue

        if args.max_frames is not None and len(frames) >= args.max_frames:
            print(f"  [SKIP] {ep_dir.name}: {len(frames)} frames >= max_frames ({args.max_frames})")
            continue

        fps = read_episode_fps(ep_dir, default_fps=args.fps)

        suffix = "_f" if ep_dir.name.endswith("_f") else ""
        grid_path = output_dir / f"episode_{ep_label}{suffix}_grid.mp4"
        if render_episode_video(frames, grid_path, fps=fps):
            print(f"  -> {grid_path}  ({len(frames)} frames @ {fps} fps)")
            ok_count += 1
        else:
            fail_count += 1

        if args.per_camera:
            for key, label in zip(CAMERA_KEYS, CAMERA_LABELS):
                cam_dir = output_dir / "per_camera" / label.lower().replace(" ", "_")
                cam_path = cam_dir / f"episode_{ep_label}{suffix}.mp4"
                render_per_camera_video(frames, key, cam_path, fps=fps)

    print()
    print(f"Done. Rendered {ok_count} videos, {fail_count} failed/skipped.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
