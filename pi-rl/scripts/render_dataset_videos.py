#!/usr/bin/env python3
# adapted from openpi
"""Render dataset episodes as grid videos (2x2 camera layout).

Reads LeRobot-format parquet episodes and produces one MP4 (H.264) per episode,
each showing all camera views side-by-side in a grid.

Output defaults to scripts/<repo_id>/ (repo_id = dataset directory name).

Usage:
    python scripts/render_dataset_videos.py
    python scripts/render_dataset_videos.py --episodes 0 1 5 10
    python scripts/render_dataset_videos.py --dataset_dir /path/to/dataset
"""

import argparse
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


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


def decode_image_bytes(raw: bytes) -> np.ndarray:
    """Decode image bytes (JPEG/PNG) to BGR numpy array."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def make_grid(frames: list[np.ndarray], labels: list[str], pad: int = 2) -> np.ndarray:
    """Combine 4 camera frames into a 2x2 grid with labels."""
    h, w = frames[0].shape[:2]

    label_h = 24
    labeled = []
    for frame, label in zip(frames, labels):
        bar = np.zeros((label_h, w, 3), dtype=np.uint8)
        cv2.putText(bar, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        labeled.append(np.vstack([bar, frame]))

    top = np.hstack([labeled[0], np.full((h + label_h, pad, 3), 128, dtype=np.uint8), labeled[1]])
    bot = np.hstack([labeled[2], np.full((h + label_h, pad, 3), 128, dtype=np.uint8), labeled[3]])
    grid = np.vstack([top, np.full((pad, top.shape[1], 3), 128, dtype=np.uint8), bot])
    return grid


def load_episode(data_dir: Path, episode_idx: int, chunk_size: int = 1000) -> list[dict]:
    """Load all frames for one episode from its parquet file.

    Returns list of dicts mapping camera_key -> BGR ndarray.
    """
    chunk = episode_idx // chunk_size
    parquet_path = data_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_idx:06d}.parquet"

    if not parquet_path.exists():
        print(f"  [SKIP] Episode {episode_idx}: file not found at {parquet_path}")
        return []

    table = pq.read_table(str(parquet_path))
    n_frames = table.num_rows

    frames = []
    for i in range(n_frames):
        images = {}
        for key in CAMERA_KEYS:
            col = table.column(key)
            cell = col[i].as_py()
            raw = cell.get("bytes") or cell.get("b")
            if raw is None:
                p = cell.get("path") or cell.get("p")
                if p:
                    img_path = data_dir / p
                    if img_path.exists():
                        images[key] = cv2.imread(str(img_path))
                if key not in images:
                    print(f"  [WARN] Missing image data for {key} at frame {i}, ep {episode_idx}")
                    continue
            else:
                images[key] = decode_image_bytes(raw)
        if len(images) == len(CAMERA_KEYS):
            frames.append(images)

    return frames


def _encode_to_tmpfile(frames, camera_keys, labels, grid_w, grid_h, fps) -> Path | None:
    """Encode frames to a local temp mp4 file. Returns the temp path or None on error."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="render_")
    # Close the fd — ffmpeg will write to this path
    import os
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
        print("  [ERROR] ffmpeg not found.")
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


def render_episode_video(
    frames: list[dict],
    output_path: Path,
    fps: int = 30,
) -> bool:
    """Write a single episode grid video via local temp file, then copy to output_path.

    This avoids NFS write corruption (moov atom not flushed).
    Returns True on success.
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

    # Copy from local tmp to final destination (works correctly on NFS)
    shutil.copy2(str(tmp), str(output_path))
    tmp.unlink()
    return True


def get_available_episodes(data_dir: Path) -> list[int]:
    """Scan parquet files and return sorted list of episode indices."""
    episodes = []
    data_root = data_dir / "data"
    if not data_root.exists():
        return episodes
    for chunk_dir in sorted(data_root.iterdir()):
        if not chunk_dir.is_dir():
            continue
        for pf in sorted(chunk_dir.iterdir()):
            if pf.suffix == ".parquet" and pf.name.startswith("episode_"):
                try:
                    idx = int(pf.stem.split("_")[1])
                    episodes.append(idx)
                except (IndexError, ValueError):
                    pass
    return sorted(episodes)


def main():
    parser = argparse.ArgumentParser(description="Render LeRobot dataset episodes as grid videos.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/path/to/franka_lerobot_data/pick_up_the_box",
        help="Path to the LeRobot dataset directory (repo_id root).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to scripts/<repo_id>/ (repo_id = dataset dir name).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Specific episode indices to render. If omitted, renders all episodes.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument(
        "--filter_json",
        type=str,
        default=None,
        help="Path to filter.json; if provided, only render episodes listed in it.",
    )
    parser.add_argument(
        "--per_camera",
        action="store_true",
        default=False,
        help="Also render individual per-camera videos.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    repo_id = dataset_dir.name  # e.g. "close_the_drawer"

    # Default output: scripts/<repo_id>/
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SCRIPT_DIR / repo_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which episodes to render
    if args.episodes is not None:
        episode_indices = sorted(args.episodes)
    elif args.filter_json:
        with open(args.filter_json) as f:
            filt = json.load(f)
        episode_indices = sorted(ep["metadata"]["ep_idx"] for ep in filt.get("episodes", []))
    else:
        episode_indices = get_available_episodes(dataset_dir)

    if not episode_indices:
        print("[ERROR] No episodes found to render.")
        sys.exit(1)

    print(f"Dataset:   {dataset_dir}")
    print(f"Repo ID:   {repo_id}")
    print(f"Output:    {output_dir}")
    print(f"Episodes:  {len(episode_indices)} ({episode_indices[0]}..{episode_indices[-1]})")
    print(f"FPS:       {args.fps}")
    print()

    ok_count = 0
    fail_count = 0

    for i, ep_idx in enumerate(episode_indices):
        print(f"[{i+1}/{len(episode_indices)}] Rendering episode {ep_idx}...")
        frames = load_episode(dataset_dir, ep_idx)
        if not frames:
            print(f"  [SKIP] No frames loaded for episode {ep_idx}")
            fail_count += 1
            continue

        # Grid video
        grid_path = output_dir / f"episode_{ep_idx:06d}_grid.mp4"
        if render_episode_video(frames, grid_path, fps=args.fps):
            print(f"  -> {grid_path}")
            ok_count += 1
        else:
            fail_count += 1

        # Optional per-camera videos
        if args.per_camera:
            for key, label in zip(CAMERA_KEYS, CAMERA_LABELS):
                cam_dir = output_dir / "per_camera" / label.lower().replace(" ", "_")
                cam_dir.mkdir(parents=True, exist_ok=True)
                cam_path = cam_dir / f"episode_{ep_idx:06d}.mp4"

                h, w = frames[0][key].shape[:2]
                if w % 2 != 0:
                    w += 1
                if h % 2 != 0:
                    h += 1

                import os
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="cam_")
                os.close(tmp_fd)

                cam_cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
                    "-r", str(args.fps), "-i", "pipe:0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "fast", "-crf", "23",
                    tmp_path,
                ]
                try:
                    proc = subprocess.Popen(cam_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
                    shutil.copy2(tmp_path, str(cam_path))
                except Exception as e:
                    print(f"  [WARN] Failed to write per-camera video {cam_path}: {e}")
                finally:
                    if Path(tmp_path).exists():
                        Path(tmp_path).unlink()

    print()
    print(f"Done. Rendered {ok_count} videos, {fail_count} failed/skipped.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
