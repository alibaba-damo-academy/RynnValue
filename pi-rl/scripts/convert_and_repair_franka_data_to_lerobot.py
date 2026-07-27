# adapted from openpi
"""
Convert Franka demo data to LeRobot format, with optional data-quality repair:

  1. (Repair) Clip gripper channels to a configurable ceiling (default 252).
  2. (Repair) Detect per-frame action jumps and drop episodes whose total
     jump count exceeds a configurable budget.
  3. (Convert) Run the regular pyvips + parallel LeRobot conversion pipeline.

Repairs are applied in-memory on each episode's timeseries.parquet during
conversion, so no cleaned copy is written to disk.

Reference:
  - scripts/convert_franka_data_to_lerobot.py   (base conversion pipeline)
  - RynnValue/repair_dataset.py                 (gripper clip + jump filter)
"""

import json
import os
import sys
import time
import shutil
import queue
import threading
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import pyvips before pyarrow/pandas: pyarrow loads the system libstdc++
# (older CXXABI) on import, which then can't satisfy the CXXABI_1.3.15 that
# pyvips's conda libicui18n requires. Loading pyvips first pulls in conda's
# newer libstdc++, which is backward-compatible for pyarrow.
import pyvips

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


# ── constants ────────────────────────────────────────────────────────────────

ARM_ACTIVITY_THRESH = 0.1

PARQUET_COLUMNS = [
    "frame_index",
    "action.arm",
    "action.gripper",
    "observation.state.arm",
    "observation.state.gripper",
]

CAM_KEYS = [
    "observation.images.left_side",
    "observation.images.left_wrist",
    "observation.images.right_side",
    "observation.images.right_wrist",
]

DEFAULT_GRIPPER_CLIP = 252.0
DEFAULT_JUMP_THRESHOLD = 0.15


# ── repair helpers ───────────────────────────────────────────────────────────

def count_action_jumps(df: pd.DataFrame, jump_threshold: float, mode: str) -> int:
    """
    Count per-frame |diff| jumps that exceed `jump_threshold` across the
    active arm joint-position columns only. Gripper channels live on a
    0~255 scale where legitimate open/close moves produce large diffs, so
    they would dominate the count under any threshold tuned for radians;
    they are intentionally excluded here. `mode` mirrors detect_active_arms:
      'dual-arm'         -> check all 14 arm dims
      'single-arm-left'  -> check left  7 arm dims
      'single-arm-right' -> check right 7 arm dims
    """
    arm_all = np.stack(df["action.arm"].values).astype(np.float32)

    if mode == "dual-arm":
        arm = arm_all
    elif mode == "single-arm-left":
        arm = arm_all[:, :7]
    else:
        arm = arm_all[:, 7:]

    if arm.shape[0] < 2:
        return 0
    diff = np.abs(np.diff(arm, axis=0))
    return int(np.sum(diff > jump_threshold))


def clip_gripper_columns(df: pd.DataFrame, max_val: float) -> tuple[pd.DataFrame, int]:
    """
    Clip both array-typed (`action.gripper` / `observation.state.gripper`)
    and any stray scalar gripper columns to `max_val`. Returns (df, n_frames)
    where n_frames is the number of rows whose gripper array was clipped.
    """
    clipped_frames = 0
    gripper_keys = ["action.gripper", "observation.state.gripper"]

    for key in gripper_keys:
        if key not in df.columns:
            continue
        sample = df[key].iloc[0]
        if isinstance(sample, np.ndarray):
            arr = np.stack(df[key].values).astype(np.float32, copy=True)
            mask = arr.max(axis=1) > max_val
            n = int(mask.sum())
            if n > 0:
                arr[mask] = np.clip(arr[mask], a_min=None, a_max=max_val)
                df = df.copy()
                df[key] = list(arr)
                clipped_frames = max(clipped_frames, n)
        else:
            col_max = float(df[key].max())
            if col_max > max_val:
                df = df.copy()
                df[key] = df[key].clip(upper=max_val)
                clipped_frames = max(clipped_frames, int((df[key] >= max_val).sum()))

    return df, clipped_frames


# ── image helpers ────────────────────────────────────────────────────────────

def resize_image_pyvips(img_path: str, size=(224, 224)) -> np.ndarray:
    target_h, target_w = size
    img = pyvips.Image.thumbnail(img_path, target_w, height=target_h, size="force")

    if img.bands == 1:
        img = img.colourspace("srgb")
    elif img.bands > 3:
        img = img[:3]

    mem = img.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint8).reshape(target_h, target_w, img.bands).copy()

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]

    return arr


def load_single_cam_image(cam_prefix: str, frame_idx: int, image_size: tuple[int, int]) -> np.ndarray:
    return resize_image_pyvips(f"{cam_prefix}/{frame_idx:06d}.jpg", image_size)


def detect_active_arms(data_dir: Path, max_scan_episodes: int = 5) -> dict:
    episode_dirs = sorted(
        ep for ep in data_dir.glob("episode_*")
        if not ep.name.endswith("_f")
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* directories found in {data_dir}")

    scan_eps = episode_dirs[:max_scan_episodes]

    left_arm_actions, right_arm_actions = [], []
    left_gripper_actions, right_gripper_actions = [], []

    for ep_dir in scan_eps:
        ts_path = ep_dir / "timeseries.parquet"
        if not ts_path.exists():
            continue
        df = pd.read_parquet(ts_path, columns=["action.arm", "action.gripper"])
        if len(df) == 0:
            continue

        arm_matrix = np.stack(df["action.arm"].values)
        gripper_matrix = np.stack(df["action.gripper"].values)

        left_arm_actions.append(arm_matrix[:, :7])
        right_arm_actions.append(arm_matrix[:, 7:])
        left_gripper_actions.append(gripper_matrix[:, 0])
        right_gripper_actions.append(gripper_matrix[:, 1])

    if not left_arm_actions:
        raise RuntimeError(f"No action data found in first {max_scan_episodes} episodes")

    la = np.concatenate(left_arm_actions, axis=0)
    ra = np.concatenate(right_arm_actions, axis=0)
    lg = np.concatenate(left_gripper_actions, axis=0)
    rg = np.concatenate(right_gripper_actions, axis=0)

    left_arm_std = float(np.max(np.std(la, axis=0)))
    right_arm_std = float(np.max(np.std(ra, axis=0)))
    left_gripper_std = float(np.std(lg))
    right_gripper_std = float(np.std(rg))

    left_arm_active = left_arm_std >= ARM_ACTIVITY_THRESH
    right_arm_active = right_arm_std >= ARM_ACTIVITY_THRESH
    left_gripper_active = left_gripper_std >= ARM_ACTIVITY_THRESH
    right_gripper_active = right_gripper_std >= ARM_ACTIVITY_THRESH

    if not left_arm_active and not right_arm_active:
        raise ValueError(
            f"Neither arm is active! left_arm_std={left_arm_std:.6f}, "
            f"right_arm_std={right_arm_std:.6f}, threshold={ARM_ACTIVITY_THRESH}."
        )

    if left_arm_active and right_arm_active:
        mode = "dual-arm"
    elif left_arm_active:
        mode = "single-arm-left"
    else:
        mode = "single-arm-right"

    if mode == "dual-arm":
        if not left_gripper_active or not right_gripper_active:
            raise ValueError(
                f"Dual-arm mode but gripper mismatch: "
                f"left_gripper_active={left_gripper_active}, "
                f"right_gripper_active={right_gripper_active}. "
                f"left_gripper_std={left_gripper_std:.6f}, "
                f"right_gripper_std={right_gripper_std:.6f}."
            )
    elif mode == "single-arm-left":
        if not left_gripper_active:
            raise ValueError(
                f"Left arm is active but left gripper is not! "
                f"left_gripper_std={left_gripper_std:.6f}."
            )
        if right_gripper_active:
            raise ValueError(
                f"Left arm is single-arm but right gripper is active! "
                f"right_gripper_std={right_gripper_std:.6f}."
            )
    elif mode == "single-arm-right":
        if not right_gripper_active:
            raise ValueError(
                f"Right arm is active but right gripper is not! "
                f"right_gripper_std={right_gripper_std:.6f}."
            )
        if left_gripper_active:
            raise ValueError(
                f"Right arm is single-arm but left gripper is active! "
                f"left_gripper_std={left_gripper_std:.6f}."
            )

    return {
        "left_arm": left_arm_active,
        "right_arm": right_arm_active,
        "left_gripper": left_gripper_active,
        "right_gripper": right_gripper_active,
        "mode": mode,
        "_left_arm_std": left_arm_std,
        "_right_arm_std": right_arm_std,
        "_left_gripper_std": left_gripper_std,
        "_right_gripper_std": right_gripper_std,
    }


def build_features(active: dict, H: int, W: int) -> dict:
    is_dual = active["mode"] == "dual-arm"
    arm_dim = 14 if is_dual else 7
    grip_dim = 2 if is_dual else 1

    return {
        "observation.images.left_side": {
            "dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"],
        },
        "observation.images.left_wrist": {
            "dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"],
        },
        "observation.images.right_side": {
            "dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"],
        },
        "observation.images.right_wrist": {
            "dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"],
        },
        "observation.state.arm": {"dtype": "float32", "shape": (arm_dim,), "names": ["state"]},
        "observation.state.gripper": {"dtype": "float32", "shape": (grip_dim,), "names": ["state"]},
        "action.arm": {"dtype": "float32", "shape": (arm_dim,), "names": ["action"]},
        "action.gripper": {"dtype": "float32", "shape": (grip_dim,), "names": ["action"]},
    }


def write_sample_videos(frames: list[dict], out_dir: Path, fps: int):
    if not frames:
        raise ValueError("No frames provided for sample videos.")

    out_dir.mkdir(parents=True, exist_ok=True)

    for cam_key in CAM_KEYS:
        vid_path = out_dir / f"{cam_key}.mp4"
        with imageio.get_writer(str(vid_path), fps=fps, codec="libx264") as writer:
            for frame in frames:
                img = frame[cam_key]
                if not isinstance(img, np.ndarray):
                    img = np.asarray(img)
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                writer.append_data(img)


def process_episode(
    ep_dir_str: str,
    active: dict,
    image_size: tuple[int, int],
    language_instruction: str,
    episode_image_threads: int,
    profile: bool = False,
    clip_gripper_max: float | None = DEFAULT_GRIPPER_CLIP,
    jump_threshold: float | None = DEFAULT_JUMP_THRESHOLD,
    max_jumps_per_episode: int = 0,
) -> dict:
    """
    Preprocess one episode in a worker process, optionally applying the
    gripper clip + action-jump filter before image loading.
    """
    t0 = time.perf_counter()
    ep_dir = Path(ep_dir_str)
    ts_path = ep_dir / "timeseries.parquet"

    if not ts_path.exists():
        return {"status": "skip", "reason": "no timeseries.parquet", "ep_dir": ep_dir_str}

    try:
        t1 = time.perf_counter()
        df = pd.read_parquet(ts_path, columns=PARQUET_COLUMNS)
        t2 = time.perf_counter()

        if len(df) == 0:
            return {"status": "skip", "reason": "empty timeseries", "ep_dir": ep_dir_str}

        # ── repair: action-jump filter ─────────────────────────────────
        if jump_threshold is not None:
            jump_count = count_action_jumps(df, jump_threshold, active["mode"])
            if jump_count > max_jumps_per_episode:
                return {
                    "status": "skip",
                    "reason": f"action jumps: {jump_count} > {max_jumps_per_episode}",
                    "ep_dir": ep_dir_str,
                    "jumps": jump_count,
                }

        # ── repair: gripper clip ───────────────────────────────────────
        n_clipped_frames = 0
        if clip_gripper_max is not None:
            df, n_clipped_frames = clip_gripper_columns(df, clip_gripper_max)

        t_repair = time.perf_counter()

        img_root = ep_dir / "raw_images"
        cam_prefixes = [str(img_root / k) for k in CAM_KEYS]
        for prefix in cam_prefixes:
            if not Path(prefix).exists():
                return {"status": "error", "reason": f"missing camera dir: {prefix}", "ep_dir": ep_dir_str}

        frame_indices = df["frame_index"].to_numpy(dtype=int)
        action_arm_all = np.stack(df["action.arm"].values).astype(np.float32)
        action_gripper_all = np.stack(df["action.gripper"].values).astype(np.float32)
        state_arm_all = np.stack(df["observation.state.arm"].values).astype(np.float32)
        state_gripper_all = np.stack(df["observation.state.gripper"].values).astype(np.float32)

        if active["mode"] == "dual-arm":
            act_arm, st_arm = action_arm_all, state_arm_all
            act_grip, st_grip = action_gripper_all, state_gripper_all
        elif active["mode"] == "single-arm-left":
            act_arm, st_arm = action_arm_all[:, :7], state_arm_all[:, :7]
            act_grip, st_grip = action_gripper_all[:, :1], state_gripper_all[:, :1]
        else:
            act_arm, st_arm = action_arm_all[:, 7:], state_arm_all[:, 7:]
            act_grip, st_grip = action_gripper_all[:, 1:], state_gripper_all[:, 1:]

        t3 = time.perf_counter()

        n_frames = len(frame_indices)
        frame_imgs = [{} for _ in range(n_frames)]

        if episode_image_threads <= 1:
            for i, fi in enumerate(frame_indices):
                for cam_idx, cam_key in enumerate(CAM_KEYS):
                    frame_imgs[i][cam_key] = load_single_cam_image(cam_prefixes[cam_idx], int(fi), image_size)
        else:
            with ThreadPoolExecutor(max_workers=episode_image_threads) as img_pool:
                futures = {}
                for i, fi in enumerate(frame_indices):
                    for cam_idx, cam_key in enumerate(CAM_KEYS):
                        fut = img_pool.submit(load_single_cam_image, cam_prefixes[cam_idx], int(fi), image_size)
                        futures[fut] = (i, cam_key)
                for fut in as_completed(futures):
                    i, cam_key = futures[fut]
                    frame_imgs[i][cam_key] = fut.result()

        t4 = time.perf_counter()

        frames = []
        for i in range(n_frames):
            frames.append({
                "action.arm": act_arm[i],
                "action.gripper": act_grip[i],
                "observation.state.arm": st_arm[i],
                "observation.state.gripper": st_grip[i],
                **frame_imgs[i],
                "task": language_instruction,
            })

        t5 = time.perf_counter()

        result = {
            "status": "ok",
            "ep_dir": ep_dir_str,
            "is_failed": ep_dir.name.endswith("_f"),
            "frames": frames,
            "n_clipped_frames": n_clipped_frames,
        }
        if profile:
            result["timing"] = {
                "metadata_s": t1 - t0,
                "parquet_s": t2 - t1,
                "repair_s": t_repair - t2,
                "slice_s": t3 - t_repair,
                "images_s": t4 - t3,
                "assemble_s": t5 - t4,
                "total_s": t5 - t0,
                "n_frames": n_frames,
            }
        return result

    except Exception as e:
        return {
            "status": "error",
            "reason": f"{type(e).__name__}: {e}",
            "ep_dir": ep_dir_str,
        }


PROCESSED_SOURCES_FILENAME = "processed_sources.txt"


def _load_processed_sources(dataset_root: Path) -> set[str]:
    log_path = dataset_root / PROCESSED_SOURCES_FILENAME
    if not log_path.exists():
        return set()
    return {
        line.strip()
        for line in log_path.read_text().splitlines()
        if line.strip()
    }


def _append_processed_source(dataset_root: Path, ep_name: str) -> None:
    log_path = dataset_root / PROCESSED_SOURCES_FILENAME
    with open(log_path, "a") as f:
        f.write(ep_name + "\n")


def _is_processed(processed: set[str], source_tag: str, ep_name: str) -> bool:
    """
    True if this source episode was already handled. New runs record a
    source-namespaced key ("<src_dir>/<episode>"); legacy sidecars stored the
    bare "<episode>" name, so honour both to avoid re-appending on an in-place
    RESUME of a repo built before namespacing.
    """
    return f"{source_tag}/{ep_name}" in processed or ep_name in processed


def main(
    data_dir: str,
    *,
    repo_name: str,
    language_instruction: str = "pick up the bread",
    max_episodes: int | None = None,
    image_size: tuple[int, int] = (224, 224),
    fps: int = 30,
    push_to_hub: bool = False,
    episode_filter_path: str | None = None,
    num_workers: int = max(1, min(8, (os.cpu_count() or 4))),
    prefetch_episodes: int = 8,
    episode_image_threads: int = 8,
    profile: bool = False,
    generate_sample_video: bool = True,
    sample_video_name: str = "sample_episode_preview",
    resume: bool = False,
    # ── repair options ────────────────────────────────────────────────
    clip_gripper_max: float | None = DEFAULT_GRIPPER_CLIP,
    jump_threshold: float | None = DEFAULT_JUMP_THRESHOLD,
    max_jumps_per_episode: int = 0,
):
    """
    Convert Franka demo data to LeRobot format with optional repairs.

    Repair flags:
      --clip-gripper-max FLOAT      clip action/observation gripper channels
                                    to this ceiling (None disables).
      --jump-threshold FLOAT        per-frame |action diff| above this counts
                                    as a jump (None disables jump filter).
      --max-jumps-per-episode INT   drop any episode with more jumps than this.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    # Namespace the resume/dedup key by source directory. When several source
    # dirs are merged into one repo via RESUME, two dirs may reuse the same
    # episode_* directory name; keying processed_sources by "<src_dir>/<ep>"
    # keeps them distinct so a later segment is never wrongly skipped.
    source_tag = data_dir.name
    output_path = HF_LEROBOT_HOME / repo_name

    resume_existing = False
    resume_skip = 0
    if output_path.exists():
        if resume:
            resume_existing = True
            print(f"[INFO] Resuming existing dataset at {output_path}")
        else:
            print(f"[INFO] Removing existing dataset at {output_path}")
            shutil.rmtree(output_path)

    H, W = image_size

    print(f"\n{'='*60}")
    print("[SCAN] Detecting active arms from first 5 episodes...")
    print(f"{'='*60}")
    active = detect_active_arms(data_dir, max_scan_episodes=5)

    mode = active["mode"]
    left_arm_str = "ACTIVE" if active["left_arm"] else "inactive"
    right_arm_str = "ACTIVE" if active["right_arm"] else "inactive"
    left_grip_str = "ACTIVE" if active["left_gripper"] else "inactive"
    right_grip_str = "ACTIVE" if active["right_gripper"] else "inactive"

    print(f"\n[DETECT] Task mode: **{mode}**")
    print(f"         Left  arm:     {left_arm_str}  (std={active['_left_arm_std']:.6f})")
    print(f"         Right arm:     {right_arm_str}  (std={active['_right_arm_std']:.6f})")
    print(f"         Left  gripper: {left_grip_str}  (std={active['_left_gripper_std']:.6f})")
    print(f"         Right gripper: {right_grip_str}  (std={active['_right_gripper_std']:.6f})")
    print(f"         Threshold:     {ARM_ACTIVITY_THRESH}")
    print(f"         → Will store only active arm features\n")

    print(f"[REPAIR] clip_gripper_max={clip_gripper_max}")
    print(f"[REPAIR] jump_threshold={jump_threshold}, max_jumps_per_episode={max_jumps_per_episode}")
    print()

    features = build_features(active, H, W)
    print(f"[FEATURES] LeRobot feature layout:")
    for key, spec in features.items():
        print(f"           {key}: {spec['shape']}")
    print()

    if resume_existing:
        dataset = LeRobotDataset(repo_id=repo_name)
        existing_features = dataset.meta.info.get("features", {})
        for key, spec in features.items():
            existing_shape = tuple(existing_features.get(key, {}).get("shape", ()))
            if existing_shape != tuple(spec["shape"]):
                raise RuntimeError(
                    f"Resume aborted: feature '{key}' has shape {existing_shape} on disk "
                    f"but freshly-detected mode '{active['mode']}' wants {spec['shape']}. "
                    f"Refusing to mix incompatible episodes."
                )
        dataset.start_image_writer(num_processes=16, num_threads=32)
        resume_skip = dataset.meta.total_episodes
        print(f"[RESUME] Dataset already has {resume_skip} episodes saved.")
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_name,
            robot_type="franka",
            fps=fps,
            features=features,
            image_writer_threads=32,
            image_writer_processes=16,
        )

    episode_dirs = sorted(data_dir.glob("episode_*"))
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* directories found in {data_dir}")

    n_failed = sum(1 for ep in episode_dirs if ep.name.endswith("_f"))
    n_success = len(episode_dirs) - n_failed
    print(f"[INFO] Found {len(episode_dirs)} episodes ({n_success} success, {n_failed} failed '_f') in {data_dir}")

    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]
        print(f"[INFO] Using first {max_episodes} episodes")

    if resume_existing:
        processed_sources = _load_processed_sources(output_path)
        before = len(episode_dirs)
        episode_dirs = [
            ep for ep in episode_dirs
            if not _is_processed(processed_sources, source_tag, ep.name)
        ]
        n_skipped_resume = before - len(episode_dirs)
        print(
            f"[RESUME] Sidecar log has {len(processed_sources)} previously-processed source episode(s); "
            f"dataset has {resume_skip} saved episode(s). "
            f"Skipping {n_skipped_resume} source dir(s) for this run."
        )
        if episode_dirs:
            print(f"[RESUME] Next source episode: {episode_dirs[0].name}")
        else:
            print(f"[RESUME] Nothing left to convert.")

    if prefetch_episodes < 1:
        raise ValueError("prefetch_episodes must be >= 1")
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if episode_image_threads < 1:
        raise ValueError("episode_image_threads must be >= 1")

    prefetch_episodes = max(prefetch_episodes, num_workers)

    print(f"[INFO] Parallel settings:")
    print(f"       num_workers={num_workers}")
    print(f"       prefetch_episodes={prefetch_episodes}")
    print(f"       episode_image_threads={episode_image_threads}")
    print(f"       image_backend=pyvips")
    print()

    n_converted = 0
    n_skipped = 0
    n_jumps_dropped = 0
    total_clipped_frames = 0
    converted_ep_indices: list[int] = []
    failed_ep_indices: list[int] = []

    total_add_frame_s = 0.0
    total_save_episode_s = 0.0
    worker_profile_rows = []

    first_success_frames = None
    first_success_ep_name = None

    write_queue = queue.Queue(maxsize=prefetch_episodes)
    writer_error = [None]

    def _writer_thread():
        nonlocal total_add_frame_s, total_save_episode_s, n_converted
        nonlocal first_success_frames, first_success_ep_name, total_clipped_frames
        try:
            while True:
                item = write_queue.get()
                if item is None:
                    break
                result, ep_name, ep_key = item

                total_clipped_frames += int(result.get("n_clipped_frames", 0))

                if (not result["is_failed"]) and first_success_frames is None:
                    first_success_frames = result["frames"]
                    first_success_ep_name = ep_name
                    if generate_sample_video:
                        _vid_dir = _SCRIPT_DIR / sample_video_name
                        try:
                            write_sample_videos(frames=first_success_frames, out_dir=_vid_dir, fps=fps)
                            print(f"\n[INFO] Sample videos written to {_vid_dir} (from {first_success_ep_name})")
                        except Exception as e:
                            print(f"\n[WARN] Failed to write sample videos: {type(e).__name__}: {e}")

                t_add0 = time.perf_counter()
                for frame in result["frames"]:
                    dataset.add_frame(frame)
                t_add1 = time.perf_counter()
                dataset.save_episode()
                t_add2 = time.perf_counter()

                _append_processed_source(output_path, ep_key)

                total_add_frame_s += (t_add1 - t_add0)
                total_save_episode_s += (t_add2 - t_add1)

                ep_idx = resume_skip + n_converted
                if result["is_failed"]:
                    failed_ep_indices.append(ep_idx)
                else:
                    converted_ep_indices.append(ep_idx)
                n_converted += 1

                if profile and "timing" in result:
                    worker_profile_rows.append((ep_name, result["timing"], t_add1 - t_add0, t_add2 - t_add1))

                pbar.update(1)
        except Exception as e:
            writer_error[0] = e

    pbar = tqdm(total=len(episode_dirs), desc="Converting episodes")
    writer = threading.Thread(target=_writer_thread, daemon=True)
    writer.start()

    if episode_dirs:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            future_to_idx = {}
            results_buffer = {}
            next_submit = 0
            next_consume = 0

            initial = min(prefetch_episodes, len(episode_dirs))
            for _ in range(initial):
                ep_dir = episode_dirs[next_submit]
                fut = pool.submit(
                    process_episode,
                    str(ep_dir),
                    active,
                    image_size,
                    language_instruction,
                    episode_image_threads,
                    profile,
                    clip_gripper_max,
                    jump_threshold,
                    max_jumps_per_episode,
                )
                future_to_idx[fut] = next_submit
                next_submit += 1

            pending_futures = set(future_to_idx.keys())
            while next_consume < len(episode_dirs):
                done_futs = set()
                for fut in as_completed(pending_futures):
                    done_futs.add(fut)
                    idx = future_to_idx.pop(fut)
                    results_buffer[idx] = fut.result()

                    while next_submit < len(episode_dirs) and len(future_to_idx) < prefetch_episodes:
                        ep_dir = episode_dirs[next_submit]
                        new_fut = pool.submit(
                            process_episode,
                            str(ep_dir),
                            active,
                            image_size,
                            language_instruction,
                            episode_image_threads,
                            profile,
                            clip_gripper_max,
                            jump_threshold,
                            max_jumps_per_episode,
                        )
                        future_to_idx[new_fut] = next_submit
                        pending_futures.add(new_fut)
                        next_submit += 1

                    while next_consume in results_buffer:
                        result = results_buffer.pop(next_consume)
                        ep_name = episode_dirs[next_consume].name
                        ep_key = f"{source_tag}/{ep_name}"

                        if result["status"] == "skip":
                            reason = result["reason"]
                            if "action jumps" in reason:
                                n_jumps_dropped += 1
                                print(f"[REPAIR] Dropping {ep_name}: {reason}")
                            else:
                                print(f"[WARN] Skipping {ep_name}: {reason}")
                            n_skipped += 1
                            _append_processed_source(output_path, ep_key)
                            pbar.update(1)
                        elif result["status"] == "error":
                            print(f"[WARN] Error processing {ep_name}, skipping: {result['reason']}")
                            n_skipped += 1
                            _append_processed_source(output_path, ep_key)
                            pbar.update(1)
                        else:
                            write_queue.put((result, ep_name, ep_key))

                        next_consume += 1

                    if writer_error[0] is not None:
                        raise writer_error[0]

                pending_futures -= done_futs

    write_queue.put(None)
    writer.join()
    pbar.close()

    if writer_error[0] is not None:
        raise writer_error[0]

    print(f"\n[INFO] Converted {n_converted} episodes, skipped {n_skipped}.")
    if clip_gripper_max is not None:
        print(f"[REPAIR] Gripper clipped in {total_clipped_frames} frame(s) (ceiling={clip_gripper_max}).")
    if jump_threshold is not None:
        print(f"[REPAIR] Dropped {n_jumps_dropped} episode(s) due to action jumps.")
    print(f"[INFO] Dataset saved to {output_path}")

    if generate_sample_video and first_success_frames is None:
        print("[WARN] No successful non-'_f' episode found; sample video not generated.")

    if profile and worker_profile_rows:
        print("\n[PROFILE] Per-episode timings:")
        for ep_name, wt, add_s, save_s in worker_profile_rows:
            print(
                f"  {ep_name}: "
                f"parquet={wt.get('parquet_s', 0):.3f}s, "
                f"repair={wt.get('repair_s', 0):.3f}s, "
                f"slice={wt.get('slice_s', 0):.3f}s, "
                f"images={wt.get('images_s', 0):.3f}s, "
                f"assemble={wt.get('assemble_s', 0):.3f}s, "
                f"worker_total={wt.get('total_s', 0):.3f}s, "
                f"add_frame={add_s:.3f}s, "
                f"save_episode={save_s:.3f}s, "
                f"frames={wt.get('n_frames', -1)}"
            )
        print(
            f"\n[PROFILE] Main-process totals: "
            f"add_frame_total={total_add_frame_s:.3f}s, "
            f"save_episode_total={total_save_episode_s:.3f}s"
        )

    if episode_filter_path is not None:
        filter_path = Path(episode_filter_path).expanduser().resolve()
        filter_path.parent.mkdir(parents=True, exist_ok=True)

        prior_success: list[int] = []
        if resume_existing and filter_path.exists():
            try:
                prior = json.loads(filter_path.read_text())
                prior_success = [
                    int(e["metadata"]["ep_idx"])
                    for e in prior.get("episodes", [])
                    if isinstance(e, dict) and "metadata" in e and "ep_idx" in e["metadata"]
                ]
            except Exception as e:
                print(f"[WARN] Could not parse existing filter file ({e}); overwriting.")

        merged = sorted(set(prior_success) | set(converted_ep_indices))

        filter_data = {
            "episodes": [
                {"metadata": {"repo_id": repo_name, "ep_idx": ep_idx}}
                for ep_idx in merged
            ]
        }
        filter_path.write_text(json.dumps(filter_data, indent=2))
        print(f"[INFO] Episode filter written to {filter_path}")
        print(
            f"       {len(merged)} success episode(s) listed "
            f"({len(converted_ep_indices)} from this run, {len(prior_success)} preexisting)."
        )
        if failed_ep_indices:
            print(f"       {len(failed_ep_indices)} '_f' episode(s) excluded (indices: {failed_ep_indices})")
            print(f"       To include any of them, add their ep_idx to the file manually.")

    if push_to_hub:
        tags = ["franka", "panda", mode]
        dataset.push_to_hub(
            tags=tags,
            private=True,
            push_videos=True,
            license="apache-2.0",
        )

    return output_path


if __name__ == "__main__":
    tyro.cli(main)
