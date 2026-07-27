import logging
import pathlib
from datetime import datetime

import cv2
import imageio
import numpy as np

from client.envs.utils import process_image_for_obs


def raw_frame_from_raw_obs(raw_obs, side_camera_id, wrist_camera_id):
    """Build raw frame (side + wrist concatenated) from raw_obs."""
    if "image" not in raw_obs or side_camera_id not in raw_obs["image"]:
        return None
    side = process_image_for_obs(raw_obs["image"][side_camera_id], bgr_to_rgb=True, image_size=None)
    if wrist_camera_id not in raw_obs.get("image", {}):
        return side
    wrist = process_image_for_obs(raw_obs["image"][wrist_camera_id], bgr_to_rgb=True, image_size=None)
    if side.shape[:2] != wrist.shape[:2]:
        wrist = cv2.resize(wrist, (side.shape[1], side.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.concatenate([side, wrist], axis=1)


def save_episode_video(frames, video_dir, ep_count, fps=30, quality=8, prefix="raw"):
    """Save buffered raw frames to disk. Call when episode ends."""
    if not video_dir or not frames:
        return
    ts = datetime.now().strftime("%m%d_%H%M%S")
    out_path = pathlib.Path(video_dir) / f"{prefix}_{ts}_train_ep{ep_count:06d}.mp4"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(out_path), frames, fps=fps, quality=quality, codec="libx264")
        logging.getLogger(__name__).info("Saved raw video to %s", out_path)
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to save raw video: %s", e)
