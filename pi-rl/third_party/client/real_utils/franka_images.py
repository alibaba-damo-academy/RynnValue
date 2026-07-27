"""Image processing utilities for the Franka ROS2 environment.

Provides ROS2 Image message conversion and aspect-ratio-preserving resize.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
from PIL import Image

from client.real_utils.franka_constants import TARGET_IMAGE_SIZE

if TYPE_CHECKING:
    from sensor_msgs.msg import Image as ROSImage


def ros_image_to_numpy(msg: "ROSImage") -> np.ndarray:
    """Convert a ROS2 ``sensor_msgs/Image`` to an ``(H, W, 3)`` uint8 RGB array."""
    enc = msg.encoding.lower()
    channel_map = {
        "rgb8": 3, "bgr8": 3,
        "rgba8": 4, "bgra8": 4,
        "mono8": 1,
    }
    channels = channel_map.get(enc, 3)
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, channels
    )

    if enc == "bgr8":
        img = img[:, :, ::-1].copy()
    elif enc == "bgra8":
        img = img[:, :, [2, 1, 0]].copy()
    elif enc == "rgba8":
        img = img[:, :, :3].copy()
    elif enc == "mono8":
        img = np.stack([img[:, :, 0]] * 3, axis=-1)

    return img


def resize_with_pad(
    img: np.ndarray, size: Tuple[int, int] = TARGET_IMAGE_SIZE
) -> np.ndarray:
    """Resize ``img`` to ``size`` (W, H) preserving aspect ratio, padding with black."""
    target_w, target_h = size
    pil_img = Image.fromarray(img)
    orig_w, orig_h = pil_img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))
    # Pillow >= 10 moved resampling to Image.Resampling; fall back for older versions.
    _bilinear = getattr(Image, "Resampling", Image).BILINEAR
    resized = pil_img.resize((new_w, new_h), _bilinear)

    padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))
    return np.array(padded)
