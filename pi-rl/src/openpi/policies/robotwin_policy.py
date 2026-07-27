# adapted from openpi
"""Policy I/O transforms for the RoboTwin (bimanual aloha-style) dataset.

The RoboTwin lerobot dump under
``/path/to/robotwin_lerobot_data/lerobot_robotwin_eef_clean_50``
stores 16-dim end-effector state/action vectors and three RGB cameras:

  - ``observation.state``  : [16]  = (left xyz, left quat(wxyz), left gripper,
                                     right xyz, right quat(wxyz), right gripper)
  - ``action``             : [16]  = same layout as state
  - ``observation.images.cam_high``       : exterior view
  - ``observation.images.cam_left_wrist`` : left wrist
  - ``observation.images.cam_right_wrist``: right wrist

We feed the high camera into ``base_0_rgb`` and the two wrist cameras into the
left/right wrist slots. Since the action representation is end-effector pose +
gripper (not standard Aloha joint angles), no aloha-specific gripper or joint
flipping conversion is needed -- we hand the values to the model as-is.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.shared import image_tools


# State / action layout: 8 dims per arm = (xyz=3) + (quat=4) + (gripper=1).
ROBOTWIN_STATE_DIM = 16
ROBOTWIN_ACTION_DIM = 16


def make_robotwin_example() -> dict:
    """Random example matching the RoboTwin pipeline (used for inference smoke tests)."""
    return {
        "state": np.ones((ROBOTWIN_STATE_DIM,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _parse_image(img) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    # LeRobot stores images as [C, H, W]; downstream transforms expect [H, W, C].
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = einops.rearrange(img, "c h w -> h w c")
    return img


@dataclasses.dataclass(frozen=True)
class RobotwinInputs(transforms.DataTransformFn):
    """Pack RoboTwin dataset rows into the dict the pi0 / pi05 model expects."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(
                f"Unexpected RoboTwin camera(s): {sorted(unexpected)}. "
                f"Expected one of {self.EXPECTED_CAMERAS}."
            )

        base_image = _parse_image(in_images["cam_high"])
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        for dest, source in (
            ("left_wrist_0_rgb", "cam_left_wrist"),
            ("right_wrist_0_rgb", "cam_right_wrist"),
        ):
            if source in in_images:
                images[dest] = _parse_image(in_images[source])
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        state = np.asarray(data["state"], dtype=np.float32)

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RobotwinOutputs(transforms.DataTransformFn):
    """Trim model actions back to RoboTwin's 16 dims for inference."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ROBOTWIN_ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class RobotwinRLInputs(transforms.DataTransformFn):
    """RobotwinInputs variant for offline RL training.

    The RL data config asks lerobot for state/cam_high/cam_left_wrist/cam_right_wrist
    at offsets ``[0, H/fps]`` and (when present) ``progress`` at
    ``[0, 1/fps, ..., H/fps]`` via ``delta_timestamps``, so each of those fields
    arrives as a 2-frame stack (progress as ``H+1``). We split them, pack the
    current view in pi0 format with all three cameras, expose next-step state
    plus base/left-wrist/right-wrist pixels as side-channel keys (the IQL CNN
    encoder uses all three RoboTwin cameras), and derive reward + mask from
    progress.

    Next-step images are pre-resized here (the standard ``ResizeImages`` model
    transform only touches ``data["image"]``).
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    # Match the standard ResizeImages step in ModelTransformFactory.
    resize_height: int = 224
    resize_width: int = 224

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(
                f"Unexpected RoboTwin camera(s): {sorted(unexpected)}. "
                f"Expected one of {self.EXPECTED_CAMERAS}."
            )

        state_pair = np.asarray(data["state"])

        # Current-view pi0 image dict: all three cameras (base + 2 wrists).
        cur_base = _parse_image(np.asarray(in_images["cam_high"])[0])
        images = {"base_0_rgb": cur_base}
        image_masks = {"base_0_rgb": np.True_}
        for dest, source in (
            ("left_wrist_0_rgb", "cam_left_wrist"),
            ("right_wrist_0_rgb", "cam_right_wrist"),
        ):
            if source in in_images:
                images[dest] = _parse_image(np.asarray(in_images[source])[0])
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(cur_base)
                image_masks[dest] = np.False_

        # Next-view side-channel: base + left wrist + right wrist (the IQL CNN
        # encoder consumes all three RoboTwin cameras). Pre-resize to match the
        # encoder input size since these keys bypass ResizeImages.
        nxt_base = _parse_image(np.asarray(in_images["cam_high"])[1])
        nxt_base = np.asarray(image_tools.resize_with_pad(nxt_base, self.resize_height, self.resize_width))
        if "cam_left_wrist" in in_images:
            nxt_left_wrist = _parse_image(np.asarray(in_images["cam_left_wrist"])[1])
            nxt_left_wrist = np.asarray(image_tools.resize_with_pad(nxt_left_wrist, self.resize_height, self.resize_width))
        else:
            nxt_left_wrist = np.zeros_like(nxt_base)
        if "cam_right_wrist" in in_images:
            nxt_right_wrist = _parse_image(np.asarray(in_images["cam_right_wrist"])[1])
            nxt_right_wrist = np.asarray(image_tools.resize_with_pad(nxt_right_wrist, self.resize_height, self.resize_width))
        else:
            nxt_right_wrist = np.zeros_like(nxt_base)

        inputs: dict = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(state_pair[0], dtype=np.float32),
            # RL side-channel keys (the pi0 model never sees these).
            "next_state": np.asarray(state_pair[1], dtype=np.float32),
            "next_image_base": nxt_base,
            "next_image_left_wrist": nxt_left_wrist,
            "next_image_right_wrist": nxt_right_wrist,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = np.asarray(data["actions_is_pad"]).reshape(-1).astype(bool)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "progress" in data:
            # progress sampled at [t, t+1, ..., t+H] (H+1 values aligned with
            # the action chunk + endpoint). Expose the raw sequence; the IQL
            # trainer reads ``batch["progress"]`` and computes the PBRS shaped
            # chunk reward R = Σ γ^h · (γ·Φ(s_{h+1}) - Φ(s_h)) itself, so we
            # don't precompute per-step deltas here. The scalar ``reward``
            # (endpoint delta) is still emitted as a human-readable debug
            # field and as the terminal-mode fallback in _chunk_reward.
            progress = np.asarray(data["progress"], dtype=np.float32).reshape(-1)
            inputs["progress"] = progress
            inputs["reward"] = np.float32(progress[-1] - progress[0])
            is_pad = data.get("progress_is_pad")
            if is_pad is not None:
                is_pad = np.asarray(is_pad).reshape(-1).astype(bool)
                inputs["progress_is_pad"] = is_pad
                terminal = bool(is_pad[-1])
            else:
                terminal = False
        else:
            is_pad = data.get("state_is_pad")
            terminal = bool(np.asarray(is_pad).reshape(-1)[1]) if is_pad is not None else False
            inputs["reward"] = np.float32(0.0 if terminal else -1.0)
        inputs["mask"] = np.float32(0.0 if terminal else 1.0)
        return inputs
