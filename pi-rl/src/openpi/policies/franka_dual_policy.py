# adapted from openpi
"""
Policy transforms for dual-arm Franka real-robot dataset.

Dataset keys (from scripts/convert_franka_data_to_lerobot.py):

  Images (all 4 cameras):
    observation.images.left_side      (224,224,3) uint8  — overhead camera
    observation.images.left_wrist     (224,224,3) uint8  — left wrist camera
    observation.images.right_side     (224,224,3) uint8  — right side camera
    observation.images.right_wrist    (224,224,3) uint8  — right wrist camera

  State & action (dual-arm):
    observation.state.arm(14,), observation.state.gripper(2,)
    action.arm(14,), action.gripper(2,)

After FrankaDualInputs transforms:
  state   → (16,) = arm(14) + gripper(2)
  actions → (16,) = arm(14) + gripper(2)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import image_tools

# Dual-arm dimensionalities
DUAL_ARM_DIM = 16  # arm(14) + gripper(2)


def make_franka_dual_example() -> dict:
    """Creates a random input example for the dual-arm Franka policy."""
    return {
        "observation.images.left_side":    np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.left_wrist":   np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.right_side":   np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.right_wrist":  np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.state.arm":           np.random.rand(14).astype(np.float32),
        "observation.state.gripper":       np.random.rand(2).astype(np.float32),
        "action.arm":                      np.random.rand(14).astype(np.float32),
        "action.gripper":                  np.random.rand(2).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Convert LeRobot float32 (C,H,W) or uint8 (H,W,C) to uint8 (H,W,C)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


_IMAGE_MAP = {
    "observation.images.left_side":  "left_side_0_rgb",
    "observation.images.left_wrist": "left_wrist_0_rgb",
    "observation.images.right_side": "right_side_0_rgb",
    "observation.images.right_wrist": "right_wrist_0_rgb",
}


@dataclasses.dataclass(frozen=True)
class FrankaDualInputs(transforms.DataTransformFn):
    """
    Convert dual-arm Franka LeRobot observation dict into the pi0.5 model's expected format.

    State: arm(14) + gripper(2) = (16,)
    Actions: arm(14) + gripper(2) = (16,)

    Image slots:
      left_side_0_rgb   <- observation.images.left_side   (overhead camera)
      left_wrist_0_rgb  <- observation.images.left_wrist
      right_side_0_rgb  <- observation.images.right_side
      right_wrist_0_rgb <- observation.images.right_wrist
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Concatenate arm(14) + gripper(2) = state(16,)
        arm_val = np.asarray(data["observation.state.arm"], dtype=np.float32)
        gripper_val = np.asarray(data["observation.state.gripper"], dtype=np.float32)
        # Defensive: keep gripper rank aligned with arm in case lerobot squeezes
        # a trailing dim (mirrors the single-arm path).
        while gripper_val.ndim < arm_val.ndim:
            gripper_val = gripper_val[..., None]
        state = np.concatenate([arm_val, gripper_val], axis=-1)

        # Concatenate action.arm(14) + action.gripper(2) = actions(16,)
        actions = None
        if "action.arm" in data:
            action_arm = np.asarray(data["action.arm"], dtype=np.float32)
            action_gripper = np.asarray(data["action.gripper"], dtype=np.float32) if "action.gripper" in data else None
            if action_gripper is not None:
                while action_gripper.ndim < action_arm.ndim:
                    action_gripper = action_gripper[..., None]
                actions = np.concatenate([action_arm, action_gripper], axis=-1)
            else:
                actions = action_arm

        inputs = {
            "state": state,
            "image": {
                target_key: _parse_image(data[src_key])
                for src_key, target_key in _IMAGE_MAP.items()
            },
            "image_mask": {
                target_key: np.True_
                for target_key in _IMAGE_MAP.values()
            },
        }

        if actions is not None:
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaDualOutputs(transforms.DataTransformFn):
    """Extract dual-arm action dims from the model output.

    Returns all 16 dims (left 8 + right 8).
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :DUAL_ARM_DIM])
        gripper_indices = [7, 15]  # left and right grippers
        for t in range(actions.shape[0]):
            gripper_values = [float(actions[t, i]) for i in gripper_indices]
            print(
                f"[FrankaDualOutputs] t={t} "
                f"action={actions[t].round(4)} gripper_values={gripper_values}"
            )
        return {"actions": actions}


_DELTA_IMAGE_MAP = {
    "observation.images.left_side":   "left_side_0_rgb",
    "observation.images.left_wrist":  "left_wrist_0_rgb",
    "observation.images.right_wrist": "right_wrist_0_rgb",
}


@dataclasses.dataclass(frozen=True)
class FrankaDualDeltaInputs(transforms.DataTransformFn):
    """Dual-arm with 3 cameras only (left_side + left_wrist + right_wrist).

    right_side is completely excluded from the model input.
    Used with full-delta actions (joint + gripper all delta).
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        arm_val = np.asarray(data["observation.state.arm"], dtype=np.float32)
        gripper_val = np.asarray(data["observation.state.gripper"], dtype=np.float32)
        while gripper_val.ndim < arm_val.ndim:
            gripper_val = gripper_val[..., None]
        state = np.concatenate([arm_val, gripper_val], axis=-1)

        actions = None
        if "action.arm" in data:
            action_arm = np.asarray(data["action.arm"], dtype=np.float32)
            action_gripper = np.asarray(data["action.gripper"], dtype=np.float32) if "action.gripper" in data else None
            if action_gripper is not None:
                while action_gripper.ndim < action_arm.ndim:
                    action_gripper = action_gripper[..., None]
                actions = np.concatenate([action_arm, action_gripper], axis=-1)
            else:
                actions = action_arm

        images = {
            target: _parse_image(data[src])
            for src, target in _DELTA_IMAGE_MAP.items()
        }

        inputs = {
            "state": state,
            "image": images,
            "image_mask": {target: np.True_ for target in images},
        }
        if actions is not None:
            inputs["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaDualDeltaOutputs(transforms.DataTransformFn):
    """Extract dual-arm action dims (16) from model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :DUAL_ARM_DIM])}


@dataclasses.dataclass(frozen=True)
class FrankaDualRLInputs(transforms.DataTransformFn):
    """FrankaDualInputs variant for offline RL training.

    The RL data config asks lerobot for state/images at offsets
    ``[0, H/fps]`` and (when present) ``progress`` at ``[0, 1/fps, ..., H/fps]``
    via ``delta_timestamps``, so each of those fields arrives as a 2-frame
    stack ``[current, next]`` (progress as ``H+1``). We split them, parse the
    images, pack the current view in pi0 format, expose next-step pixels/state
    as side-channel keys, and derive reward+mask from progress.

    Next-step images are resized here (the standard ``ResizeImages`` model
    transform only touches ``data["image"]``, so without this the next-step
    pixels would skip resizing and end up shape-mismatched against the IQL
    encoder).
    """

    model_type: _model.ModelType
    # Match the standard ResizeImages step in ModelTransformFactory.
    resize_height: int = 224
    resize_width: int = 224

    def __call__(self, data: dict) -> dict:
        # Parse 2-frame stacked images: shape is [2, C, H, W] or [2, H, W, C]
        left_side_pair = np.asarray(data["observation.images.left_side"])
        left_wrist_pair = np.asarray(data["observation.images.left_wrist"])
        right_side_pair = np.asarray(data["observation.images.right_side"])
        right_wrist_pair = np.asarray(data["observation.images.right_wrist"])

        cur_left_side = _parse_image(left_side_pair[0])
        cur_left_wrist = _parse_image(left_wrist_pair[0])
        cur_right_side = _parse_image(right_side_pair[0])
        cur_right_wrist = _parse_image(right_wrist_pair[0])

        nxt_left_side = _parse_image(left_side_pair[1])
        nxt_left_wrist = _parse_image(left_wrist_pair[1])
        nxt_right_side = _parse_image(right_side_pair[1])
        nxt_right_wrist = _parse_image(right_wrist_pair[1])

        # Pre-resize next-step images for IQL encoder
        nxt_left_side = np.asarray(image_tools.resize_with_pad(nxt_left_side, self.resize_height, self.resize_width))
        nxt_left_wrist = np.asarray(image_tools.resize_with_pad(nxt_left_wrist, self.resize_height, self.resize_width))
        nxt_right_side = np.asarray(image_tools.resize_with_pad(nxt_right_side, self.resize_height, self.resize_width))
        nxt_right_wrist = np.asarray(image_tools.resize_with_pad(nxt_right_wrist, self.resize_height, self.resize_width))

        # Parse 2-frame stacked state: shape is [2, arm_dim] and [2, gripper_dim]
        arm_pair = np.asarray(data["observation.state.arm"], dtype=np.float32)
        gripper_pair = np.asarray(data["observation.state.gripper"], dtype=np.float32)
        # Match FrankaDualInputs: gripper may be stored as scalar (shape [2])
        # rather than length-N vector; pad its trailing dim so both pairs have
        # the same ndim before per-frame concat.
        while gripper_pair.ndim < arm_pair.ndim:
            gripper_pair = gripper_pair[..., None]

        # Concatenate arm + gripper for current state
        cur_state = np.concatenate([arm_pair[0], gripper_pair[0]], axis=-1)
        nxt_state = np.concatenate([arm_pair[1], gripper_pair[1]], axis=-1)

        inputs: dict = {
            "state": cur_state,
            "image": {
                "left_side_0_rgb": cur_left_side,
                "left_wrist_0_rgb": cur_left_wrist,
                "right_side_0_rgb": cur_right_side,
                "right_wrist_0_rgb": cur_right_wrist,
            },
            "image_mask": {
                "left_side_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_side_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            # RL side-channel keys (the pi0 model never sees these)
            "next_state": nxt_state,
            "next_image_left_side": nxt_left_side,
            "next_image_left_wrist": nxt_left_wrist,
            "next_image_right_side": nxt_right_side,
            "next_image_right_wrist": nxt_right_wrist,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        elif "action.arm" in data:
            action_arm = np.asarray(data["action.arm"], dtype=np.float32)
            action_gripper = (
                np.asarray(data["action.gripper"], dtype=np.float32)
                if "action.gripper" in data
                else None
            )
            if action_gripper is not None:
                while action_gripper.ndim < action_arm.ndim:
                    action_gripper = action_gripper[..., None]
                inputs["actions"] = np.concatenate([action_arm, action_gripper], axis=-1)
            else:
                inputs["actions"] = action_arm
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = np.asarray(data["actions_is_pad"]).reshape(-1).astype(bool)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "progress" in data:
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
            is_pad = data.get("observation.state_is_pad")
            terminal = bool(np.asarray(is_pad).reshape(-1)[1]) if is_pad is not None else False
            inputs["reward"] = np.float32(0.0 if terminal else -1.0)
        inputs["mask"] = np.float32(0.0 if terminal else 1.0)

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaDualDeltaRLInputs(transforms.DataTransformFn):
    """Dual-arm RL with 3 cameras (left_side + left_wrist + right_wrist).

    right_side completely excluded. Provides next-step for IQL critic.
    """

    model_type: _model.ModelType
    resize_height: int = 224
    resize_width: int = 224

    def __call__(self, data: dict) -> dict:
        left_side_pair = np.asarray(data["observation.images.left_side"])
        left_wrist_pair = np.asarray(data["observation.images.left_wrist"])
        right_wrist_pair = np.asarray(data["observation.images.right_wrist"])

        cur_left_side = _parse_image(left_side_pair[0])
        cur_left_wrist = _parse_image(left_wrist_pair[0])
        cur_right_wrist = _parse_image(right_wrist_pair[0])

        nxt_left_side = np.asarray(image_tools.resize_with_pad(
            _parse_image(left_side_pair[1]), self.resize_height, self.resize_width))
        nxt_left_wrist = np.asarray(image_tools.resize_with_pad(
            _parse_image(left_wrist_pair[1]), self.resize_height, self.resize_width))
        nxt_right_wrist = np.asarray(image_tools.resize_with_pad(
            _parse_image(right_wrist_pair[1]), self.resize_height, self.resize_width))

        arm_pair = np.asarray(data["observation.state.arm"], dtype=np.float32)
        gripper_pair = np.asarray(data["observation.state.gripper"], dtype=np.float32)
        while gripper_pair.ndim < arm_pair.ndim:
            gripper_pair = gripper_pair[..., None]
        cur_state = np.concatenate([arm_pair[0], gripper_pair[0]], axis=-1)
        nxt_state = np.concatenate([arm_pair[1], gripper_pair[1]], axis=-1)

        inputs: dict = {
            "state": cur_state,
            "image": {
                "left_side_0_rgb": cur_left_side,
                "left_wrist_0_rgb": cur_left_wrist,
                "right_wrist_0_rgb": cur_right_wrist,
            },
            "image_mask": {
                "left_side_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "next_state": nxt_state,
            "next_image_left_side": nxt_left_side,
            "next_image_left_wrist": nxt_left_wrist,
            "next_image_right_wrist": nxt_right_wrist,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        elif "action.arm" in data:
            action_arm = np.asarray(data["action.arm"], dtype=np.float32)
            action_gripper = (
                np.asarray(data["action.gripper"], dtype=np.float32)
                if "action.gripper" in data else None
            )
            if action_gripper is not None:
                while action_gripper.ndim < action_arm.ndim:
                    action_gripper = action_gripper[..., None]
                inputs["actions"] = np.concatenate([action_arm, action_gripper], axis=-1)
            else:
                inputs["actions"] = action_arm
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = np.asarray(data["actions_is_pad"]).reshape(-1).astype(bool)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "progress" in data:
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
            is_pad = data.get("observation.state_is_pad")
            terminal = bool(np.asarray(is_pad).reshape(-1)[1]) if is_pad is not None else False
            inputs["reward"] = np.float32(0.0 if terminal else -1.0)
        inputs["mask"] = np.float32(0.0 if terminal else 1.0)

        return inputs
