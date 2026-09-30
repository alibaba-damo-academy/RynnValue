# adapted from openpi
"""
Optimized policy transforms for Franka real-robot dataset.

Changes vs. original franka_single_policy / franka_dual_policy:
  1. Camera selection: single-arm uses 2 cameras (left_side + left_wrist),
     dual-arm uses 3 (+ right_wrist). Unused cameras are zero-filled with
     image_mask=False.
  2. Gripper binarization: continuous gripper values are thresholded to {0, 1}
     before normalization, so the model learns a clean open/close decision.
  3. Output gripper threshold: inference outputs are sigmoid-thresholded on
     gripper dims to produce clean binary commands.

Dataset keys (same as original):
  Images:  observation.images.{left|right}_{side|wrist}  (224,224,3) uint8
  State:   observation.state.arm(7 or 14), observation.state.gripper(1 or 2)
  Action:  action.arm(7 or 14), action.gripper(1 or 2)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import image_tools

SINGLE_ARM_DIM = 8   # arm(7) + gripper(1)
DUAL_ARM_DIM = 16    # arm(14) + gripper(2)

# Indices of gripper dims inside the concatenated action vector.
_SINGLE_GRIPPER_INDICES = [7]
_DUAL_GRIPPER_INDICES = [7, 15]


def _terminal_bonus_value(data, failed_episodes, terminal, terminal_bonus) -> float:
    """Terminal reward for an RL transition.

    On the terminal transition, a FAILED episode -- one whose ``episode_index`` is in
    ``failed_episodes`` -- gets ``-terminal_bonus`` (a failure penalty, e.g. -1), while a
    SUCCESSFUL episode gets 0. Non-terminal transitions get 0. Failed episodes therefore stay in
    offline-RL training but are penalized for reaching the end without success. When there is no
    failure label (``failed_episodes`` empty or ``episode_index`` absent) the episode is treated
    as successful (0).
    """
    if not terminal:
        return 0.0
    ep = data.get("episode_index")
    is_failed = (
        ep is not None and bool(failed_episodes)
        and int(np.asarray(ep).reshape(-1)[0]) in failed_episodes
    )
    return -float(terminal_bonus) if is_failed else 0.0


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _binarize_gripper(actions: np.ndarray, gripper_indices: list[int], threshold: float) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    for idx in gripper_indices:
        actions[..., idx] = (actions[..., idx] >= threshold).astype(np.float32)
    return actions


# ───────────────────── single-arm ─────────────────────

@dataclasses.dataclass(frozen=True)
class OptimizedFrankaSingleInputs(transforms.DataTransformFn):
    """Single-arm: 2 cameras (left_side + left_wrist), binarized gripper."""

    model_type: _model.ModelType
    gripper_threshold: float = 120.0

    # Active cameras.
    _ACTIVE_CAMERAS = {
        "observation.images.left_side":  "left_side_0_rgb",
        "observation.images.left_wrist": "left_wrist_0_rgb",
    }
    # Masked cameras (zero-filled).
    _MASKED_CAMERAS = {
        "observation.images.right_side":  "right_side_0_rgb",
        "observation.images.right_wrist": "right_wrist_0_rgb",
    }

    def __call__(self, data: dict) -> dict:
        # Concatenate arm(7) + gripper(1) = state(8,)
        if "observation.state" in data:
            # Client sends pre-concatenated state (8,)
            state = np.asarray(data["observation.state"], dtype=np.float32)
        else:
            arm_val = np.asarray(data["observation.state.arm"], dtype=np.float32)
            gripper_val = np.asarray(data["observation.state.gripper"], dtype=np.float32)
            # lerobot can squeeze (1,)-shaped fields to 0-d scalars (or (H,) instead
            # of (H,1) for action chunks); pad gripper's trailing dim to match arm.
            while gripper_val.ndim < arm_val.ndim:
                gripper_val = gripper_val[..., None]
            state = np.concatenate([arm_val, gripper_val], axis=-1)

        actions = None
        if "action.arm" in data:
            action_arm = np.asarray(data["action.arm"], dtype=np.float32)
            action_gripper = (
                np.asarray(data["action.gripper"], dtype=np.float32)
                if "action.gripper" in data else None
            )
            if action_gripper is not None:
                while action_gripper.ndim < action_arm.ndim:
                    action_gripper = action_gripper[..., None]
                actions = np.concatenate([action_arm, action_gripper], axis=-1)
            else:
                actions = action_arm
            actions = _binarize_gripper(actions, _SINGLE_GRIPPER_INDICES, self.gripper_threshold)

        # Parse active camera images.
        active_images = {
            target: _parse_image(data[src])
            for src, target in self._ACTIVE_CAMERAS.items()
        }
        # Zero-fill masked cameras using the first active image as shape reference.
        ref_image = next(iter(active_images.values()))
        masked_images = {
            target: np.zeros_like(ref_image)
            for _, target in self._MASKED_CAMERAS.items()
        }

        inputs = {
            "state": state,
            "image": {**active_images, **masked_images},
            "image_mask": {
                target: np.True_ for target in active_images
            } | {
                target: np.False_ for target in masked_images
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
class OptimizedFrankaSingleOutputs(transforms.DataTransformFn):
    """Slice first 8 dims, threshold gripper to binary."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :SINGLE_ARM_DIM])
        for idx in _SINGLE_GRIPPER_INDICES:
            actions[:, idx] = (actions[:, idx] >= 0.5).astype(np.float32)
        return {"actions": actions}


# ───────────────────── dual-arm ─────────────────────

@dataclasses.dataclass(frozen=True)
class OptimizedFrankaDualInputs(transforms.DataTransformFn):
    """Dual-arm: 3 cameras (left_side + left_wrist + right_wrist), binarized gripper."""

    model_type: _model.ModelType
    gripper_threshold: float = 120.0

    _ACTIVE_CAMERAS = {
        "observation.images.left_side":   "left_side_0_rgb",
        "observation.images.left_wrist":  "left_wrist_0_rgb",
        "observation.images.right_wrist": "right_wrist_0_rgb",
    }
    _MASKED_CAMERAS = {
        "observation.images.right_side": "right_side_0_rgb",
    }

    def __call__(self, data: dict) -> dict:
        arm_val = np.asarray(data["observation.state.arm"], dtype=np.float32)
        gripper_val = np.asarray(data["observation.state.gripper"], dtype=np.float32)
        while gripper_val.ndim < arm_val.ndim:
            gripper_val = gripper_val[..., None]
        state = np.concatenate([arm_val, gripper_val], axis=-1)

        actions = None
        if "action.arm" in data:
            action_arm = np.asarray(data["action.arm"], dtype=np.float32)
            action_gripper = (
                np.asarray(data["action.gripper"], dtype=np.float32)
                if "action.gripper" in data else None
            )
            if action_gripper is not None:
                while action_gripper.ndim < action_arm.ndim:
                    action_gripper = action_gripper[..., None]
                actions = np.concatenate([action_arm, action_gripper], axis=-1)
            else:
                actions = action_arm
            actions = _binarize_gripper(actions, _DUAL_GRIPPER_INDICES, self.gripper_threshold)

        active_images = {
            target: _parse_image(data[src])
            for src, target in self._ACTIVE_CAMERAS.items()
        }
        ref_image = next(iter(active_images.values()))
        masked_images = {
            target: np.zeros_like(ref_image)
            for _, target in self._MASKED_CAMERAS.items()
        }

        inputs = {
            "state": state,
            "image": {**active_images, **masked_images},
            "image_mask": {
                target: np.True_ for target in active_images
            } | {
                target: np.False_ for target in masked_images
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
class OptimizedFrankaDualOutputs(transforms.DataTransformFn):
    """Slice first 16 dims, threshold both grippers to binary."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :DUAL_ARM_DIM])
        for idx in _DUAL_GRIPPER_INDICES:
            actions[:, idx] = (actions[:, idx] >= 0.5).astype(np.float32)
        return {"actions": actions}


# ───────────────────── RL (IQL) inputs ─────────────────────

@dataclasses.dataclass(frozen=True)
class OptimizedFrankaSingleRLInputs(transforms.DataTransformFn):
    """Single-arm RL: 2 cameras, binarized gripper, next-step for IQL critic."""

    model_type: _model.ModelType
    gripper_threshold: float = 120.0
    resize_height: int = 224
    resize_width: int = 224
    progress_reward_weight: float = 1.0
    terminal_bonus: float = 0.0
    failed_episodes: frozenset[int] = frozenset()

    def __call__(self, data: dict) -> dict:
        # Parse 2-frame stacked images
        left_side_pair = np.asarray(data["observation.images.left_side"])
        left_wrist_pair = np.asarray(data["observation.images.left_wrist"])
        right_side_pair = np.asarray(data["observation.images.right_side"])
        right_wrist_pair = np.asarray(data["observation.images.right_wrist"])

        # Current images (for pi05 policy)
        cur_left_side = _parse_image(left_side_pair[0])
        cur_left_wrist = _parse_image(left_wrist_pair[0])
        # Next images (for IQL critic, pre-resized)
        nxt_left_side = np.asarray(image_tools.resize_with_pad(
            _parse_image(left_side_pair[1]), self.resize_height, self.resize_width))
        nxt_left_wrist = np.asarray(image_tools.resize_with_pad(
            _parse_image(left_wrist_pair[1]), self.resize_height, self.resize_width))

        # Parse 2-frame stacked state
        arm_pair = np.asarray(data["observation.state.arm"], dtype=np.float32)
        gripper_pair = np.asarray(data["observation.state.gripper"], dtype=np.float32)
        while gripper_pair.ndim < arm_pair.ndim:
            gripper_pair = gripper_pair[..., None]
        cur_state = np.concatenate([arm_pair[0], gripper_pair[0]], axis=-1)
        nxt_state = np.concatenate([arm_pair[1], gripper_pair[1]], axis=-1)

        # Zero-fill masked cameras
        zero_img = np.zeros_like(cur_left_side)

        inputs: dict = {
            "state": cur_state,
            "image": {
                "left_side_0_rgb": cur_left_side,
                "left_wrist_0_rgb": cur_left_wrist,
                "right_side_0_rgb": zero_img,
                "right_wrist_0_rgb": zero_img,
            },
            "image_mask": {
                "left_side_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_side_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
            "next_state": nxt_state,
            "next_image_left_side": nxt_left_side,
            "next_image_left_wrist": nxt_left_wrist,
        }

        # Actions with gripper binarization
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
        if "actions" in inputs:
            inputs["actions"] = _binarize_gripper(
                inputs["actions"], _SINGLE_GRIPPER_INDICES, self.gripper_threshold)
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = np.asarray(data["actions_is_pad"]).reshape(-1).astype(bool)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        # Reward / mask
        if "progress" in data:
            progress = np.asarray(data["progress"], dtype=np.float32).reshape(-1)
            inputs["progress"] = progress
            delta = float(progress[-1] - progress[0])
            is_pad = data.get("progress_is_pad")
            if is_pad is not None:
                inputs["progress_is_pad"] = np.asarray(is_pad).reshape(-1).astype(bool)
                terminal = bool(inputs["progress_is_pad"][-1])
            else:
                terminal = False
            term = _terminal_bonus_value(data, self.failed_episodes, terminal, self.terminal_bonus)
            inputs["terminal_reward"] = np.float32(term)
            inputs["reward"] = np.float32(self.progress_reward_weight * delta + term)
        else:
            is_pad = data.get("observation.state_is_pad")
            terminal = bool(np.asarray(is_pad).reshape(-1)[1]) if is_pad is not None else False
            inputs["reward"] = np.float32(0.0 if terminal else -1.0)
        inputs["mask"] = np.float32(0.0 if terminal else 1.0)

        return inputs


@dataclasses.dataclass(frozen=True)
class OptimizedFrankaDualRLInputs(transforms.DataTransformFn):
    """Dual-arm RL: 3 cameras, binarized gripper, next-step for IQL critic."""

    model_type: _model.ModelType
    gripper_threshold: float = 120.0
    resize_height: int = 224
    resize_width: int = 224
    progress_reward_weight: float = 1.0
    terminal_bonus: float = 0.0
    failed_episodes: frozenset[int] = frozenset()

    def __call__(self, data: dict) -> dict:
        # Parse 2-frame stacked images
        left_side_pair = np.asarray(data["observation.images.left_side"])
        left_wrist_pair = np.asarray(data["observation.images.left_wrist"])
        right_side_pair = np.asarray(data["observation.images.right_side"])
        right_wrist_pair = np.asarray(data["observation.images.right_wrist"])

        # Current images
        cur_left_side = _parse_image(left_side_pair[0])
        cur_left_wrist = _parse_image(left_wrist_pair[0])
        cur_right_wrist = _parse_image(right_wrist_pair[0])
        # Next images (for IQL critic, pre-resized)
        nxt_left_side = np.asarray(image_tools.resize_with_pad(
            _parse_image(left_side_pair[1]), self.resize_height, self.resize_width))
        nxt_left_wrist = np.asarray(image_tools.resize_with_pad(
            _parse_image(left_wrist_pair[1]), self.resize_height, self.resize_width))
        nxt_right_wrist = np.asarray(image_tools.resize_with_pad(
            _parse_image(right_wrist_pair[1]), self.resize_height, self.resize_width))

        # Parse 2-frame stacked state
        arm_pair = np.asarray(data["observation.state.arm"], dtype=np.float32)
        gripper_pair = np.asarray(data["observation.state.gripper"], dtype=np.float32)
        while gripper_pair.ndim < arm_pair.ndim:
            gripper_pair = gripper_pair[..., None]
        cur_state = np.concatenate([arm_pair[0], gripper_pair[0]], axis=-1)
        nxt_state = np.concatenate([arm_pair[1], gripper_pair[1]], axis=-1)

        # Zero-fill masked camera (right_side)
        zero_img = np.zeros_like(cur_left_side)

        inputs: dict = {
            "state": cur_state,
            "image": {
                "left_side_0_rgb": cur_left_side,
                "left_wrist_0_rgb": cur_left_wrist,
                "right_side_0_rgb": zero_img,
                "right_wrist_0_rgb": cur_right_wrist,
            },
            "image_mask": {
                "left_side_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_side_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
            "next_state": nxt_state,
            "next_image_left_side": nxt_left_side,
            "next_image_left_wrist": nxt_left_wrist,
            "next_image_right_wrist": nxt_right_wrist,
        }

        # Actions with gripper binarization
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
        if "actions" in inputs:
            inputs["actions"] = _binarize_gripper(
                inputs["actions"], _DUAL_GRIPPER_INDICES, self.gripper_threshold)
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = np.asarray(data["actions_is_pad"]).reshape(-1).astype(bool)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        # Reward / mask
        if "progress" in data:
            progress = np.asarray(data["progress"], dtype=np.float32).reshape(-1)
            inputs["progress"] = progress
            delta = float(progress[-1] - progress[0])
            is_pad = data.get("progress_is_pad")
            if is_pad is not None:
                inputs["progress_is_pad"] = np.asarray(is_pad).reshape(-1).astype(bool)
                terminal = bool(inputs["progress_is_pad"][-1])
            else:
                terminal = False
            term = _terminal_bonus_value(data, self.failed_episodes, terminal, self.terminal_bonus)
            inputs["terminal_reward"] = np.float32(term)
            inputs["reward"] = np.float32(self.progress_reward_weight * delta + term)
        else:
            is_pad = data.get("observation.state_is_pad")
            terminal = bool(np.asarray(is_pad).reshape(-1)[1]) if is_pad is not None else False
            inputs["reward"] = np.float32(0.0 if terminal else -1.0)
        inputs["mask"] = np.float32(0.0 if terminal else 1.0)

        return inputs


# ───────────────────── v2: no-state, continuous actions ─────────────────────
#
# v2 changes vs. the v1 Optimized* classes above:
#   * The model no longer receives joint state or gripper as input. We feed a
#     dummy zero state of shape (1,) that PadStatesAndActions pads to the
#     model action dim. pi05 (with discrete_state_input=False) never reads the
#     state: TokenizePrompt pops it, and embed_suffix gates the state token on
#     `not self.pi05`.
#   * Actions remain continuous absolute positions for both joints and gripper
#     — no gripper binarization in the inputs, no sigmoid thresholding in the
#     outputs.


@dataclasses.dataclass(frozen=True)
class OptimizedFrankaSingleInputsV2(transforms.DataTransformFn):
    """Single-arm v2: 2 cameras, NO state input, continuous gripper action."""

    model_type: _model.ModelType

    _ACTIVE_CAMERAS = {
        "observation.images.left_side":  "left_side_0_rgb",
        "observation.images.left_wrist": "left_wrist_0_rgb",
    }
    _MASKED_CAMERAS = {
        "observation.images.right_side":  "right_side_0_rgb",
        "observation.images.right_wrist": "right_wrist_0_rgb",
    }

    def __call__(self, data: dict) -> dict:
        # Dummy state — unused by pi05 when discrete_state_input=False.
        state = np.zeros((1,), dtype=np.float32)

        actions = None
        if "action.arm" in data:
            action_arm = np.asarray(data["action.arm"], dtype=np.float32)
            action_gripper = (
                np.asarray(data["action.gripper"], dtype=np.float32)
                if "action.gripper" in data else None
            )
            if action_gripper is not None:
                while action_gripper.ndim < action_arm.ndim:
                    action_gripper = action_gripper[..., None]
                actions = np.concatenate([action_arm, action_gripper], axis=-1)
            else:
                actions = action_arm

        active_images = {
            target: _parse_image(data[src])
            for src, target in self._ACTIVE_CAMERAS.items()
        }
        ref_image = next(iter(active_images.values()))
        masked_images = {
            target: np.zeros_like(ref_image)
            for _, target in self._MASKED_CAMERAS.items()
        }

        inputs = {
            "state": state,
            "image": {**active_images, **masked_images},
            "image_mask": {
                target: np.True_ for target in active_images
            } | {
                target: np.False_ for target in masked_images
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
class OptimizedFrankaSingleOutputsV2(transforms.DataTransformFn):
    """Slice first 8 dims; keep joint + gripper as continuous absolute values."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :SINGLE_ARM_DIM], dtype=np.float32)
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class OptimizedFrankaDualInputsV2(transforms.DataTransformFn):
    """Dual-arm v2: 3 cameras, NO state input, continuous gripper actions."""

    model_type: _model.ModelType

    _ACTIVE_CAMERAS = {
        "observation.images.left_side":   "left_side_0_rgb",
        "observation.images.left_wrist":  "left_wrist_0_rgb",
        "observation.images.right_wrist": "right_wrist_0_rgb",
    }
    _MASKED_CAMERAS = {
        "observation.images.right_side": "right_side_0_rgb",
    }

    def __call__(self, data: dict) -> dict:
        state = np.zeros((1,), dtype=np.float32)

        actions = None
        if "action.arm" in data:
            action_arm = np.asarray(data["action.arm"], dtype=np.float32)
            action_gripper = (
                np.asarray(data["action.gripper"], dtype=np.float32)
                if "action.gripper" in data else None
            )
            if action_gripper is not None:
                while action_gripper.ndim < action_arm.ndim:
                    action_gripper = action_gripper[..., None]
                actions = np.concatenate([action_arm, action_gripper], axis=-1)
            else:
                actions = action_arm

        active_images = {
            target: _parse_image(data[src])
            for src, target in self._ACTIVE_CAMERAS.items()
        }
        ref_image = next(iter(active_images.values()))
        masked_images = {
            target: np.zeros_like(ref_image)
            for _, target in self._MASKED_CAMERAS.items()
        }

        inputs = {
            "state": state,
            "image": {**active_images, **masked_images},
            "image_mask": {
                target: np.True_ for target in active_images
            } | {
                target: np.False_ for target in masked_images
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
class OptimizedFrankaDualOutputsV2(transforms.DataTransformFn):
    """Slice first 16 dims; reorder from [arm(14), gripper(2)] to [L_arm(7), L_grip(1), R_arm(7), R_grip(1)]."""

    def __call__(self, data: dict) -> dict:
        raw = np.asarray(data["actions"][:, :DUAL_ARM_DIM], dtype=np.float32)
        # Model produces [left_arm(7), right_arm(7), left_gripper(1), right_gripper(1)]
        # Client expects  [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
        left_arm = raw[:, :7]
        right_arm = raw[:, 7:14]
        left_gripper = raw[:, 14:15]
        right_gripper = raw[:, 15:16]
        actions = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=-1)
        return {"actions": actions}


# ───────────────────── v2 RL (IQL) inputs ─────────────────────

@dataclasses.dataclass(frozen=True)
class OptimizedFrankaSingleRLInputsV2(transforms.DataTransformFn):
    """Single-arm RL v2: 2 cameras, NO state input, continuous gripper actions."""

    model_type: _model.ModelType
    resize_height: int = 224
    resize_width: int = 224
    progress_reward_weight: float = 1.0
    terminal_bonus: float = 0.0
    failed_episodes: frozenset[int] = frozenset()

    def __call__(self, data: dict) -> dict:
        left_side_raw = np.asarray(data["observation.images.left_side"])
        left_wrist_raw = np.asarray(data["observation.images.left_wrist"])

        # Handle both paired (training: shape (2, H, W, C)) and single-frame
        # (inference: shape (H, W, C)) observation formats.
        if left_side_raw.ndim == 4:  # (2, H, W, C) — paired training format
            cur_left_side = _parse_image(left_side_raw[0])
            cur_left_wrist = _parse_image(left_wrist_raw[0])
            nxt_left_side = np.asarray(image_tools.resize_with_pad(
                _parse_image(left_side_raw[1]), self.resize_height, self.resize_width))
            nxt_left_wrist = np.asarray(image_tools.resize_with_pad(
                _parse_image(left_wrist_raw[1]), self.resize_height, self.resize_width))
        else:  # (H, W, C) — single-frame inference format
            cur_left_side = _parse_image(left_side_raw)
            cur_left_wrist = _parse_image(left_wrist_raw)
            # No next observation at inference — fill with zeros
            nxt_left_side = np.zeros((self.resize_height, self.resize_width, 3), dtype=np.uint8)
            nxt_left_wrist = np.zeros((self.resize_height, self.resize_width, 3), dtype=np.uint8)

        dummy_state = np.zeros((1,), dtype=np.float32)
        zero_img = np.zeros_like(cur_left_side)

        inputs: dict = {
            "state": dummy_state,
            "image": {
                "left_side_0_rgb": cur_left_side,
                "left_wrist_0_rgb": cur_left_wrist,
                "right_side_0_rgb": zero_img,
                "right_wrist_0_rgb": zero_img,
            },
            "image_mask": {
                "left_side_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_side_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
            "next_state": dummy_state,
            "next_image_left_side": nxt_left_side,
            "next_image_left_wrist": nxt_left_wrist,
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
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        if "progress" in data:
            progress = np.asarray(data["progress"], dtype=np.float32).reshape(-1)
            inputs["progress"] = progress
            delta = float(progress[-1] - progress[0])
            is_pad = data.get("progress_is_pad")
            if is_pad is not None:
                inputs["progress_is_pad"] = np.asarray(is_pad).reshape(-1).astype(bool)
                terminal = bool(inputs["progress_is_pad"][-1])
            else:
                terminal = False
            term = _terminal_bonus_value(data, self.failed_episodes, terminal, self.terminal_bonus)
            inputs["terminal_reward"] = np.float32(term)
            inputs["reward"] = np.float32(self.progress_reward_weight * delta + term)
        else:
            is_pad = data.get("observation.state_is_pad")
            terminal = bool(np.asarray(is_pad).reshape(-1)[1]) if is_pad is not None else False
            inputs["reward"] = np.float32(0.0 if terminal else -1.0)
        inputs["mask"] = np.float32(0.0 if terminal else 1.0)

        return inputs


@dataclasses.dataclass(frozen=True)
class OptimizedFrankaDualRLInputsV2(transforms.DataTransformFn):
    """Dual-arm RL v2: 3 cameras, NO state input, continuous gripper actions."""

    model_type: _model.ModelType
    resize_height: int = 224
    resize_width: int = 224
    progress_reward_weight: float = 1.0
    terminal_bonus: float = 0.0
    failed_episodes: frozenset[int] = frozenset()

    def __call__(self, data: dict) -> dict:
        left_side_raw = np.asarray(data["observation.images.left_side"])
        left_wrist_raw = np.asarray(data["observation.images.left_wrist"])
        right_side_raw = np.asarray(data["observation.images.right_side"])
        right_wrist_raw = np.asarray(data["observation.images.right_wrist"])

        # Handle both paired (training: shape (2, H, W, C)) and single-frame
        # (inference: shape (H, W, C)) observation formats.
        if left_side_raw.ndim == 4:  # (2, H, W, C) — paired training format
            cur_left_side = _parse_image(left_side_raw[0])
            cur_left_wrist = _parse_image(left_wrist_raw[0])
            cur_right_wrist = _parse_image(right_wrist_raw[0])
            nxt_left_side = np.asarray(image_tools.resize_with_pad(
                _parse_image(left_side_raw[1]), self.resize_height, self.resize_width))
            nxt_left_wrist = np.asarray(image_tools.resize_with_pad(
                _parse_image(left_wrist_raw[1]), self.resize_height, self.resize_width))
            nxt_right_wrist = np.asarray(image_tools.resize_with_pad(
                _parse_image(right_wrist_raw[1]), self.resize_height, self.resize_width))
        else:  # (H, W, C) — single-frame inference format
            cur_left_side = _parse_image(left_side_raw)
            cur_left_wrist = _parse_image(left_wrist_raw)
            cur_right_wrist = _parse_image(right_wrist_raw)
            # No next observation at inference — fill with zeros
            nxt_left_side = np.zeros((self.resize_height, self.resize_width, 3), dtype=np.uint8)
            nxt_left_wrist = np.zeros((self.resize_height, self.resize_width, 3), dtype=np.uint8)
            nxt_right_wrist = np.zeros((self.resize_height, self.resize_width, 3), dtype=np.uint8)

        dummy_state = np.zeros((1,), dtype=np.float32)
        zero_img = np.zeros_like(cur_left_side)

        inputs: dict = {
            "state": dummy_state,
            "image": {
                "left_side_0_rgb": cur_left_side,
                "left_wrist_0_rgb": cur_left_wrist,
                "right_side_0_rgb": zero_img,
                "right_wrist_0_rgb": cur_right_wrist,
            },
            "image_mask": {
                "left_side_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_side_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
            "next_state": dummy_state,
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
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        if "progress" in data:
            progress = np.asarray(data["progress"], dtype=np.float32).reshape(-1)
            inputs["progress"] = progress
            delta = float(progress[-1] - progress[0])
            is_pad = data.get("progress_is_pad")
            if is_pad is not None:
                inputs["progress_is_pad"] = np.asarray(is_pad).reshape(-1).astype(bool)
                terminal = bool(inputs["progress_is_pad"][-1])
            else:
                terminal = False
            term = _terminal_bonus_value(data, self.failed_episodes, terminal, self.terminal_bonus)
            inputs["terminal_reward"] = np.float32(term)
            inputs["reward"] = np.float32(self.progress_reward_weight * delta + term)
        else:
            is_pad = data.get("observation.state_is_pad")
            terminal = bool(np.asarray(is_pad).reshape(-1)[1]) if is_pad is not None else False
            inputs["reward"] = np.float32(0.0 if terminal else -1.0)
        inputs["mask"] = np.float32(0.0 if terminal else 1.0)

        return inputs


@dataclasses.dataclass(frozen=True)
class OptimizedFrankaMixedOptimizedV2Inputs(transforms.DataTransformFn):
    """Mixed single/dual-arm inputs for multi-task BC over optimized-v2 Franka repos.

    Single-arm (arm7+grip1) and dual-arm (arm14+grip2) samples are unified into the dual
    16-dim action layout ``[left_arm(7), right_arm(7), grip_l(1), grip_r(1)]`` — single-arm
    rows zero-pad the right arm/gripper and mask out the right cameras via ``image_mask``.
    Like the other v2 transforms, no state is fed to the model (dummy zero state) and the
    prompt is the task string only.

    This is the input transform used by the filter-BC data config: the frame whitelist is
    applied at data-loading time, so the transform itself stays plain BC.
    """

    model_type: _model.ModelType
    # Arm-mode override for inference ("single"/"dual"); "auto" detects from the data.
    arm_mode: str = "auto"

    def _detect_arm_mode(self, data: dict) -> str:
        if self.arm_mode != "auto":
            return self.arm_mode
        if "action.arm" in data:
            return "dual" if np.asarray(data["action.arm"]).shape[-1] >= 14 else "single"
        # Inference: fall back to whether the right wrist camera carries any content.
        right_wrist = data.get("observation.images.right_wrist")
        if right_wrist is not None and np.asarray(right_wrist).any():
            return "dual"
        return "single"

    def __call__(self, data: dict) -> dict:
        dual = self._detect_arm_mode(data) == "dual"

        left_side = _parse_image(data["observation.images.left_side"])
        left_wrist = _parse_image(data["observation.images.left_wrist"])
        right_wrist = _parse_image(data["observation.images.right_wrist"]) if dual else np.zeros_like(left_side)
        zero_img = np.zeros_like(left_side)

        dummy_state = np.zeros((1,), dtype=np.float32)
        inputs: dict = {
            "state": dummy_state,
            "image": {
                "left_side_0_rgb": left_side,
                "left_wrist_0_rgb": left_wrist,
                "right_side_0_rgb": zero_img,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "left_side_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_side_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_ if dual else np.False_,
            },
        }

        if "action.arm" in data:
            arm = np.asarray(data["action.arm"], dtype=np.float32)
            gripper = np.asarray(data["action.gripper"], dtype=np.float32)
            while gripper.ndim < arm.ndim:
                gripper = gripper[..., None]
            if dual:
                inputs["actions"] = np.concatenate([arm, gripper], axis=-1)
            else:
                # Unify into the dual layout: zero-pad the right arm + right gripper.
                unified = np.zeros(arm.shape[:-1] + (DUAL_ARM_DIM,), dtype=np.float32)
                unified[..., :7] = arm
                unified[..., 14:15] = gripper
                inputs["actions"] = unified
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = np.asarray(data["actions_is_pad"]).reshape(-1).astype(bool)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


# ───────────────────── v3: real state input, continuous actions ─────────────────────
#
# v3 = v2 shape but with real joint + gripper state routed to the model.
# Model configs must keep pi05's default ``discrete_state_input=True`` so that
# ``PaligemmaTokenizer`` discretizes the state into 256 bins and appends it to
# the language prompt as ``Task: ..., State: <bin ids>;\nAction:``.
# Actions remain continuous absolute joint + gripper values (no binarization).


@dataclasses.dataclass(frozen=True)
class OptimizedFrankaSingleInputsV3(transforms.DataTransformFn):
    """Single-arm v3: 2 cameras, real state (arm+gripper), continuous actions.

    Args:
        model_type: The model type enum (PI0, PI05, ...).
        use_next_state_action: When True, the action target at step t becomes
            the *next* step's joint + gripper observation (i.e. the dataset's
            ``observation.state.{arm,gripper}`` sequence sliced as ``[1:]``),
            instead of the original ``action.{arm,gripper}``. Requires the
            data config to request ``action_horizon + 1`` frames for both
            ``observation.state.arm`` and ``observation.state.gripper``.
        state_as_input: When True (default), the current joint + gripper
            state is passed to the model as ``state``. When False, a zero-dim
            dummy is substituted so state does not reach the model (matching
            the v2 "vision-only" setup).
    """

    model_type: _model.ModelType
    use_next_state_action: bool = False
    state_as_input: bool = True

    _ACTIVE_CAMERAS = {
        "observation.images.left_side":  "left_side_0_rgb",
        "observation.images.left_wrist": "left_wrist_0_rgb",
    }
    _MASKED_CAMERAS = {
        "observation.images.right_side":  "right_side_0_rgb",
        "observation.images.right_wrist": "right_wrist_0_rgb",
    }

    def __call__(self, data: dict) -> dict:
        if "observation.state" in data:
            state_seq = np.asarray(data["observation.state"], dtype=np.float32)
        else:
            arm_val = np.asarray(data["observation.state.arm"], dtype=np.float32)
            gripper_val = np.asarray(data["observation.state.gripper"], dtype=np.float32)
            while gripper_val.ndim < arm_val.ndim:
                gripper_val = gripper_val[..., None]
            state_seq = np.concatenate([arm_val, gripper_val], axis=-1)

        if self.state_as_input:
            state = state_seq[0] if state_seq.ndim >= 2 else state_seq
        else:
            state = np.zeros((1,), dtype=np.float32)

        if self.use_next_state_action:
            if state_seq.ndim < 2:
                raise ValueError(
                    "use_next_state_action=True requires a state sequence with a time "
                    f"dimension (action_horizon+1 frames); got shape {state_seq.shape}."
                )
            actions = state_seq[1:]
        else:
            actions = None
            if "action.arm" in data:
                action_arm = np.asarray(data["action.arm"], dtype=np.float32)
                action_gripper = (
                    np.asarray(data["action.gripper"], dtype=np.float32)
                    if "action.gripper" in data else None
                )
                if action_gripper is not None:
                    while action_gripper.ndim < action_arm.ndim:
                        action_gripper = action_gripper[..., None]
                    actions = np.concatenate([action_arm, action_gripper], axis=-1)
                else:
                    actions = action_arm

        active_images = {
            target: _parse_image(data[src])
            for src, target in self._ACTIVE_CAMERAS.items()
        }
        ref_image = next(iter(active_images.values()))
        masked_images = {
            target: np.zeros_like(ref_image)
            for _, target in self._MASKED_CAMERAS.items()
        }

        inputs = {
            "state": state,
            "image": {**active_images, **masked_images},
            "image_mask": {
                target: np.True_ for target in active_images
            } | {
                target: np.False_ for target in masked_images
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
class OptimizedFrankaDualInputsV3(transforms.DataTransformFn):
    """Dual-arm v3: 3 cameras, real state (arm+gripper), continuous actions.

    See :class:`OptimizedFrankaSingleInputsV3` for the semantics of
    ``use_next_state_action`` and ``state_as_input``.
    """

    model_type: _model.ModelType
    use_next_state_action: bool = False
    state_as_input: bool = True

    _ACTIVE_CAMERAS = {
        "observation.images.left_side":   "left_side_0_rgb",
        "observation.images.left_wrist":  "left_wrist_0_rgb",
        "observation.images.right_wrist": "right_wrist_0_rgb",
    }
    _MASKED_CAMERAS = {
        "observation.images.right_side": "right_side_0_rgb",
    }

    def __call__(self, data: dict) -> dict:
        arm_val = np.asarray(data["observation.state.arm"], dtype=np.float32)
        gripper_val = np.asarray(data["observation.state.gripper"], dtype=np.float32)
        while gripper_val.ndim < arm_val.ndim:
            gripper_val = gripper_val[..., None]
        state_seq = np.concatenate([arm_val, gripper_val], axis=-1)

        if self.state_as_input:
            state = state_seq[0] if state_seq.ndim >= 2 else state_seq
        else:
            state = np.zeros((1,), dtype=np.float32)

        if self.use_next_state_action:
            if state_seq.ndim < 2:
                raise ValueError(
                    "use_next_state_action=True requires a state sequence with a time "
                    f"dimension (action_horizon+1 frames); got shape {state_seq.shape}."
                )
            actions = state_seq[1:]
        else:
            actions = None
            if "action.arm" in data:
                action_arm = np.asarray(data["action.arm"], dtype=np.float32)
                action_gripper = (
                    np.asarray(data["action.gripper"], dtype=np.float32)
                    if "action.gripper" in data else None
                )
                if action_gripper is not None:
                    while action_gripper.ndim < action_arm.ndim:
                        action_gripper = action_gripper[..., None]
                    actions = np.concatenate([action_arm, action_gripper], axis=-1)
                else:
                    actions = action_arm

        active_images = {
            target: _parse_image(data[src])
            for src, target in self._ACTIVE_CAMERAS.items()
        }
        ref_image = next(iter(active_images.values()))
        masked_images = {
            target: np.zeros_like(ref_image)
            for _, target in self._MASKED_CAMERAS.items()
        }

        inputs = {
            "state": state,
            "image": {**active_images, **masked_images},
            "image_mask": {
                target: np.True_ for target in active_images
            } | {
                target: np.False_ for target in masked_images
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
class OptimizedFrankaSingleRLInputsV3(transforms.DataTransformFn):
    """Single-arm RL v3: 2 cameras, real state (arm+gripper), continuous actions.

    See :class:`OptimizedFrankaSingleInputsV3` for the semantics of
    ``use_next_state_action`` and ``state_as_input``. When
    ``use_next_state_action=True``, the data config fetches ``action_horizon+1``
    frames of ``observation.state.{arm,gripper}``; this transform slices
    ``[0]`` as ``cur_state``, ``[-1]`` as ``next_state`` (still at ``t + H/fps``
    for the RL transition), and ``[1:H+1]`` as the action chunk.
    """

    model_type: _model.ModelType
    resize_height: int = 224
    resize_width: int = 224
    use_next_state_action: bool = False
    state_as_input: bool = True
    progress_reward_weight: float = 1.0
    terminal_bonus: float = 0.0
    failed_episodes: frozenset[int] = frozenset()

    def __call__(self, data: dict) -> dict:
        left_side_pair = np.asarray(data["observation.images.left_side"])
        left_wrist_pair = np.asarray(data["observation.images.left_wrist"])

        cur_left_side = _parse_image(left_side_pair[0])
        cur_left_wrist = _parse_image(left_wrist_pair[0])
        nxt_left_side = np.asarray(image_tools.resize_with_pad(
            _parse_image(left_side_pair[1]), self.resize_height, self.resize_width))
        nxt_left_wrist = np.asarray(image_tools.resize_with_pad(
            _parse_image(left_wrist_pair[1]), self.resize_height, self.resize_width))

        arm_seq = np.asarray(data["observation.state.arm"], dtype=np.float32)
        gripper_seq = np.asarray(data["observation.state.gripper"], dtype=np.float32)
        while gripper_seq.ndim < arm_seq.ndim:
            gripper_seq = gripper_seq[..., None]
        state_seq = np.concatenate([arm_seq, gripper_seq], axis=-1)

        if state_seq.ndim >= 2 and state_seq.shape[0] >= 2:
            cur_state_full = state_seq[0]
            nxt_state = state_seq[-1]
        else:
            cur_state_full = state_seq[0] if state_seq.ndim >= 2 else state_seq
            nxt_state = state_seq[-1] if state_seq.ndim >= 2 else state_seq
        cur_state = cur_state_full if self.state_as_input else np.zeros((1,), dtype=np.float32)

        zero_img = np.zeros_like(cur_left_side)

        inputs: dict = {
            "state": cur_state,
            "image": {
                "left_side_0_rgb": cur_left_side,
                "left_wrist_0_rgb": cur_left_wrist,
                "right_side_0_rgb": zero_img,
                "right_wrist_0_rgb": zero_img,
            },
            "image_mask": {
                "left_side_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_side_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
            "next_state": nxt_state,
            "next_image_left_side": nxt_left_side,
            "next_image_left_wrist": nxt_left_wrist,
        }

        if self.use_next_state_action:
            if state_seq.ndim < 2 or state_seq.shape[0] < 2:
                raise ValueError(
                    "use_next_state_action=True requires an H+1 state sequence; "
                    f"got shape {state_seq.shape}."
                )
            inputs["actions"] = state_seq[1:]
        elif "actions" in data:
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
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        if "progress" in data:
            progress = np.asarray(data["progress"], dtype=np.float32).reshape(-1)
            inputs["progress"] = progress
            delta = float(progress[-1] - progress[0])
            is_pad = data.get("progress_is_pad")
            if is_pad is not None:
                inputs["progress_is_pad"] = np.asarray(is_pad).reshape(-1).astype(bool)
                terminal = bool(inputs["progress_is_pad"][-1])
            else:
                terminal = False
            term = _terminal_bonus_value(data, self.failed_episodes, terminal, self.terminal_bonus)
            inputs["terminal_reward"] = np.float32(term)
            inputs["reward"] = np.float32(self.progress_reward_weight * delta + term)
        else:
            is_pad = data.get("observation.state_is_pad")
            terminal = bool(np.asarray(is_pad).reshape(-1)[1]) if is_pad is not None else False
            inputs["reward"] = np.float32(0.0 if terminal else -1.0)
        inputs["mask"] = np.float32(0.0 if terminal else 1.0)

        return inputs


@dataclasses.dataclass(frozen=True)
class OptimizedFrankaDualRLInputsV3(transforms.DataTransformFn):
    """Dual-arm RL v3: 3 cameras, real state (arm+gripper), continuous actions.

    See :class:`OptimizedFrankaSingleRLInputsV3` for flag semantics.
    """

    model_type: _model.ModelType
    resize_height: int = 224
    resize_width: int = 224
    use_next_state_action: bool = False
    state_as_input: bool = True
    progress_reward_weight: float = 1.0
    terminal_bonus: float = 0.0
    failed_episodes: frozenset[int] = frozenset()

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

        arm_seq = np.asarray(data["observation.state.arm"], dtype=np.float32)
        gripper_seq = np.asarray(data["observation.state.gripper"], dtype=np.float32)
        while gripper_seq.ndim < arm_seq.ndim:
            gripper_seq = gripper_seq[..., None]
        state_seq = np.concatenate([arm_seq, gripper_seq], axis=-1)

        if state_seq.ndim >= 2 and state_seq.shape[0] >= 2:
            cur_state_full = state_seq[0]
            nxt_state = state_seq[-1]
        else:
            cur_state_full = state_seq[0] if state_seq.ndim >= 2 else state_seq
            nxt_state = state_seq[-1] if state_seq.ndim >= 2 else state_seq
        cur_state = cur_state_full if self.state_as_input else np.zeros((1,), dtype=np.float32)

        zero_img = np.zeros_like(cur_left_side)

        inputs: dict = {
            "state": cur_state,
            "image": {
                "left_side_0_rgb": cur_left_side,
                "left_wrist_0_rgb": cur_left_wrist,
                "right_side_0_rgb": zero_img,
                "right_wrist_0_rgb": cur_right_wrist,
            },
            "image_mask": {
                "left_side_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_side_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
            "next_state": nxt_state,
            "next_image_left_side": nxt_left_side,
            "next_image_left_wrist": nxt_left_wrist,
            "next_image_right_wrist": nxt_right_wrist,
        }

        if self.use_next_state_action:
            if state_seq.ndim < 2 or state_seq.shape[0] < 2:
                raise ValueError(
                    "use_next_state_action=True requires an H+1 state sequence; "
                    f"got shape {state_seq.shape}."
                )
            inputs["actions"] = state_seq[1:]
        elif "actions" in data:
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
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        if "progress" in data:
            progress = np.asarray(data["progress"], dtype=np.float32).reshape(-1)
            inputs["progress"] = progress
            delta = float(progress[-1] - progress[0])
            is_pad = data.get("progress_is_pad")
            if is_pad is not None:
                inputs["progress_is_pad"] = np.asarray(is_pad).reshape(-1).astype(bool)
                terminal = bool(inputs["progress_is_pad"][-1])
            else:
                terminal = False
            term = _terminal_bonus_value(data, self.failed_episodes, terminal, self.terminal_bonus)
            inputs["terminal_reward"] = np.float32(term)
            inputs["reward"] = np.float32(self.progress_reward_weight * delta + term)
        else:
            is_pad = data.get("observation.state_is_pad")
            terminal = bool(np.asarray(is_pad).reshape(-1)[1]) if is_pad is not None else False
            inputs["reward"] = np.float32(0.0 if terminal else -1.0)
        inputs["mask"] = np.float32(0.0 if terminal else 1.0)

        return inputs
