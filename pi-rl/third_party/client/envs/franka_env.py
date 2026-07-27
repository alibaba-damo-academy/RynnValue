"""Franka gym-like environments with ROS2 integration for EXPO-FT.

This module provides gym-like wrappers around Franka robots that communicate
via ROS2 topics. They are designed to be served by ``client/run_client.py``.

Two concrete classes are provided:

- :class:`FrankaDualArmEnv` (also exported as ``FrankaEnv`` for backward
  compatibility) -- two 7-DOF arms + two grippers, action/state dim ``16``.
- :class:`FrankaSingleArmEnv` -- a single 7-DOF arm + one gripper,
  action/state dim ``8``.

Implementation details (ROS2 node, base env machinery, image/gripper utils)
live in ``client/real_utils/franka_*.py`` modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Re-export constants and utilities for backward compatibility.
from client.real_utils.franka_constants import (
    DEFAULT_RESET_JOINT_TOLERANCE,
    DEFAULT_RESET_TIMEOUT,
    GRIPPER_ACTION_MAX,
    GRIPPER_MODE,
    GRIPPER_STATE_SCALE,
    TARGET_IMAGE_SIZE,
    driver_gripper_to_policy,
    policy_gripper_to_cmd,
)
from client.real_utils.franka_env_base import _FrankaEnvBase
from client.real_utils.franka_images import resize_with_pad, ros_image_to_numpy
from client.real_utils.franka_ros2_node import FrankaROS2Node


# ─── Concrete envs ─────────────────────────────────────────────────────

class FrankaDualArmEnv(_FrankaEnvBase):
    """Dual-arm Franka env: 14 joints + 2 grippers (action/state dim 16)."""

    ARM_MODE = "dual"
    JOINT_DIM = 14
    GRIPPER_DIM = 2

    def __init__(
        self,
        camera_topics: Optional[Dict[str, str]] = None,
        joint_state_topic: str = "",
        gripper_state_topic: str = "",
        joint_cmd_topic: str = "",
        gripper_cmd_topic: str = "",
        left_arm_joint_names: Optional[List[str]] = None,
        right_arm_joint_names: Optional[List[str]] = None,
        left_gripper_names: Optional[List[str]] = None,
        right_gripper_names: Optional[List[str]] = None,
        reset_joints: Any = None,
        reset_grippers: Any = None,
        language_instruction: str = "",
        image_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
        auto_reset_steps: int = 0,
        reset_joint_tolerance: float = DEFAULT_RESET_JOINT_TOLERANCE,
        reset_timeout_sec: float = DEFAULT_RESET_TIMEOUT,
        video_dir: Optional[str] = None,
        env_usage: str = "train",
        wrist_image_key: str = "left_wrist_image",
        gripper_mode: str = GRIPPER_MODE,
        gripper_state_scale: float = GRIPPER_STATE_SCALE,
        gripper_action_max: float = GRIPPER_ACTION_MAX,
        **kwargs: Any,
    ) -> None:
        arm_joint_names = list(left_arm_joint_names or []) + list(right_arm_joint_names or [])
        gripper_names = list(left_gripper_names or []) + list(right_gripper_names or [])

        super().__init__(
            camera_topics=camera_topics,
            joint_state_topic=joint_state_topic,
            gripper_state_topic=gripper_state_topic,
            joint_cmd_topic=joint_cmd_topic,
            gripper_cmd_topic=gripper_cmd_topic,
            arm_joint_names=arm_joint_names,
            gripper_names=gripper_names,
            wrist_image_key=wrist_image_key,
            reset_joints=reset_joints,
            reset_grippers=reset_grippers,
            language_instruction=language_instruction,
            image_size=image_size,
            auto_reset_steps=auto_reset_steps,
            reset_joint_tolerance=reset_joint_tolerance,
            reset_timeout_sec=reset_timeout_sec,
            video_dir=video_dir,
            env_usage=env_usage,
            gripper_mode=gripper_mode,
            gripper_state_scale=gripper_state_scale,
            gripper_action_max=gripper_action_max,
            extra_kwargs=kwargs,
        )


class FrankaSingleArmEnv(_FrankaEnvBase):
    """Single-arm Franka env: 7 joints + 1 gripper (action/state dim 8).

    Parameters
    ----------
    side:
        ``"left"`` or ``"right"``. Selects which wrist camera is exposed as
        the observation's ``"wrist_image"``.
    arm_joint_names:
        7 joint names for the single arm.
    gripper_names:
        1 gripper name.
    all_arm_joint_names:
        Full 14 joint names for dual-arm publishing (non-controlled side holds).
    all_gripper_names:
        Full 2 gripper names for dual-arm publishing.
    """

    ARM_MODE = "single"
    JOINT_DIM = 7
    GRIPPER_DIM = 1

    def __init__(
        self,
        camera_topics: Optional[Dict[str, str]] = None,
        joint_state_topic: str = "",
        gripper_state_topic: str = "",
        joint_cmd_topic: str = "",
        gripper_cmd_topic: str = "",
        side: str = "left",
        arm_joint_names: Optional[List[str]] = None,
        gripper_names: Optional[List[str]] = None,
        all_arm_joint_names: Optional[List[str]] = None,
        all_gripper_names: Optional[List[str]] = None,
        reset_joints: Any = None,
        reset_grippers: Any = None,
        language_instruction: str = "",
        image_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
        auto_reset_steps: int = 0,
        reset_joint_tolerance: float = DEFAULT_RESET_JOINT_TOLERANCE,
        reset_timeout_sec: float = DEFAULT_RESET_TIMEOUT,
        video_dir: Optional[str] = None,
        env_usage: str = "train",
        gripper_mode: str = GRIPPER_MODE,
        gripper_state_scale: float = GRIPPER_STATE_SCALE,
        gripper_action_max: float = GRIPPER_ACTION_MAX,
        **kwargs: Any,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side: str = side

        super().__init__(
            camera_topics=camera_topics,
            joint_state_topic=joint_state_topic,
            gripper_state_topic=gripper_state_topic,
            joint_cmd_topic=joint_cmd_topic,
            gripper_cmd_topic=gripper_cmd_topic,
            arm_joint_names=list(arm_joint_names or []),
            gripper_names=list(gripper_names or []),
            wrist_image_key=f"{side}_wrist_image",
            all_arm_joint_names=all_arm_joint_names,
            all_gripper_names=all_gripper_names,
            reset_joints=reset_joints,
            reset_grippers=reset_grippers,
            language_instruction=language_instruction,
            image_size=image_size,
            auto_reset_steps=auto_reset_steps,
            reset_joint_tolerance=reset_joint_tolerance,
            reset_timeout_sec=reset_timeout_sec,
            video_dir=video_dir,
            env_usage=env_usage,
            gripper_mode=gripper_mode,
            gripper_state_scale=gripper_state_scale,
            gripper_action_max=gripper_action_max,
            extra_kwargs=kwargs,
        )


# Backward-compatible alias: the original ``FrankaEnv`` was the dual-arm one.
FrankaEnv = FrankaDualArmEnv


__all__ = [
    "FrankaEnv",
    "FrankaDualArmEnv",
    "FrankaSingleArmEnv",
    "FrankaROS2Node",
    "ros_image_to_numpy",
    "resize_with_pad",
    "TARGET_IMAGE_SIZE",
    "GRIPPER_STATE_SCALE",
    "GRIPPER_ACTION_MAX",
    "driver_gripper_to_policy",
    "policy_gripper_to_cmd",
]
