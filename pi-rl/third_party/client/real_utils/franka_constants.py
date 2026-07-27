"""Franka robot constants and gripper conversion utilities.

These constants are aligned with ``franka_client_sync.py`` and shared across
the Franka env modules (ROS2 node, env base, concrete envs).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Default image size (W, H) used by ``resize_with_pad`` when the task
# config does not override it.
TARGET_IMAGE_SIZE: Tuple[int, int] = (224, 224)

# Legacy constants kept for reference (not used in manual-reset mode).
DEFAULT_RESET_JOINT_TOLERANCE: float = 0.05
DEFAULT_RESET_TIMEOUT: float = 30.0

# ─── Gripper constants (aligned with franka_client_sync.py) ───────────

# Gripper physical scale: driver feedback is 0~252 mm; normalized to
# [0=closed, 1=open] for the VLA
GRIPPER_STATE_SCALE: float = 252.0

# Action output mode:
#   "binary" = threshold to 0/GRIPPER_ACTION_MAX
#   "raw"    = send the raw policy value directly
GRIPPER_MODE: str = "binary"
GRIPPER_ACTION_MAX: float = 252.0
GRIPPER_ACTION_THRESHOLD: float = 0.5


def driver_gripper_to_policy(value: float) -> float:
    """Driver mm / raw -> VLA [0=closed, 1=open] (normalized by the 252 mm full stroke)"""
    return float(np.clip(value / GRIPPER_STATE_SCALE, 0.0, 1.0))


def policy_gripper_to_cmd(value: float) -> float:
    """Policy gripper output -> robot gripper command.

    The policy output, after denormalization via norm stats, is already in
    mm scale (0~252); just clip it to the valid range — no need to multiply
    by GRIPPER_ACTION_MAX again.
    """
    return float(np.clip(value, 0.0, GRIPPER_ACTION_MAX))
