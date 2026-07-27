# adapted from openpi
"""Franka single-arm task configuration (aligned with franka_client_sync.py).

Uses :class:`FrankaSingleArmEnv` (7-DOF arm + 1 gripper, action/state dim 8).
Shares the dual-arm ROS2 driver (``/dual_franka_driver``, ``/dual_franka_planner``)
but only controls one side. The ``side`` field (``"left"`` or ``"right"``)
selects which arm joints/gripper are used.

Camera topics subscribe to all 4 cameras (left_side, right_side, left_wrist,
right_wrist) matching the obs keys used by ``franka_env.py``'s
``get_observation()``.
"""

import os

import numpy as np

from configs.task import real_base

try:
    from client.envs.franka_env import FrankaSingleArmEnv
except Exception:
    print("Not importing franka env [module]")


def get_config():
    config = real_base.get_config()

    try:
        config.env = FrankaSingleArmEnv
    except Exception:
        print("Not importing franka env [env]")

    config.env_type = "franka"
    config.env_name = "franka_single_arm"
    config.language_instruction = os.environ.get(
        "FRANKA_LANGUAGE_INSTRUCTION", "complete the task"
    )

    # Which arm this single-arm env controls. Determines joint/gripper names
    # and which wrist camera the policy primarily uses.
    config.side = "left"

    config.action_space = "joint_position"
    config.gripper_action_space = "position"
    config.image_size = (224, 224)
    # Legacy per-action step paths still read this; async chunk training uses
    # exec_publish_hz below for robot-side publishing.
    config.control_hz = 1000

    config.example_action = np.zeros((1, 8))
    config.auto_reset_steps = 1000
    config.enable_hil = False
    config.residual_action_xyzg = False

    # ─── ROS2 topic configuration (shared dual-arm driver) ──────────────
    # Camera obs keys MUST match franka_client_sync.py CAMERA_TOPICS values
    # so that get_observation() can find them via _img("left_side") etc.
    config.camera_topics = {
        "/camera/d435_a/color/image_raw":      "left_side",    # left view
        "/camera/d435_c/color/image_raw":      "right_side",   # right view
        "/camera/d405_b/color/image_rect_raw": "left_wrist",   # left wrist
        "/camera/d405_a/color/image_rect_raw": "right_wrist",  # right wrist
    }

    # Dual-arm driver topics (shared driver, single-arm env only commands one side)
    config.joint_state_topic = "/dual_franka_driver/joint_states"
    config.gripper_state_topic = "/dual_franka_driver/gripper_states"
    config.joint_cmd_topic = "/dual_franka_planner/joint_command"
    config.gripper_cmd_topic = "/dual_franka_planner/gripper_command"

    # Joint/gripper names must match the selected side in the dual-arm driver
    config.arm_joint_names = [f"left_joint{i}" for i in range(1, 8)]
    config.gripper_names = ["left_gripper"]

    # Full dual-arm joint/gripper names for publishing (single-arm env still
    # publishes to ALL joints; the non-controlled side holds current position).
    config.all_arm_joint_names = (
        [f"left_joint{i}" for i in range(1, 8)]
        + [f"right_joint{i}" for i in range(1, 8)]
    )
    config.all_gripper_names = ["left_gripper", "right_gripper"]

    # Gripper scaling (step() always does: max(v, 0) * gripper_action_max)
    config.gripper_state_scale = 252.0
    config.gripper_action_max = 252.0

    # ─── Action-chunk execution (mirror franka_client_sync.py CLI) ──────
    # Used by FrankaEnv.send_action_chunk() for smooth eval/deploy execution:
    # publish `exec_steps` waypoints (from `exec_action_index`) at
    # `exec_publish_hz` Hz. High Hz => controller follows the chunk smoothly and
    # avoids the mid-chunk back-and-forth oscillation. Replicates the reference
    # client defaults: --action_index 0, --execute_steps 20, --frequency 1000.
    config.exec_action_index = 0      # reference --action_index
    config.exec_steps = 20            # reference --execute_steps (clamped to horizon)
    config.exec_publish_hz = 1000.0   # reference --frequency

    # Reset configuration
    config.reset_joints = np.zeros(7)   # placeholder - set actual home position
    config.reset_grippers = np.zeros(1)
    config.reset_joint_tolerance = 0.05
    config.reset_timeout_sec = 15.0

    return config
