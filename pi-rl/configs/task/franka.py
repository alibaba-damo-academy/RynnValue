# adapted from openpi
"""Franka dual-arm task configuration."""

import os

import numpy as np

from configs.task import real_base

try:
    from client.envs.franka_env import FrankaDualArmEnv
except Exception:
    print("Not importing franka env [module]")


def get_config():
    config = real_base.get_config()

    try:
        config.env = FrankaDualArmEnv
    except Exception:
        print("Not importing franka env [env]")

    config.env_type = "franka"
    config.env_name = "franka_dual_arm"
    config.language_instruction = os.environ.get(
        "FRANKA_LANGUAGE_INSTRUCTION", "complete the task"
    )

    config.action_space = "joint_position"
    config.gripper_action_space = "position"
    config.image_size = (224, 224)
    config.control_hz = 10

    config.example_action = np.zeros((1, 16))
    config.auto_reset_steps = 600
    config.enable_hil = False
    config.residual_action_xyzg = False

    # ROS2 topic configuration
    # Camera obs keys MUST match franka_client_sync.py CAMERA_TOPICS values
    # so that get_observation() can find them via _img("left_side") etc.
    config.camera_topics = {
        "/camera/d435_a/color/image_raw":      "left_side",    # left view
        "/camera/d435_c/color/image_raw":      "right_side",   # right view
        "/camera/d405_b/color/image_rect_raw": "left_wrist",   # left wrist
        "/camera/d405_a/color/image_rect_raw": "right_wrist",  # right wrist
    }
    config.joint_state_topic = "/dual_franka_driver/joint_states"
    config.gripper_state_topic = "/dual_franka_driver/gripper_states"
    config.joint_cmd_topic = "/dual_franka_planner/joint_command"
    config.gripper_cmd_topic = "/dual_franka_planner/gripper_command"
    config.left_arm_joint_names = [f"left_joint{i}" for i in range(1, 8)]
    config.right_arm_joint_names = [f"right_joint{i}" for i in range(1, 8)]
    config.left_gripper_names = ["left_gripper"]
    config.right_gripper_names = ["right_gripper"]

    # Reset configuration
    config.reset_joints = np.zeros(14)  # placeholder - user will set actual home position
    config.reset_grippers = np.zeros(2)
    config.reset_joint_tolerance = 0.05
    config.reset_timeout_sec = 15.0

    # ─── Action-chunk execution (mirror franka_client_sync.py CLI) ──────
    # Used by FrankaEnv.send_action_chunk() for smooth eval/deploy execution.
    # Replicates the reference client defaults: --action_index 0,
    # --execute_steps 20, --frequency 1000.
    config.exec_action_index = 0      # reference --action_index
    config.exec_steps = 20            # reference --execute_steps (clamped to horizon)
    config.exec_publish_hz = 1000.0   # reference --frequency

    return config
