# adapted from openpi
"""Light task 2: DROID real robot. Success when yellow plug is removed (no yellow in ROI)."""

import numpy as np

from configs.task import real_base

try:
    from client.envs.droid_env import Light2Env
except Exception:
    print("Not importing droid env [module]")


def get_config():
    config = real_base.get_config()

    try:
        config.env = Light2Env
    except Exception:
        print("Not importing droid env [env]")

    config.env_name = "light2"
    config.language_instruction = "plug in the cable"

    config.bounds = np.array([[0.3, 0.5], [-0.3, 0.], [0.1, 0.3]])

    config.reset_joints = np.array([
        0.23025356233119965, 0.3106425404548645, -0.044641271233558655, -2.477023124694824, -0.0829055905342102, 2.7981555461883545, 0.31343474984169006
    ])

    config.auto_reset_steps = 90

    # REPLACE with your own ZED camera serials (see README "DROID Setup").
    config.detector_image_key = "27904255_left"
    config.side_camera_id = "29838012_left"
    config.record_camera = "27904255_left" # extra camera for recording

    # detecting params
    config.max_yellow_pixels = 400
    config.consecutive_frames = 5
    config.detect_size = 384

    # pre-reset joint positions for randomizing drop position after success
    config.pre_reset_joints = np.array([
        0.25788426399230957, 0.46801185607910156, -0.03795424848794937, -2.379973888397217, -0.06309568136930466, 2.870250701904297, 0.31227147579193115
    ])
    config.success_reset_randomize_magnitude = 0.025

    config.residual_action_xyzg = False

    config.collect_max_lin_vel = 0.6
    config.collect_max_rot_vel = 0.01

    return config
