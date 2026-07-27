# adapted from openpi
import ml_collections
import numpy as np


def get_config():
    config = ml_collections.ConfigDict()

    config.env_type = "sim"

    config.action_space = "cartesian_velocity"
    config.gripper_action_space = "velocity"

    config.bounds = None
    config.reset_joints = None
    config.reset_random = False
    config.randomize_low = np.array([0.0, 0.0, 0.0, 0, 0, 0, 0])
    config.randomize_high = np.array([0.0, 0.0, 0.0, 0, 0, 0, 0])

    config.image_size = None
    config.env_resolution = 256
    config.control_hz = 10

    config.example_action = np.array([[0., 0., 0., 0., 0., 0., 0.]])

    config.residual_action_xyzg = False

    config.task_suite_name = "libero_spatial"
    config.task_id = 0
    config.seed = 7
    config.num_steps_wait = 10

    return config
