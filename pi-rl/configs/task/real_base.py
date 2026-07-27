# adapted from openpi
import ml_collections
import numpy as np

def get_config():
    config = ml_collections.ConfigDict()

    config.env_type = "droid"
    
    # DROID action stream names used when loading and converting datasets.
    config.action_space = "cartesian_velocity"
    config.gripper_action_space = "velocity"

    # Cartesian workspace bounds and nominal joint reset pose; override per task.
    config.bounds = None
    config.reset_joints = None
    
    # Robot initial position randomization around reset_joints.
    config.reset_random = False
    config.randomize_low = np.array([0.0, 0.0, 0.0, 0, 0, 0, 0])
    config.randomize_high = np.array([0.0, 0.0, 0.0, 0, 0, 0, 0])

    # REPLACE with your own ZED camera serials (see README "DROID Setup").
    config.side_camera_id = "27904255_left"
    config.wrist_camera_id = "12841040_left"
    
    # Observation image resize and control loop frequency.
    config.image_size = (180, 320)
    config.control_hz = 10

    # No depth, left view only for faster camera read.
    config.camera_kwargs = {
        "hand_camera": {"image": True, "depth": False, "left_only": True},
        "static_camera": {"image": True, "depth": False, "left_only": True},
    }

    # Action-shaped zero used for env/replay-buffer initialization.
    config.example_action = np.array([[ 0., 0., 0., 0., 0., 0., 0.]])

    # Residual policy: when True, only apply residual to xyz and gripper (not rotation). Use for tasks that don't need rotation.
    config.residual_action_xyzg = False

    return config

