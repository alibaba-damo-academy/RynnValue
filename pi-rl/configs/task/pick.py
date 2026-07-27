# adapted from openpi
"""Pick task: DROID real robot, pick up the cube."""

import numpy as np

from configs.task import real_base

try:
    from client.envs.droid_env import PickBlocksEnv
except Exception:
    print("Not importing droid env [module]")


def get_config():
    config = real_base.get_config()

    try:
        config.env = PickBlocksEnv
    except Exception:
        print("Not importing droid env [env]")

    config.env_name = "pick"
    config.language_instruction = "pick up the cube"

    delta = 0.2
    config.bounds = np.array([[0.48-delta, 0.48+delta], [0.007-delta, 0.007+delta], [0.1, 0.45]])
    config.reset_joints = np.array([
        0.005655035376548767, 0.10885079205036163, 0.009642823599278927, -2.44547176361084, -0.021355044096708298, 2.542558431625366, -0.04585753753781319
    ])

    config.auto_reset_steps = 80

    # Pick task: no rotation needed; residual only on xyz and gripper
    config.residual_action_xyzg = True

    # randomizing drop position after success
    config.success_reset_randomize_magnitude = 0.06  # ±0.06 for x,y when randomizing drop position after success

    # spacemouse scaling
    config.collect_max_lin_vel = 0.5
    config.collect_max_rot_vel = 0.1

    return config
