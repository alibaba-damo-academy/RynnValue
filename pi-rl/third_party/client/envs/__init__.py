# Environment registry -- guarded imports so that missing deps on a given
# machine (e.g. no ROS2, no DROID polymetis) don't prevent importing the
# package on the training machine.

try:
    from client.envs.franka_env import FrankaEnv
except Exception:
    pass

try:
    from client.envs.droid_env import DroidEnv, PickBlocksEnv, Light2Env
except Exception:
    pass

try:
    from client.envs.libero_env import LiberoEnv
except Exception:
    pass
