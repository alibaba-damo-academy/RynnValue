# adapted from openpi
"""LIBERO simulation task config."""

from configs.task import libero_base

try:
    from client.envs.libero_env import LiberoEnv
except Exception:
    print("Not importing libero env [module]")


def get_config():
    config = libero_base.get_config()

    try:
        config.env = LiberoEnv
    except Exception:
        print("Not importing libero env [env]")

    config.env_name = "libero"

    config.task_suite_name = "libero_spatial"
    config.task_id = 0
    config.auto_reset_steps = 300

    # Resolved at env creation time from the benchmark task description.
    config.language_instruction = None

    return config
