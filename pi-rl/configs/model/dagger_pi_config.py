# adapted from openpi
"""Config for BCLearner: imitation-only BC on human-intervention chunks, no critic."""

from configs.model import expo_ft_pi_config


def get_config():
    config = expo_ft_pi_config.get_config()

    config.model_cls = "BCLearner"
    config.pi05_config_name = "expo_pi05_droid_lora_finetune_sft_cartesian_state"
    config.pi05_weight_loader_path = ""
    config.actor_success_only = True

    return config
