# adapted from openpi
from ml_collections.config_dict import config_dict

from configs.model import td_config


def get_config():
    config = td_config.get_config()

    config.temp_lr = 3e-4

    config.init_temperature = 1.0
    config.target_entropy = config_dict.placeholder(float)

    config.critic_weight_decay = config_dict.placeholder(float)

    return config
