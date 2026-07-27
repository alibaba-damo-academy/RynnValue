# adapted from openpi
from configs.model import sac_config


def get_config():
    config = sac_config.get_config()

    config.model_cls = "EXPOLearner"

    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm = True

    config.N = 8
    config.n_edit_samples = 8

    config.adjust_target_entropy = False
    config.entropy_scale = 1.0
    config.edit_scale = 0.2
    config.actor_drop = 0.0
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.latent_dim_image = 512
    config.latent_dim_state = 64
    config.include_state = True
    config.encoder_stage_sizes = (3, 4, 6, 3)
    config.encoder_num_filters = 64
    config.hidden_dims = (256, 256, 256)

    config.encode_batch_split = 1
    config.batch_split = 1

    config.use_pi05 = True
    config.pi05_config_name = "expo_pi05_franka_lora_sft"
    config.pi05_resize_size = 224
    config.freeze_pi05_encoder = True
    config.freeze_critic_encoder = False  # if True, encoder is frozen for Q (only extract embeddings)

    # NOTE: these MUST point at the Franka SFT checkpoint's bundled assets so that
    # Normalize/Unnormalize use the SAME norm_stats as serve_policy. Pointing them
    # at libero (or leaving them unset) will silently mis-normalize and break the
    # alignment with franka_client_sync.py. norm stats are loaded from
    # assets_dir/asset_id (i.e. <franka_sft_ckpt>/assets/<asset_id>/norm_stats.json).
    config.pi05_weight_loader_path = ""  # TODO: Franka SFT checkpoint params path
    config.pi05_assets_dir = ""          # TODO: <franka_sft_ckpt>/assets
    config.pi05_asset_id = ""            # TODO: Franka SFT asset_id (norm_stats subdir)
    config.actor_success_only = True
    config.use_full_augmentation = True  # False = only crop (no rotate/color jitter)

    return config
