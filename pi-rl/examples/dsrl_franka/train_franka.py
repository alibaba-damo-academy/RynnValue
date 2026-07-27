# adapted from openpi
"""DSRL Franka online RL training with SAC + pi05.

This script orchestrates the full training pipeline:
1. Environment initialization (real via WebSocket or fake for testing)
2. Pi05 policy loading (with norm_stats fallback mechanism)
3. SAC learner initialization (dynamic state_dim based on arm_mode)
4. Replay buffer setup
5. Training loop delegation to train_utils_franka
"""

import os

# Tell XLA to use Triton GEMM — improves steps/sec by ~30% on some GPUs.
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import logging
import tempfile
from functools import partial
from pathlib import Path

import jax
import numpy as np
import tensorflow as tf
import wandb
from gym.spaces import Box, Dict
from jax.experimental.compilation_cache import compilation_cache
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.general_utils import add_batch_dim

from examples.dsrl_franka.env.env_client import EnvClientWrapper
from examples.dsrl_franka.env.fake_env_wrapper import FakeEnvLocalWrapper
from examples.dsrl_franka.train_utils_franka import trajwise_alternating_training_loop

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import checkpoints as openpi_checkpoints
from openpi.training import config as openpi_config

logger = logging.getLogger(__name__)

# Initialize JAX compilation cache for faster restarts.
_home = os.environ.get("HOME", "/tmp")
compilation_cache.initialize_cache(os.path.join(_home, "jax_compilation_cache"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AttrDict(dict):
    """Dict subclass that supports attribute-style access.

    Required because jaxrl2's ``PixelSACLearner`` and the existing
    ``train_utils_franka`` code access variant fields via both
    ``variant['key']`` and ``variant.key``.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


def shard_batch(batch, sharding):
    """Shard a pytree batch across devices along the leading dimension."""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class _DummyEnv:
    """Defines observation/action spaces for SAC (without a real gym env).

    Pixels are stacked as (H, W, 3*num_cameras, 1) — Franka uses base + left
    wrist (2 cameras → 6 channels). Low-dim state is the joint+gripper vector.
    """

    def __init__(self, variant: dict):
        resize = int(variant.get("resize_image", 64))
        num_cameras = int(variant.get("num_cameras", 2))
        add_states = int(variant.get("add_states", 1))
        state_dim = 16 if variant["arm_mode"] == "dual" else 8
        chunk_size = int(variant.get("agent_chunk_size", 1))

        self.image_shape = (resize, resize, 3 * num_cameras, 1)
        obs_dict = {"pixels": Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)}
        if add_states:
            obs_dict["state"] = Box(
                low=-np.inf, high=np.inf, shape=(state_dim, 1), dtype=np.float32
            )
        self.observation_space = Dict(obs_dict)
        # SAC actor predicts noise vectors in pi0's latent space (dim=32).
        self.action_space = Box(low=-1, high=1, shape=(chunk_size, 32), dtype=np.float32)


# ---------------------------------------------------------------------------
# Component initialization functions
# ---------------------------------------------------------------------------


def load_pi05_policy(variant: dict):
    """Load the Pi05 policy model.

    Weight loading priority:
    1. Environment variables FRANKA_SFT_CKPT_BASE + FRANKA_SFT_TASK
    2. --pi0_checkpoint_dir argument
    3. Fallback to GCS path (gs://openpi-assets/checkpoints/pi05_libero)

    Includes norm_stats fallback mechanism:
    - Tries to load from checkpoint_dir/assets/{franka_norm_stats_asset_id}
    """
    # Resolve checkpoint directory
    checkpoint_dir_raw = variant["pi0_checkpoint_dir"]
    checkpoint_dir = download.maybe_download(checkpoint_dir_raw)
    logger.info("Pi05 checkpoint resolved to: %s", checkpoint_dir)

    # Use the policy config specified by --policy_config (default: pi05_franka_single_optimized)
    policy_config_name = variant.get("policy_config", "pi05_franka_single_optimized")
    config = openpi_config.get_config(policy_config_name)
    logger.info("Using OpenPI config: %s (action_horizon=%d)", policy_config_name, config.model.action_horizon)

    # Override repo_id and disable auto_repack to avoid HuggingFace access during inference.
    # The actual dataset is not accessed — we only need norm_stats from the checkpoint.
    if hasattr(config.data, 'repo_id') and not config.data.repo_id:
        import dataclasses as _dc
        config = _dc.replace(config, data=_dc.replace(config.data, repo_id=None, auto_repack=False))

    # --- norm_stats fallback ---
    # If the checkpoint lacks the data-config's default asset_id, fall back to
    # a compatible norm_stats so the policy can still load.
    norm_stats_override = None
    fallback_asset_id = variant.get("franka_norm_stats_asset_id", "libero") or None
    if fallback_asset_id:
        candidate = Path(checkpoint_dir) / "assets" / fallback_asset_id / "norm_stats.json"
        if candidate.exists():
            norm_stats_override = openpi_checkpoints.load_norm_stats(
                Path(checkpoint_dir) / "assets", fallback_asset_id
            )
            logger.info(
                "[franka] Using fallback norm_stats from assets/%s", fallback_asset_id
            )

    pi0_policy = policy_config.create_trained_policy(
        config, checkpoint_dir, norm_stats=norm_stats_override
    )
    logger.info("Loaded pi05 policy from %s", checkpoint_dir)
    return pi0_policy


def init_sac_learner(variant: dict, sample_obs: dict, sample_action: np.ndarray):
    """Initialize PixelSACLearner.

    Uses jaxrl2's PixelSACLearner with hyperparameters injected by the
    launcher into the variant dict.
    """
    # Build kwargs from variant's SAC hyperparameters
    kwargs = dict(
        actor_lr=variant["actor_lr"],
        critic_lr=variant["critic_lr"],
        temp_lr=variant["temp_lr"],
        hidden_dims=variant["hidden_dims"],
        cnn_features=variant["cnn_features"],
        cnn_strides=variant["cnn_strides"],
        cnn_padding=variant["cnn_padding"],
        latent_dim=variant["latent_dim"],
        discount=variant["discount"],
        tau=variant["tau"],
        critic_reduction=variant["critic_reduction"],
        dropout_rate=variant.get("dropout_rate", 0.0),
        aug_next=variant.get("aug_next", 1),
        use_bottleneck=variant.get("use_bottleneck", True),
        encoder_type=variant.get("encoder_type", "small"),
        encoder_norm=variant.get("encoder_norm", "group"),
        use_spatial_softmax=variant.get("use_spatial_softmax", True),
        softmax_temperature=variant.get("softmax_temperature", -1),
        target_entropy=variant.get("target_entropy", "auto"),
        num_qs=variant.get("num_qs", 10),
        num_cameras=variant.get("num_cameras", 2),
    )

    agent = PixelSACLearner(variant["seed"], sample_obs, sample_action, **kwargs)
    logger.info("SAC learner initialized (seed=%d).", variant["seed"])
    return agent


def create_replay_buffer(variant: dict, dummy_env: _DummyEnv):
    """Create a jaxrl2 ReplayBuffer sized for the training run."""
    max_steps = int(variant.get("max_steps", 500_000))
    utd = int(variant.get("multi_grad_step", variant.get("utd_ratio", 20)))
    buffer_size = max(max_steps // utd, 10_000)

    replay_buffer = ReplayBuffer(
        dummy_env.observation_space, dummy_env.action_space, int(buffer_size)
    )
    replay_buffer.seed(variant["seed"])
    logger.info("Replay buffer created (capacity=%d).", buffer_size)
    return replay_buffer


def create_environment(variant: dict):
    """Create either the real WebSocket env or a local fake env.

    Returns:
        An env object with the same interface (reset, step_chunk,
        get_observation, get_info_for_step, close).
    """
    arm_mode = variant["arm_mode"]
    side = variant.get("side", "left")

    if variant.get("fake_env", False):
        env = FakeEnvLocalWrapper(
            arm_mode=arm_mode,
            side=side,
            task_description=variant["task_description"],
            max_timesteps=int(variant["franka_max_timesteps"]),
        )
        logger.info("Using FakeEnvLocalWrapper (no hardware).")
    else:
        env_config = {
            "language_instruction": variant["task_description"],
            "image_size": (224, 224),
            "auto_reset_steps": int(variant["franka_max_timesteps"]),
            "env_usage": "train",
        }
        # Pass video_dir if specified
        video_dir = variant.get("video_dir", "")
        if video_dir:
            env_config["video_dir"] = video_dir

        env = EnvClientWrapper(
            host=variant.get("client_host", "localhost"),
            port=int(variant.get("client_port", 8102)),
            arm_mode=arm_mode,
            side=side,
            env_config=env_config,
        )
        logger.info(
            "Using EnvClientWrapper (host=%s, port=%d).",
            variant.get("client_host", "localhost"),
            variant.get("client_port", 8102),
        )
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(variant: dict) -> None:
    """Main training entry point.

    Called by launch_train_franka.py with the variant dict containing all
    CLI arguments + injected SAC hyperparameters.
    """
    # ── 0. Ensure variant supports attribute access ──────────────────
    if not isinstance(variant, _AttrDict):
        variant = _AttrDict(variant)

    # ── 1. Basic setup ────────────────────────────────────────────────
    seed = int(variant["seed"])
    arm_mode = variant["arm_mode"]
    state_dim = 16 if arm_mode == "dual" else 8

    # Inject derived fields into variant for downstream use
    variant["franka_state_dim"] = state_dim
    variant["max_timesteps"] = int(variant.get("franka_max_timesteps", 500))
    variant["pi0_action_horizon"] = int(variant.get("pi0_action_horizon", 10))
    variant["env_max_reward"] = 1  # sparse 0/1 success
    variant.setdefault("add_states", 1)
    variant.setdefault("resize_image", 64)
    variant.setdefault("query_freq", variant["pi0_action_horizon"])
    variant.setdefault("multi_grad_step", variant.get("utd_ratio", 20))
    variant.setdefault("max_steps", 500_000)
    variant.setdefault("batch_size", 256)
    variant.setdefault("log_interval", 500)
    variant.setdefault("start_online_updates", 500)
    variant.setdefault("agent_chunk_size", 1)

    # JAX devices
    devices = jax.local_devices()
    num_devices = len(devices)
    logger.info("JAX devices: %d, batch_size: %d", num_devices, variant["batch_size"])
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # Prevent TensorFlow from grabbing GPU memory
    tf.config.set_visible_devices([], "GPU")

    # ── 2. Output directory ───────────────────────────────────────────
    exp_dir = os.environ.get("EXP", "./experiments")
    run_name = variant.get("wandb_run_name", "") or f"dsrl_franka_s{seed}"
    outputdir = os.path.abspath(os.path.join(exp_dir, run_name))
    os.makedirs(outputdir, exist_ok=True)
    variant["outputdir"] = outputdir
    logger.info("Output directory: %s", outputdir)

    # ── 3. WandB initialization ───────────────────────────────────────
    # Filter out non-serializable objects before passing to wandb config
    wandb_config = {k: v for k, v in variant.items() if isinstance(v, (int, float, str, bool, tuple, list, dict, type(None)))}
    wandb.init(
        project=variant.get("wandb_project", "dsrl_franka"),
        name=variant.get("wandb_run_name") or None,
        config=wandb_config,
    )

    # ── 4. SAC Learner initialization (before pi0.5 to avoid OOM in cuSolver)
    dummy_env = _DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    logger.info("Sample obs shapes: %s", [(k, v.shape) for k, v in sample_obs.items()])
    logger.info("Sample action shape: %s", sample_action.shape)

    agent = init_sac_learner(variant, sample_obs, sample_action)

    # ── 5. Replay Buffer ──────────────────────────────────────────────
    replay_buffer = create_replay_buffer(variant, dummy_env)

    # ── 6. Environment initialization ─────────────────────────────────
    env = create_environment(variant)
    # Reuse same env for eval — FrankaEnv.reset() blocks for human
    # repositioning, so a separate eval env is impractical.
    eval_env = env

    # ── 7. Pi05 model loading ─────────────────────────────────────────
    pi0_policy = load_pi05_policy(variant)

    # ── 8. WandB logger (jaxrl2 compatible) ───────────────────────────
    # We use jaxrl2's WandBLogger so train_utils_franka can log via
    # wandb_logger.log({...}, step=i) with the expected interface.
    from jaxrl2.utils.wandb_logger import WandBLogger

    wandb_output_dir = tempfile.mkdtemp()
    prefix = run_name.split("_")[0] if run_name else "franka"
    variant.setdefault("prefix", prefix)
    wandb_logger = WandBLogger(
        True,
        variant,
        variant.get("wandb_project", "dsrl_franka"),
        experiment_id=run_name,
        output_dir=wandb_output_dir,
        group_name=prefix,
    )

    # ── 9. Start training loop ────────────────────────────────────────
    trajwise_alternating_training_loop(
        variant,
        agent,
        env,
        eval_env,
        replay_buffer,
        replay_buffer,
        wandb_logger,
        shard_fn=shard_fn,
        agent_dp=pi0_policy,
    )
