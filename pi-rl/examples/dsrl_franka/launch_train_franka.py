#!/usr/bin/env python3
# adapted from openpi
"""CLI launcher for DSRL Franka online RL training."""

import argparse
import logging
import os
import sys

# Ensure the project root is on sys.path so absolute imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from examples.dsrl_franka.train_franka import main


def _default_checkpoint_dir() -> str:
    """Build default pi0 checkpoint path from env vars, with fallback."""
    base = os.environ.get("FRANKA_SFT_CKPT_BASE", "")
    task = os.environ.get("FRANKA_SFT_TASK", "")
    if base and task:
        return os.path.join(base, task)
    if base:
        return base
    return "gs://openpi-assets/checkpoints/pi05_libero"


def parse_args():
    parser = argparse.ArgumentParser(description="DSRL Franka Training")

    # ---- Arm / env mode ----
    parser.add_argument("--arm_mode", choices=["dual", "single"], default="dual",
                        help="Determines FrankaEnv type (dual-arm or single-arm).")
    parser.add_argument("--side", choices=["left", "right"], default="left",
                        help="Which arm to use in single-arm mode.")
    parser.add_argument("--update_type", choices=["episode", "step"], default="episode",
                        help="Gradient update trigger: per episode or per step.")
    parser.add_argument("--utd_ratio", type=int, default=20,
                        help="Update-to-Data ratio.")
    parser.add_argument("--publish_hz", type=float, default=10.0,
                        help="Action chunk execution frequency (Hz).")

    # ---- Environment connection (EXPO-FT style) ----
    parser.add_argument("--client_host", default="localhost",
                        help="Host for Franka env server (robot machine IP).")
    parser.add_argument("--client_port", type=int, default=8102,
                        help="Port for Franka env server.")
    parser.add_argument("--fake_env", action="store_true",
                        help="Use local fake environment for testing (no WebSocket server needed).")

    # ---- Pi05 model ----
    parser.add_argument("--policy_config", default="pi05_franka_single_optimized",
                        help="OpenPI config name for the pi0.5 model (e.g. pi05_franka_single_optimized, pi05_franka_single_delta).")
    parser.add_argument("--pi0_checkpoint_dir", default=_default_checkpoint_dir(),
                        help="Pi0 checkpoint dir to load.")
    parser.add_argument("--pi0_action_horizon", type=int, default=10,
                        help="Action horizon used by the loaded pi0 model.")
    parser.add_argument("--franka_norm_stats_asset_id", default="libero",
                        help="Norm stats asset_id fallback (e.g. 'libero'). Empty to skip override.")

    # ---- Training ----
    parser.add_argument("--max_episodes", type=int, default=1000,
                        help="Total number of training episodes.")
    parser.add_argument("--franka_max_timesteps", type=int, default=500,
                        help="Hard cap on rollout length per episode.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="SAC training mini-batch size.")
    parser.add_argument("--query_freq", type=int, default=10,
                        help="Action chunk query frequency (execute steps per inference).")
    parser.add_argument("--max_steps", type=int, default=500000,
                        help="Maximum total gradient steps.")
    parser.add_argument("--log_interval", type=int, default=500,
                        help="Logging interval (gradient steps).")
    parser.add_argument("--start_online_updates", type=int, default=100,
                        help="Minimum buffer size before starting online gradient updates.")
    parser.add_argument("--resize_image", type=int, default=64,
                        help="SAC encoder input image size (pixels).")
    parser.add_argument("--buffer_capacity", type=int, default=100000,
                        help="Replay buffer capacity.")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Evaluate every N episodes.")
    parser.add_argument("--eval_episodes", type=int, default=3,
                        help="Number of episodes per evaluation.")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                        help="Save checkpoint every N episodes.")
    parser.add_argument("--checkpoint_dir", default="./checkpoints",
                        help="Directory to save checkpoints.")
    parser.add_argument("--resume_checkpoint", default="",
                        help="Path to checkpoint for resuming training.")
    parser.add_argument("--noise_episodes", type=int, default=5,
                        help="Number of initial episodes using pure Gaussian noise.")
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="Std of Gaussian noise during noise episodes.")

    # ---- Reward shaping ----
    parser.add_argument("--score_server", default="",
                        help="URL of reward model server (e.g. http://localhost:8001). Empty to disable.")
    parser.add_argument("--shaping_weight", type=float, default=1.0,
                        help="Weight for progress-based reward shaping term.")
    parser.add_argument("--shaping_gamma", type=float, default=0.999,
                        help="Discount factor for reward shaping potential.")

    # ---- Logging ----
    parser.add_argument("--wandb_project", default="dsrl_franka",
                        help="Wandb project name.")
    parser.add_argument("--wandb_run_name", default="",
                        help="Wandb run name (empty for auto-generated).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--task_description", default="pick up the object",
                        help="Language instruction sent to the pi0 model and FrankaEnv.")
    parser.add_argument("--video_dir", default="",
                        help="Directory to save rollout videos. Empty to disable.")

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    args = parse_args()
    variant = vars(args)

    # Inject SAC hyperparameters (aligned with DSRL sim)
    variant.update(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(128, 128, 128),
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(2, 1, 1, 1),
        cnn_padding="VALID",
        latent_dim=50,
        discount=0.999,
        tau=0.005,
        critic_reduction="mean",
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type="small",
        encoder_norm="group",
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy="auto",
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=4 if variant["arm_mode"] == "dual" else 2,
    )

    print(variant)
    main(variant)
