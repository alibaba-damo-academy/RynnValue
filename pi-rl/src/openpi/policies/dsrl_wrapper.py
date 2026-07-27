# adapted from openpi
"""DSRL Policy Wrapper: wraps a pi05 Policy with a trained SAC noise predictor.

At inference time, the SAC model predicts a noise vector from the current
observation (camera images + robot state). This noise is then injected into
the pi05 diffusion model instead of random noise, guiding the VLA toward
task-specific behaviors learned via online RL.

Usage:
    # Load pi05 policy as usual
    pi05_policy = create_trained_policy(config, ckpt_dir)

    # Wrap with DSRL
    dsrl_policy = DsrlPolicyWrapper(
        base_policy=pi05_policy,
        dsrl_ckpt_dir="/path/to/experiments/single_left_train",
        arm_mode="single",
    )

    # Use like a normal policy
    actions = dsrl_policy.infer(obs)
"""

import logging
from typing import Any

import jax
import numpy as np
import PIL.Image
from openpi_client import base_policy as _base_policy
from typing_extensions import override

logger = logging.getLogger(__name__)

# Default noise horizon for pi05 diffusion model.
_PI0_NOISE_HORIZON = 16
_NOISE_DIM = 32


class DsrlPolicyWrapper(_base_policy.BasePolicy):
    """Wraps a pi05 policy with a trained SAC noise predictor for DSRL inference.

    The SAC model was trained to predict noise vectors in pi05's latent diffusion
    space. At inference time, this wrapper:
    1. Extracts camera images and robot state from the observation
    2. Preprocesses them into SAC's expected input format
    3. Calls SAC to predict a noise vector
    4. Passes the noise to pi05's infer() method
    """

    def __init__(
        self,
        base_policy: _base_policy.BasePolicy,
        dsrl_ckpt_dir: str,
        *,
        arm_mode: str = "single",
        resize_image: int = 64,
        noise_horizon: int = _PI0_NOISE_HORIZON,
        seed: int = 42,
    ):
        """Initialize the DSRL policy wrapper.

        Args:
            base_policy: The underlying pi05 policy to wrap.
            dsrl_ckpt_dir: Directory containing the trained SAC checkpoint
                (e.g., "experiments/single_left_train"). The latest checkpoint
                in this directory will be restored.
            arm_mode: "single" or "dual" — determines observation format.
            resize_image: SAC encoder input image size (pixels).
            noise_horizon: pi05's noise horizon (typically 16).
            seed: Random seed for SAC learner initialization.
        """
        self._base_policy = base_policy
        self._arm_mode = arm_mode
        self._resize_image = resize_image
        self._noise_horizon = noise_horizon

        # Load the SAC agent
        self._sac_agent = self._load_sac_agent(
            dsrl_ckpt_dir, arm_mode=arm_mode, resize_image=resize_image, seed=seed
        )
        logger.info(
            "DsrlPolicyWrapper initialized: arm_mode=%s, resize=%d, noise_horizon=%d",
            arm_mode, resize_image, noise_horizon,
        )

    def _load_sac_agent(
        self,
        ckpt_dir: str,
        *,
        arm_mode: str,
        resize_image: int,
        seed: int,
    ):
        """Load and restore the trained SAC agent from checkpoint.

        Reconstructs the PixelSACLearner with the same hyperparameters used
        during DSRL training, then restores weights from the checkpoint.
        """
        from gym.spaces import Box, Dict
        from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
        from jaxrl2.utils.general_utils import add_batch_dim

        # Determine dimensions based on arm mode
        num_cameras = 4 if arm_mode == "dual" else 2
        state_dim = 16 if arm_mode == "dual" else 8
        chunk_size = 1

        # Construct dummy observation/action spaces (matching training config)
        image_shape = (resize_image, resize_image, 3 * num_cameras, 1)
        obs_dict = {"pixels": Box(low=0, high=255, shape=image_shape, dtype=np.uint8)}
        obs_dict["state"] = Box(
            low=-np.inf, high=np.inf, shape=(state_dim, 1), dtype=np.float32
        )
        observation_space = Dict(obs_dict)
        action_space = Box(low=-1, high=1, shape=(chunk_size, _NOISE_DIM), dtype=np.float32)

        # Create dummy samples for initialization
        sample_obs = add_batch_dim(observation_space.sample())
        sample_action = add_batch_dim(action_space.sample())

        # SAC hyperparameters (must match training — from launch_train_franka.py)
        agent = PixelSACLearner(
            seed=seed,
            observations=sample_obs,
            actions=sample_action,
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
            num_cameras=num_cameras,
        )

        # Restore checkpoint
        agent.restore_checkpoint(ckpt_dir)
        logger.info("SAC agent restored from: %s", ckpt_dir)
        logger.info("SAC action_chunk_shape: %s", agent.action_chunk_shape)
        return agent

    def _obs_to_sac_input(self, obs: dict) -> dict:
        """Convert raw policy server observation to SAC encoder input format.

        The policy server receives observations like:
            observation.images.left_side:   (224,224,3) uint8
            observation.images.left_wrist:  (224,224,3) uint8
            observation.images.right_side:  (224,224,3) uint8
            observation.images.right_wrist: (224,224,3) uint8
            observation.state:              (8,) or (16,) float32

        SAC expects:
            pixels: (1, H, W, 3*num_cameras, 1)
            state:  (1, state_dim, 1)
        """
        # Extract camera images
        if self._arm_mode == "dual":
            cam_keys = [
                "observation.images.left_side",
                "observation.images.right_side",
                "observation.images.left_wrist",
                "observation.images.right_wrist",
            ]
        else:
            cam_keys = [
                "observation.images.left_side",
                "observation.images.left_wrist",
            ]

        cams = []
        for key in cam_keys:
            img = obs.get(key)
            if img is None:
                # Fallback: try without "observation.images." prefix
                short_key = key.split(".")[-1]
                img = obs.get(short_key)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            cams.append(np.asarray(img, dtype=np.uint8))

        # Resize images
        resize = self._resize_image
        if resize > 0:
            cams = [
                np.array(PIL.Image.fromarray(c).resize((resize, resize)))
                for c in cams
            ]

        # Stack cameras along channel axis: (H, W, 3*num_cameras)
        stacked = np.concatenate(cams, axis=-1)

        # Extract state
        state_key = "observation.state"
        state = obs.get(state_key)
        if state is None:
            # Try alternative keys
            for alt_key in ["observation.state.arm", "state"]:
                state = obs.get(alt_key)
                if state is not None:
                    break
        if state is None:
            state_dim = 16 if self._arm_mode == "dual" else 8
            state = np.zeros(state_dim, dtype=np.float32)

        state = np.asarray(state, dtype=np.float32)

        # Handle split state format (arm + gripper separate)
        if "observation.state.arm" in obs and "observation.state.gripper" in obs:
            arm = np.asarray(obs["observation.state.arm"], dtype=np.float32)
            gripper = np.asarray(obs["observation.state.gripper"], dtype=np.float32)
            state = np.concatenate([arm, gripper])

        # Format for SAC: add batch dim and trailing dim
        return {
            "pixels": stacked[np.newaxis, ..., np.newaxis],  # (1, H, W, C, 1)
            "state": state[np.newaxis, ..., np.newaxis],     # (1, state_dim, 1)
        }

    def _get_noise(self, obs: dict) -> np.ndarray:
        """Get noise prediction from the trained SAC model.

        Args:
            obs: Raw observation dict from the policy server client.

        Returns:
            Noise array of shape (noise_horizon, noise_dim) ready for pi05.
        """
        sac_obs = self._obs_to_sac_input(obs)

        # SAC predicts noise: flat array of shape (chunk_size * noise_dim,)
        raw_noise = self._sac_agent.sample_actions(sac_obs)

        # Reshape to (chunk_size, noise_dim)
        noise_chunk = np.reshape(raw_noise, self._sac_agent.action_chunk_shape)

        # Pad to full noise horizon: repeat last row
        pad_len = self._noise_horizon - noise_chunk.shape[0]
        if pad_len > 0:
            noise_pad = np.repeat(noise_chunk[-1:, :], pad_len, axis=0)
            noise = np.concatenate([noise_chunk, noise_pad], axis=0)
        else:
            noise = noise_chunk[:self._noise_horizon]

        return noise  # (noise_horizon, noise_dim)

    @override
    def infer(self, obs: dict) -> dict:
        """Run DSRL-guided inference.

        1. Extract SAC input from observation
        2. Predict noise via trained SAC model
        3. Pass noise to base pi05 policy
        """
        noise = self._get_noise(obs)
        return self._base_policy.infer(obs, noise=noise)

    @property
    def metadata(self) -> dict[str, Any]:
        """Forward metadata from the base policy."""
        base_meta = self._base_policy.metadata if hasattr(self._base_policy, "metadata") else {}
        return {**base_meta, "dsrl_enabled": True}
