# adapted from openpi
"""Per-rollout helpers for Franka DSRL online RL training.

Core utility functions for the DSRL Franka training pipeline:
- Observation adaptation (env → SAC encoder, env → pi05 model)
- Data collection (collect_traj with step_chunk execution)
- Reward shaping (progress-based potential shaping via reward server)
- Main training loop (trajectory-wise alternating with UTD gradient updates)
- Evaluation (multi-episode control evaluation)

Environment interface contract (EnvClientWrapper / FakeEnvLocalWrapper):
  reset()            → standardized obs {left_side, right_side, left_wrist, right_wrist, state, prompt}
  step_chunk(...)    → {actions, observations, dones, successes, rewards, masks}
  get_observation()  → standardized obs
  get_info_for_step()→ (done, success, reward, mask)
  state_dim          → int (16 for dual, 8 for single)
  task_description   → str
"""

import io
import json
import logging
import os
import time

import jax
import numpy as np
import PIL.Image
import requests
import wandb
from tqdm import tqdm

from examples.rl_client.realtime_plotter import start_plotter, send_plot_data, stop_plotter

logger = logging.getLogger(__name__)

# Default noise horizon for pi05 diffusion model (must match model's action_horizon).
_PI0_NOISE_HORIZON = 16


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def obs_to_img(obs, variant):
    """Stack cameras at the SAC encoder's resolution.

    Single-arm: left_side + left_wrist (2 cameras, 6 channels).
    Dual-arm:   left_side + right_side + left_wrist + right_wrist (4 cameras, 12 channels).

    jaxrl2's pixel_sac expects (H, W, 3*num_cameras, 1) and color-jitters
    each 3-chan slice; num_cameras is set accordingly in launch_train_franka.py.

    Args:
        obs: Standardized observation dict.
        variant: Config dict with 'resize_image' and 'arm_mode'.

    Returns:
        np.ndarray of shape (H, W, 3*num_cameras) — concatenated and resized cameras.
    """
    arm_mode = variant.get("arm_mode", "dual") if hasattr(variant, "get") else getattr(variant, "arm_mode", "dual")
    if arm_mode == "dual":
        cams = [obs["left_side"], obs["right_side"], obs["left_wrist"], obs["right_wrist"]]
    else:
        cams = [obs["left_side"], obs["left_wrist"]]

    resize = int(variant.get("resize_image", 0) if hasattr(variant, "get") else getattr(variant, "resize_image", 0))
    if resize > 0:
        cams = [
            np.array(PIL.Image.fromarray(c).resize((resize, resize)))
            for c in cams
        ]
    return np.concatenate(cams, axis=-1)


def obs_to_pi_zero_input(obs, instruction, arm_mode="dual"):
    """Convert standardized FrankaEnv observation to pi05 model input format.

    pi05_franka_single_optimized's OptimizedFrankaSingleInputs expects:
      observation.state.arm:           (7,) float32 — joint positions
      observation.state.gripper:       (1,) float32 — gripper state
      observation.images.left_side:    (H, W, 3) uint8
      observation.images.right_side:   (H, W, 3) uint8
      observation.images.left_wrist:   (H, W, 3) uint8
      observation.images.right_wrist:  (H, W, 3) uint8
      prompt:                          str — language instruction

    Args:
        obs: Standardized observation dict with all 4 cameras + state + prompt.
        instruction: Task language description.
        arm_mode: "dual" (state_dim=16) or "single" (state_dim=8).

    Returns:
        Dict suitable for agent_dp.infer().
    """
    state = np.asarray(obs["state"], dtype=np.float32)

    if arm_mode == "single":
        # state is (8,) = arm(7) + gripper(1)
        arm_val = state[:7]
        gripper_val = state[7:8]
    else:
        # state is (16,) = left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)
        arm_val = np.concatenate([state[:7], state[8:15]])
        gripper_val = np.array([state[7], state[15]], dtype=np.float32)

    return {
        "observation.state.arm": arm_val,
        "observation.state.gripper": gripper_val,
        "observation.images.left_side": np.asarray(obs["left_side"], dtype=np.uint8),
        "observation.images.right_side": np.asarray(obs["right_side"], dtype=np.uint8),
        "observation.images.left_wrist": np.asarray(obs["left_wrist"], dtype=np.uint8),
        "observation.images.right_wrist": np.asarray(obs["right_wrist"], dtype=np.uint8),
        "prompt": str(instruction),
    }


def obs_to_qpos(obs, variant):
    """Return the state vector from a standardized observation as float32.

    Args:
        obs: Standardized observation dict with 'state' key.
        variant: Config dict (unused but kept for API symmetry with dsrl_sim).

    Returns:
        float32 array of shape (state_dim,).
    """
    return np.asarray(obs["state"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Observation standardization helper (for step_chunk responses)
# ---------------------------------------------------------------------------


def _standardize_obs(obs, arm_mode="dual", side="left", task_description=""):
    """Ensure an observation dict is in standardized format.

    Handles both already-standardized observations (from FakeEnvLocalWrapper)
    and raw server observations (from EnvClientWrapper's step_chunk).

    Args:
        obs: Observation dict — either standardized or raw server format.
        arm_mode: "dual" or "single".
        side: "left" or "right" (for single-arm camera selection).
        task_description: Fallback prompt string.

    Returns:
        Standardized observation: {left_side, right_side, left_wrist, right_wrist, state, prompt}.
    """
    if "left_side" in obs and "state" in obs and not isinstance(obs["state"], dict):
        # Already in standardized format
        return obs

    # Raw server format: state dict + named camera images
    state_dict = obs.get("state", {})
    if arm_mode == "dual":
        left_arm = np.asarray(state_dict.get("left_arm", np.zeros(7)), dtype=np.float32)[:7]
        left_grip = np.array([float(state_dict.get("left_gripper", 0.0))], dtype=np.float32)
        right_arm = np.asarray(state_dict.get("right_arm", np.zeros(7)), dtype=np.float32)[:7]
        right_grip = np.array([float(state_dict.get("right_gripper", 0.0))], dtype=np.float32)
        state = np.concatenate([left_arm, left_grip, right_arm, right_grip])
    else:
        arm = np.asarray(state_dict.get("arm", np.zeros(7)), dtype=np.float32)[:7]
        grip = np.array([float(state_dict.get("gripper", 0.0))], dtype=np.float32)
        state = np.concatenate([arm, grip])

    _zeros = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
        "left_side": np.asarray(obs.get("left_side", _zeros), dtype=np.uint8),
        "right_side": np.asarray(obs.get("right_side", _zeros), dtype=np.uint8),
        "left_wrist": np.asarray(obs.get("left_wrist", _zeros), dtype=np.uint8),
        "right_wrist": np.asarray(obs.get("right_wrist", _zeros), dtype=np.uint8),
        "state": state,
        "prompt": obs.get("prompt", task_description),
    }


# ---------------------------------------------------------------------------
# Reward model scoring
# ---------------------------------------------------------------------------


def score_trajectory(episode_obs, task_description, server_url, timeout=120.0):
    """Call external reward model server to get per-frame progress scores.

    Sends base-camera frames from the episode and receives progress values
    (0.0 to 1.0) indicating task completion at each timestep.

    Args:
        episode_obs: List of standardized obs dicts (uses "left_side" key for frames).
        task_description: Language instruction for the episode.
        server_url: URL of the reward server (e.g. http://localhost:8001).
        timeout: HTTP request timeout in seconds.

    Returns:
        List of float progress values per frame, or None on failure.
    """
    try:
        frames = np.stack([obs["left_side"] for obs in episode_obs], axis=0)
    except Exception as e:
        logger.warning("[score_trajectory] Failed to stack frames: %s", e)
        return None

    buf = io.BytesIO()
    np.save(buf, frames)
    buf.seek(0)

    sample = {
        "sample_type": "progress",
        "trajectory": {
            "task": task_description,
            "frames": {"__numpy_file__": "sample_0_trajectory_frames"},
            "metadata": {"video_path": None},
        },
    }
    files = {
        "sample_0_trajectory_frames": (
            "sample_0_trajectory_frames.npy",
            buf,
            "application/octet-stream",
        ),
    }
    data = {"sample_0": json.dumps(sample)}

    try:
        resp = requests.post(
            server_url.rstrip("/") + "/evaluate_batch_npy",
            files=files,
            data=data,
            timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        progress_pred = result.get("outputs_progress", {}).get("progress_pred", [[]])
        values = progress_pred[0] if progress_pred else []
        values = np.array(values)
        if values.ndim == 2:
            values = values[:, -1].tolist()
        elif values.ndim == 1:
            values = values.tolist()
        else:
            raise ValueError(f"Invalid progress values shape: {values.shape}")
        logger.debug("[score_trajectory] Got %d progress values, first=%.3f last=%.3f",
                     len(values), float(values[0]) if values else 0.0,
                     float(values[-1]) if values else 0.0)
        return [float(v) for v in values]
    except Exception as e:
        logger.warning("[score_trajectory] Reward model call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------


def apply_reward_shaping(
    episode_rewards,
    episode_masks,
    progress_scores,
    gamma=0.999,
    weight=1.0,
    query_freq=10,
):
    """Apply progress-based potential reward shaping (episodic PBRS).

    For continuing steps (mask=1):
        F_t = gamma^k * phi(s_{t+k}) - phi(s_t)

    For terminal steps (mask=0):
        F_t = 0 * phi(s_terminal) - phi(s_t) = -phi(s_t)
        (terminal state has potential 0 by convention)

    Final reward: shaped_r_t = sparse_r_t + weight * F_t

    Args:
        episode_rewards: Sparse reward array of shape (query_steps,).
        episode_masks: Continuation mask array of shape (query_steps,).
        progress_scores: Progress values at query boundaries (query_steps + 1 entries).
        gamma: Discount factor for potential shaping.
        weight: Scalar weight for the shaping term.
        query_freq: Number of env steps between queries (used for gamma exponent).

    Returns:
        Shaped reward array of shape (query_steps,).
    """
    query_steps = len(episode_rewards)
    gamma_chunk = gamma ** query_freq
    shaped_rewards = np.zeros(query_steps)

    for idx in range(query_steps):
        phi_curr = progress_scores[idx]
        if episode_masks[idx] == 0:
            # Terminal: next state potential is 0 by convention
            shaped_rewards[idx] = -phi_curr
        else:
            phi_next = progress_scores[idx + 1]
            shaped_rewards[idx] = gamma_chunk * phi_next - phi_curr

    return np.asarray(episode_rewards) + shaped_rewards * weight


# ---------------------------------------------------------------------------
# Environment interaction helpers
# ---------------------------------------------------------------------------


def _franka_reset(env, variant):
    """Reset FrankaEnv and return (obs, instruction).

    On real hardware, env.reset() blocks until a human operator repositions
    the robot and confirms readiness.

    Args:
        env: Environment wrapper (EnvClientWrapper or FakeEnvLocalWrapper).
        variant: Config dict with 'task_description'.

    Returns:
        Tuple of (standardized_obs, instruction_string).
    """
    obs = env.reset()
    instruction = str(variant.get("task_description", "") if hasattr(variant, "get")
                      else getattr(variant, "task_description", ""))
    logger.info("[franka] Episode reset. Instruction: %s", instruction)
    return obs, instruction


def _build_sac_obs_dict(obs, variant):
    """Build the SAC encoder observation dict from a standardized obs.

    Returns the format expected by jaxrl2's PixelSACLearner:
      pixels: (1, H, W, 3*num_cameras, 1)
      state:  (1, state_dim, 1)  [optional, if add_states=1]

    Args:
        obs: Standardized observation dict.
        variant: Config dict.

    Returns:
        Dict suitable for agent.sample_actions().
    """
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)

    add_states = int(variant.get("add_states", 1) if hasattr(variant, "get")
                     else getattr(variant, "add_states", 1))
    if add_states:
        return {
            "pixels": curr_image[np.newaxis, ..., np.newaxis],
            "state": qpos[np.newaxis, ..., np.newaxis],
        }
    else:
        return {"pixels": curr_image[np.newaxis, ..., np.newaxis]}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_traj(variant, agent, env, i, agent_dp=None):
    """Collect a single episode of data using step_chunk execution.

    Flow:
      1. env.reset() → initial observation
      2. Loop until done or max_timesteps:
         a. obs_to_pi_zero_input → pi05 inference → action_chunk
         b. SAC noise injection (random for early episodes, learned after)
         c. env.step_chunk(action_chunk, execute_steps, publish_hz)
         d. Record transitions at query level
      3. Optionally apply reward shaping
      4. Return episode data dict

    Args:
        variant: Config dict with all hyperparameters.
        agent: PixelSACLearner instance (provides sample_actions, action_chunk_shape).
        env: Environment wrapper with step_chunk interface.
        i: Current gradient step count (0 means no updates done yet).
        agent_dp: Pi05 policy for action inference.

    Returns:
        Dict with keys: observations, actions, rewards, masks, is_success,
        episode_return, images, env_steps.
    """
    query_frequency = int(variant.get("query_freq", 10) if hasattr(variant, "get")
                          else getattr(variant, "query_freq", 10))
    max_timesteps = int(variant.get("max_timesteps", 500) if hasattr(variant, "get")
                        else getattr(variant, "max_timesteps", 500))
    env_max_reward = variant.get("env_max_reward", 1) if hasattr(variant, "get") \
        else getattr(variant, "env_max_reward", 1)
    pi0_horizon = int(variant.get("pi0_action_horizon", 10) if hasattr(variant, "get")
                      else getattr(variant, "pi0_action_horizon", 10))
    noise_episodes = int(variant.get("noise_episodes", 5) if hasattr(variant, "get")
                         else getattr(variant, "noise_episodes", 5))
    noise_std = float(variant.get("noise_std", 0.1) if hasattr(variant, "get")
                      else getattr(variant, "noise_std", 0.1))
    publish_hz = float(variant.get("publish_hz", 10.0) if hasattr(variant, "get")
                       else getattr(variant, "publish_hz", 10.0))
    arm_mode = variant.get("arm_mode", "dual") if hasattr(variant, "get") \
        else getattr(variant, "arm_mode", "dual")
    side = variant.get("side", "left") if hasattr(variant, "get") \
        else getattr(variant, "side", "left")
    pi0_noise_horizon = int(variant.get("pi0_noise_horizon", _PI0_NOISE_HORIZON)
                            if hasattr(variant, "get")
                            else getattr(variant, "pi0_noise_horizon", _PI0_NOISE_HORIZON))

    # Track episode count for noise_episodes logic
    episode_count = int(variant.get("_episode_count", 0) if hasattr(variant, "get")
                        else getattr(variant, "_episode_count", 0))
    use_random_noise = (i == 0) or (episode_count < noise_episodes)

    agent._rng, rng = jax.random.split(agent._rng)
    obs, instruction = _franka_reset(env, variant)

    # Storage
    image_list = []          # base camera images for visualization/scoring
    raw_obs_list = []        # standardized obs for scoring
    action_list = []         # SAC noise vectors (one per query)
    obs_list = []            # SAC-format obs dicts (query_steps + 1)
    all_rewards = []         # per-step rewards across all chunks
    all_dones = []           # per-step done flags

    t = 0
    is_success = False
    last_reward = 0.0

    with tqdm(total=max_timesteps, desc="collect_traj") as pbar:
        while t < max_timesteps:
            # Record current observation for scoring and SAC buffer
            raw_obs_list.append(obs)
            image_list.append(obs["left_side"].copy() if isinstance(obs.get("left_side"), np.ndarray) else obs.get("left_side"))

            # Build SAC obs dict for noise prediction
            obs_dict = _build_sac_obs_dict(obs, variant)
            obs_list.append(obs_dict)

            # --- Pi05 inference with noise ---
            assert agent_dp is not None, "agent_dp (pi05 policy) is required"
            rng, key = jax.random.split(rng)
            obs_pi_zero = obs_to_pi_zero_input(obs, instruction, arm_mode)

            if use_random_noise:
                # Random Gaussian noise for exploration
                noise_vec = jax.random.normal(key, (1, *agent.action_chunk_shape)) * noise_std
                noise_for_buffer = noise_vec[0, :agent.action_chunk_shape[0], :]
                # Pad to model noise horizon
                noise_repeat = jax.numpy.repeat(
                    noise_vec[:, -1:, :], pi0_noise_horizon - noise_vec.shape[1], axis=1
                )
                noise = jax.numpy.concatenate([noise_vec, noise_repeat], axis=1)
            else:
                # SAC-predicted noise
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                noise_for_buffer = actions_noise
                # Pad to model noise horizon
                noise_pad = np.repeat(
                    actions_noise[-1:, :], pi0_noise_horizon - actions_noise.shape[0], axis=0
                )
                noise = jax.numpy.concatenate([actions_noise, noise_pad], axis=0)[None]

            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
            action_list.append(np.asarray(noise_for_buffer))

            # --- Send data to realtime plotter ---
            _plot_queue = variant.get("_plot_queue", None)
            if _plot_queue is not None:
                send_plot_data(_plot_queue, obs["state"], np.asarray(actions))

            # --- Execute action chunk via step_chunk ---
            execute_steps = min(query_frequency, max_timesteps - t)
            action_chunk = np.asarray(actions[:execute_steps], dtype=np.float64)

            result = env.step_chunk(action_chunk, execute_steps=execute_steps, publish_hz=publish_hz)

            chunk_rewards = result.get("rewards", [])
            chunk_dones = result.get("dones", [])
            chunk_successes = result.get("successes", [])
            chunk_observations = result.get("observations", [])

            num_executed = len(chunk_rewards)
            all_rewards.extend(chunk_rewards)
            all_dones.extend(chunk_dones)

            if chunk_successes and any(chunk_successes):
                is_success = True
                logger.info("[collect_traj] SUCCESS at t=%d", t + num_executed)
            if chunk_rewards:
                last_reward = float(chunk_rewards[-1])

            # Collect images for scoring
            for step_obs in chunk_observations:
                if isinstance(step_obs.get("left_side"), np.ndarray):
                    image_list.append(step_obs["left_side"].copy())

            # Advance observation
            obs = env.get_observation()

            # Check done
            done = any(chunk_dones) if chunk_dones else False
            if done:
                if is_success:
                    logger.info("[collect_traj] DONE (SUCCESS) at t=%d", t + num_executed)
                else:
                    logger.info("[collect_traj] DONE (FAILED) at t=%d", t + num_executed)

            t += num_executed
            pbar.update(num_executed)

            # Episode terminated — break outer loop
            if done:
                break

    # Add terminal observation for SAC buffer (query_steps + 1)
    obs_dict = _build_sac_obs_dict(obs, variant)
    obs_list.append(obs_dict)
    raw_obs_list.append(obs)

    # Compute episode statistics from per-step rewards
    rewards_array = np.array(all_rewards, dtype=np.float64) if all_rewards else np.zeros(1)
    episode_return = float(np.sum(rewards_array))

    # Check success from last reward or explicit flag
    if not is_success:
        is_success = (last_reward == env_max_reward)

    logger.info("Rollout Done: episode_return=%.3f, Success=%s, steps=%d",
                episode_return, is_success, t)

    # --- Construct query-level rewards and masks ---
    query_steps = len(action_list)

    # Progress-based reward shaping
    score_server = (variant.get("score_server", "") if hasattr(variant, "get")
                    else getattr(variant, "score_server", ""))
    shaping_weight = float(variant.get("shaping_weight", 1.0) if hasattr(variant, "get")
                           else getattr(variant, "shaping_weight", 1.0))
    shaping_gamma = float(variant.get("shaping_gamma", 0.999) if hasattr(variant, "get")
                          else getattr(variant, "shaping_gamma", 0.999))

    if score_server and query_steps > 0:
        progress = score_trajectory(raw_obs_list, instruction, score_server)
        if progress is not None and len(progress) > 0:
            logger.info("[score] Raw progress (%d frames): %s",
                        len(progress), [round(v, 3) for v in progress])
            # progress is already at query boundaries (one per raw_obs_list entry)
            progress_at_query = progress[:query_steps + 1]
            # Pad if server returned fewer values than expected
            if len(progress_at_query) < query_steps + 1:
                progress_at_query = list(progress_at_query) + [progress_at_query[-1]] * (query_steps + 1 - len(progress_at_query))
            logger.info("[score] Progress at query boundaries (%d points): %s",
                        len(progress_at_query),
                        [round(v, 3) for v in progress_at_query])
            # Build sparse rewards at query level
            if is_success:
                sparse_rewards = np.concatenate([-np.ones(query_steps - 1), [0.0]])
                masks = np.concatenate([np.ones(query_steps - 1), [0.0]])
            else:
                sparse_rewards = -np.ones(query_steps)
                masks = np.ones(query_steps)

            rewards = apply_reward_shaping(
                sparse_rewards, masks, progress_at_query,
                gamma=shaping_gamma, weight=shaping_weight, query_freq=query_frequency,
            )
            logger.info("[score] Shaped rewards (%d steps): %s",
                        len(rewards), [round(float(r), 4) for r in rewards])
        else:
            logger.warning("[score] Reward model returned no progress, using sparse reward")
            if is_success:
                rewards = np.concatenate([-np.ones(query_steps - 1), [0.0]])
                masks = np.concatenate([np.ones(query_steps - 1), [0.0]])
            else:
                rewards = -np.ones(query_steps)
                masks = np.ones(query_steps)
    else:
        # Sparse -1/0 reward for SAC training
        if is_success:
            rewards = np.concatenate([-np.ones(query_steps - 1), [0.0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0.0]])
        else:
            rewards = -np.ones(query_steps)
            masks = np.ones(query_steps)

    # Increment episode counter
    if hasattr(variant, "__setitem__"):
        variant["_episode_count"] = episode_count + 1
    elif hasattr(variant, "__setattr__"):
        variant._episode_count = episode_count + 1

    return {
        "observations": obs_list,
        "actions": action_list,
        "rewards": rewards,
        "masks": masks,
        "is_success": is_success,
        "episode_return": episode_return,
        "images": image_list,
        "env_steps": t,
    }


# ---------------------------------------------------------------------------
# Replay buffer insertion
# ---------------------------------------------------------------------------


def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    """Insert episode transitions into the replay buffer at query granularity.

    Each query step becomes one buffer transition:
      - obs_t → obs_{t+1} at query boundaries
      - action = SAC noise vector
      - reward = shaped (or sparse) reward for that query interval
      - mask = continuation mask
      - discount = gamma^query_freq

    Args:
        variant: Config dict with discount, query_freq, add_states.
        traj: Episode data dict from collect_traj().
        online_replay_buffer: jaxrl2 ReplayBuffer instance.
    """
    discount_horizon = int(variant.get("query_freq", 10) if hasattr(variant, "get")
                           else getattr(variant, "query_freq", 10))
    discount = float(variant.get("discount", 0.999) if hasattr(variant, "get")
                     else getattr(variant, "discount", 0.999))
    add_states = int(variant.get("add_states", 1) if hasattr(variant, "get")
                     else getattr(variant, "add_states", 1))

    actions = np.array(traj["actions"])  # (query_steps, chunk_size, noise_dim)
    episode_len = len(actions)
    rewards = np.array(traj["rewards"])
    masks = np.array(traj["masks"])

    for t in range(episode_len):
        obs = traj["observations"][t]
        next_obs = traj["observations"][t + 1]
        # Remove batch dimension added by _build_sac_obs_dict
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}

        if not add_states:
            obs.pop("state", None)
            next_obs.pop("state", None)

        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=discount ** discount_horizon,
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def trajwise_alternating_training_loop(
    variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
    perform_control_evals=True, shard_fn=None, agent_dp=None
):
    """Trajectory-wise alternating training loop.

    Alternates between:
    1. Collecting a full trajectory (collect_traj)
    2. Performing UTD-ratio gradient updates on the SAC learner

    Periodically evaluates and checkpoints.

    Args:
        variant: Config dict with all hyperparameters.
        agent: PixelSACLearner (SAC noise predictor).
        env: Training environment wrapper.
        eval_env: Evaluation environment wrapper (may be same as env).
        online_replay_buffer: Replay buffer for online data insertion.
        replay_buffer: Replay buffer for sampling (may be same object).
        wandb_logger: jaxrl2 WandBLogger instance.
        perform_control_evals: Whether to run control evaluations.
        shard_fn: Optional function to shard batches across devices.
        agent_dp: Pi05 policy for action inference.
    """
    batch_size = int(variant.get("batch_size", 256) if hasattr(variant, "get")
                     else getattr(variant, "batch_size", 256))
    max_steps = int(variant.get("max_steps", 500_000) if hasattr(variant, "get")
                    else getattr(variant, "max_steps", 500_000))
    multi_grad_step = int(variant.get("multi_grad_step", 20) if hasattr(variant, "get")
                          else getattr(variant, "multi_grad_step", 20))
    log_interval = int(variant.get("log_interval", 500) if hasattr(variant, "get")
                       else getattr(variant, "log_interval", 500))
    eval_interval = int(variant.get("eval_interval", 10_000) if hasattr(variant, "get")
                        else getattr(variant, "eval_interval", 10_000))
    checkpoint_interval = int(variant.get("checkpoint_interval", -1) if hasattr(variant, "get")
                              else getattr(variant, "checkpoint_interval", -1))
    start_online_updates = int(variant.get("start_online_updates", 500) if hasattr(variant, "get")
                               else getattr(variant, "start_online_updates", 500))

    replay_buffer_iterator = replay_buffer.get_iterator(batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    total_env_steps = 0
    i = 0  # gradient step counter

    # --- Start realtime plotter ---
    outputdir = variant.get("outputdir", "./experiments")
    plot_dir = os.path.join(outputdir, "plots") if isinstance(outputdir, str) else "./plots"
    _plot_queue, _plot_proc = start_plotter(plot_dir)
    variant["_plot_queue"] = _plot_queue
    logger.info("Realtime plotter started, saving to: %s", plot_dir)

    wandb_logger.log({"num_online_samples": 0}, step=i)
    wandb_logger.log({"num_online_trajs": 0}, step=i)
    wandb_logger.log({"env_steps": 0}, step=i)

    with tqdm(total=max_steps, initial=0, desc="training") as pbar:
        while i <= max_steps:
            # --- 1. Collect trajectory ---
            traj_start = time.time()
            traj = collect_traj(variant, agent, env, i, agent_dp)
            traj_time = time.time() - traj_start

            traj_id = online_replay_buffer._traj_counter
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj["env_steps"]

            logger.info(
                "Buffer: %d timesteps, %d trajs, %d total env steps (traj_time=%.1fs)",
                len(online_replay_buffer), traj_id + 1, total_env_steps, traj_time,
            )

            # --- 2. Determine gradient steps ---
            num_gradsteps = multi_grad_step

            logger.info("Traj query_steps=%d, utd=%d → num_gradsteps=%d",
                        len(traj["rewards"]), multi_grad_step, num_gradsteps)

            # --- 3. Gradient updates ---
            logger.info("Replay buffer size: %d (start_online_updates=%d)", len(online_replay_buffer), start_online_updates)
            if len(online_replay_buffer) > start_online_updates:
                for _ in range(num_gradsteps):
                    # Initial evaluation before first update
                    if i == 0:
                        logger.info("Performing evaluation for initial checkpoint")
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, "perform_eval"):
                            eval_images = agent.perform_eval(
                                variant, i, replay_buffer, replay_buffer_iterator, eval_env
                            )
                            if eval_images is not None:
                                wandb_logger.log({"reward_value_images": wandb.Image(eval_images)}, step=i)

                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1

                    # Periodic logging
                    if i % log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f"training/{k}": v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f"training/{k}", v, i)
                        wandb_logger.log({
                            "replay_buffer_size": len(online_replay_buffer),
                            "episode_return (exploration)": traj["episode_return"],
                            "is_success (exploration)": int(traj["is_success"]),
                            "timing/traj_collect_sec": traj_time,
                        }, i)

                    # Periodic evaluation
                    if i % eval_interval == 0:
                        wandb_logger.log({"num_online_samples": len(online_replay_buffer)}, step=i)
                        wandb_logger.log({"num_online_trajs": traj_id + 1}, step=i)
                        wandb_logger.log({"env_steps": total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, "perform_eval"):
                            eval_images = agent.perform_eval(
                                variant, i, replay_buffer, replay_buffer_iterator, eval_env
                            )
                            if eval_images is not None:
                                wandb_logger.log({"reward_value_images": wandb.Image(eval_images)}, step=i)

                    # Periodic checkpointing
                    if checkpoint_interval != -1 and i % checkpoint_interval == 0:
                        outputdir = (variant.get("outputdir", "./checkpoints")
                                     if hasattr(variant, "get")
                                     else getattr(variant, "outputdir", "./checkpoints"))
                        agent.save_checkpoint(outputdir, i, checkpoint_interval, keep=5)
                        logger.info("Saved checkpoint at step %d", i)

                # --- Post-update reward/value visualization ---
                make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger)

    # --- Stop realtime plotter ---
    stop_plotter(_plot_queue, _plot_proc)
    logger.info("Realtime plotter stopped.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def perform_control_eval(agent, env, i, variant, wandb_logger, agent_dp=None):
    """Run multiple evaluation episodes and log metrics.

    During evaluation, SAC noise is still used (to evaluate the learned
    exploration policy), except at i==0 where the base policy is evaluated.

    Args:
        agent: PixelSACLearner instance.
        env: Evaluation environment wrapper.
        i: Current gradient step count.
        variant: Config dict.
        wandb_logger: Logger for metrics and videos.
        agent_dp: Pi05 policy.
    """
    query_frequency = int(variant.get("query_freq", 10) if hasattr(variant, "get")
                          else getattr(variant, "query_freq", 10))
    max_timesteps = int(variant.get("max_timesteps", 500) if hasattr(variant, "get")
                        else getattr(variant, "max_timesteps", 500))
    env_max_reward = variant.get("env_max_reward", 1) if hasattr(variant, "get") \
        else getattr(variant, "env_max_reward", 1)
    pi0_horizon = int(variant.get("pi0_action_horizon", 10) if hasattr(variant, "get")
                      else getattr(variant, "pi0_action_horizon", 10))
    eval_episodes = int(variant.get("eval_episodes", 3) if hasattr(variant, "get")
                        else getattr(variant, "eval_episodes", 3))
    arm_mode = variant.get("arm_mode", "dual") if hasattr(variant, "get") \
        else getattr(variant, "arm_mode", "dual")
    publish_hz = float(variant.get("publish_hz", 10.0) if hasattr(variant, "get")
                       else getattr(variant, "publish_hz", 10.0))
    side = variant.get("side", "left") if hasattr(variant, "get") \
        else getattr(variant, "side", "left")
    pi0_noise_horizon = int(variant.get("pi0_noise_horizon", _PI0_NOISE_HORIZON)
                            if hasattr(variant, "get")
                            else getattr(variant, "pi0_noise_horizon", _PI0_NOISE_HORIZON))

    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    rng = jax.random.PRNGKey(
        int(variant.get("seed", 42) if hasattr(variant, "get") else getattr(variant, "seed", 42)) + 456
    )

    logger.info("Starting evaluation: %d episodes, query_freq=%d", eval_episodes, query_frequency)

    for rollout_id in range(eval_episodes):
        obs, instruction = _franka_reset(env, variant)

        image_list = []
        rewards = []
        reward = 0.0
        t = 0

        while t < max_timesteps:
            curr_image = obs_to_img(obs, variant)

            # Build SAC obs and get noise
            obs_dict = _build_sac_obs_dict(obs, variant)
            rng, key = jax.random.split(rng)
            assert agent_dp is not None

            obs_pi_zero = obs_to_pi_zero_input(obs, instruction, arm_mode)

            if i == 0:
                # Evaluate base policy with random noise
                noise = jax.random.normal(rng, (1, pi0_noise_horizon, 32))
            else:
                # Evaluate with SAC-predicted noise
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                noise_pad = np.repeat(
                    actions_noise[-1:, :], pi0_noise_horizon - actions_noise.shape[0], axis=0
                )
                noise = jax.numpy.concatenate([actions_noise, noise_pad], axis=0)[None]

            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]

            # Execute actions one by one via env.step
            execute_steps = min(query_frequency, max_timesteps - t)
            actions_np = np.asarray(actions[:execute_steps], dtype=np.float64)

            step_failed = False
            num_executed = 0
            for step_idx in range(execute_steps):
                single_action = actions_np[step_idx]
                try:
                    env.step(single_action)
                    step_obs = env.get_observation()
                    done, success, step_reward, mask = env.get_info_for_step()
                except Exception as e:
                    logger.error("[eval] step failed at t=%d, step_idx=%d: %s", t, step_idx, e)
                    step_failed = True
                    break

                rewards.append(step_reward)
                num_executed += 1

                # Record images for video
                img = obs_to_img(step_obs, variant)
                image_list.append(img)

                if done:
                    reward = step_reward
                    obs = step_obs
                    break

            if step_failed and num_executed == 0:
                break

            t += num_executed

            # Advance observation
            if not done:
                try:
                    obs = env.get_observation()
                except Exception:
                    break
                if num_executed > 0:
                    reward = float(rewards[-1])

            if done:
                break

        # Episode statistics
        episode_lens.append(t)
        rewards = np.array(rewards) if rewards else np.zeros(1)
        episode_return = float(np.sum(rewards))
        episode_returns.append(episode_return)
        episode_highest_reward = float(np.max(rewards)) if len(rewards) else 0.0
        highest_rewards.append(episode_highest_reward)
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)

        logger.info("Eval rollout %d: return=%.3f, success=%s, len=%d",
                    rollout_id, episode_return, is_success, t)

        # Log eval video
        if image_list:
            try:
                frames = np.stack(image_list)
                # Handle multi-camera stacked images: split into tiles
                num_channels = frames.shape[-1]
                if num_channels > 3:
                    tiles = [frames[..., k * 3:(k + 1) * 3] for k in range(num_channels // 3)]
                    video = np.concatenate(tiles, axis=2).transpose(0, 3, 1, 2)
                else:
                    video = frames.transpose(0, 3, 1, 2)
                wandb_logger.log({f"eval_video/{rollout_id}": wandb.Video(video, fps=10, format="mp4")}, step=i)
            except Exception as e:
                logger.debug("Failed to log eval video: %s", e)

    # Aggregate metrics
    success_rate = float(np.mean(np.array(success_rates)))
    avg_return = float(np.mean(episode_returns))
    avg_episode_len = float(np.mean(episode_lens)) if episode_lens else 0.0

    wandb_logger.log({"evaluation/avg_return": avg_return}, step=i)
    wandb_logger.log({"evaluation/success_rate": success_rate}, step=i)
    wandb_logger.log({"evaluation/avg_episode_len": avg_episode_len}, step=i)

    for r in range(int(env_max_reward) + 1):
        more_or_equal_r = int((np.array(highest_rewards) >= r).sum())
        more_or_equal_r_rate = more_or_equal_r / max(eval_episodes, 1)
        wandb_logger.log({f"evaluation/Reward >= {r}": more_or_equal_r_rate}, step=i)

    summary_str = (
        f"\nEval @ step {i}: success_rate={success_rate:.3f}, "
        f"avg_return={avg_return:.3f}, avg_len={avg_episode_len:.1f}\n"
    )
    logger.info(summary_str)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger):
    """Generate value/reward visualization from random trajectories."""
    try:
        trajs = replay_buffer.get_random_trajs(3)
        images = agent.make_value_reward_visulization(variant, trajs)
        wandb_logger.log({"reward_value_images": wandb.Image(images)}, step=i)
    except Exception as e:
        logger.debug("Value/reward visualization failed: %s", e)
