"""Fake Franka gym-like environments for testing without hardware.

This module provides fake (simulated) versions of the Franka environments
that do NOT require ROS2 or real robot hardware. They generate random
observations and accept actions in the same format as the real envs.

Two concrete classes are provided:

- :class:`FrankaFakeDualArmEnv` -- two 7-DOF arms + two grippers,
  action/state dim ``16``.
- :class:`FrankaFakeSingleArmEnv` -- a single 7-DOF arm + one gripper,
  action/state dim ``8``.

These are useful for:
- Unit testing the training pipeline without hardware
- Debugging data flow and observation/action shapes
- Developing new features offline
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Constants duplicated here to avoid importing client.real_utils (which
# pulls in ROS2 dependencies). Values must stay in sync with
# client/real_utils/franka_constants.py.
TARGET_IMAGE_SIZE: Tuple[int, int] = (224, 224)
GRIPPER_STATE_SCALE: float = 252.0
GRIPPER_ACTION_MAX: float = 252.0

logger = logging.getLogger(__name__)


class _FrankaFakeEnvBase:
    """Base class for fake Franka environments.

    Provides the same public API as :class:`_FrankaEnvBase` but returns
    synthetic observations (random images, zero states) and accepts actions
    without publishing to any real hardware.

    Subclasses must define ``ARM_MODE``, ``JOINT_DIM``, ``GRIPPER_DIM``.
    """

    ARM_MODE: str = "base"
    JOINT_DIM: int = 0
    GRIPPER_DIM: int = 0

    def __init__(
        self,
        *,
        camera_topics: Optional[Dict[str, str]] = None,
        joint_state_topic: str = "",
        gripper_state_topic: str = "",
        joint_cmd_topic: str = "",
        gripper_cmd_topic: str = "",
        arm_joint_names: Optional[List[str]] = None,
        gripper_names: Optional[List[str]] = None,
        wrist_image_key: str = "left_wrist_image",
        reset_joints: Any = None,
        reset_grippers: Any = None,
        language_instruction: str = "",
        image_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
        auto_reset_steps: int = 0,
        reset_joint_tolerance: float = 0.05,
        reset_timeout_sec: float = 30.0,
        video_dir: Optional[str] = None,
        env_usage: str = "train",
        gripper_state_scale: float = GRIPPER_STATE_SCALE,
        gripper_action_max: float = GRIPPER_ACTION_MAX,
        gripper_mode: str = "binary",
        all_arm_joint_names: Optional[List[str]] = None,
        all_gripper_names: Optional[List[str]] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._config_kwargs = dict(extra_kwargs or {})
        self.env_usage = env_usage
        self.video_dir = video_dir

        # Gripper config
        self.gripper_state_scale: float = float(gripper_state_scale)
        self.gripper_action_max: float = float(gripper_action_max)
        self.gripper_mode: str = gripper_mode

        # Action-chunk execution config
        self.exec_publish_hz: float = float(self._config_kwargs.get("exec_publish_hz", 1000.0))
        self.exec_steps: int = int(self._config_kwargs.get("exec_steps", 20))
        self.exec_action_index: int = int(self._config_kwargs.get("exec_action_index", 0))

        # Dimensions
        self.arm_mode: str = self.ARM_MODE
        self.joint_dim: int = self.JOINT_DIM
        self.gripper_dim: int = self.GRIPPER_DIM
        self.action_dim: int = self.joint_dim + self.gripper_dim
        self.state_dim: int = self.action_dim
        self.wrist_image_key: str = wrist_image_key

        # Config
        self.language_instruction: str = language_instruction or ""
        self.image_size: Tuple[int, int] = tuple(image_size)  # type: ignore[arg-type]
        self.auto_reset_steps: int = int(auto_reset_steps or 0)

        # Reset states
        if reset_joints is not None:
            self.reset_joints = np.asarray(reset_joints, dtype=np.float64).reshape(-1)
        else:
            self.reset_joints = np.zeros(self.joint_dim, dtype=np.float64)

        if reset_grippers is not None:
            self.reset_grippers = np.asarray(reset_grippers, dtype=np.float64).reshape(-1)
        else:
            self.reset_grippers = np.zeros(self.gripper_dim, dtype=np.float64)

        # Episode state
        self._steps_since_reset: int = 0
        self._ep_count: int = 0
        self.done: bool = False
        self.success: bool = False
        self.reward: float = 0.0
        self.info: Dict[str, Any] = {}

        # Internal fake state (simulated joint positions and gripper)
        self._arm_state = self.reset_joints.copy()
        self._gripper_state = self.reset_grippers.copy()

        logger.info(
            "%s ready (arm_mode=%s, action_dim=%d, usage=%s, image_size=%s, "
            "auto_reset_steps=%d) [FAKE - no hardware]",
            type(self).__name__, self.arm_mode, self.action_dim, env_usage,
            self.image_size, self.auto_reset_steps,
        )

    # ── Observation ──────────────────────────────────────────────────

    def get_observation(self) -> Dict[str, Any]:
        """Return a synthetic observation dict matching the real env format.

        Keys (same as franka_env_base.py):
        - ``"state"``: dict with arm/gripper info
        - ``"left_side"``, ``"right_side"``, ``"left_wrist"``, ``"right_wrist"``
        - ``"prompt"``: str language instruction
        """
        h, w = self.image_size

        # Generate random images (simulate camera input)
        left_side = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        right_side = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        left_wrist = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        right_wrist = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        # State dict aligned with franka_env_base.py
        if self.arm_mode == "dual":
            observation_state = {
                "left_arm": self._arm_state[:7].astype(np.float32),
                "left_gripper": float(self._gripper_state[0]),
                "right_arm": self._arm_state[7:14].astype(np.float32),
                "right_gripper": float(self._gripper_state[1]),
            }
        else:
            observation_state = {
                "arm": self._arm_state[:7].astype(np.float32),
                "gripper": float(self._gripper_state[0]),
            }

        obs: Dict[str, Any] = {
            "state": observation_state,
            "left_side": left_side,
            "right_side": right_side,
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
            "prompt": self.language_instruction,
        }
        return obs

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        """Reset the fake environment and return the initial observation."""
        self._steps_since_reset = 0
        self.done = False
        self.success = False
        self.reward = 0.0
        self.info = {}

        # Reset internal state to initial positions
        self._arm_state = self.reset_joints.copy()
        self._gripper_state = self.reset_grippers.copy()

        logger.info(
            "[FAKE] Episode reset (ep_count=%d).", self._ep_count
        )
        return self.get_observation()

    # ── Step ─────────────────────────────────────────────────────────

    def step(self, action: Any) -> Dict[str, Any]:
        """Execute one action (fake -- just updates internal state).

        Action format (same as real env):
          - 16-dim dual arm: [left_joints(7), left_grip(1), right_joints(7), right_grip(1)]
          - 8-dim single arm: [joints(7), grip(1)]
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] < self.action_dim:
            raise ValueError(
                f"Action must be {self.action_dim}-d for {type(self).__name__}, "
                f"got shape {a.shape}"
            )
        a = np.where(np.isfinite(a), a, 0.0)
        executed_action = a[: self.action_dim].copy()

        # Update fake internal state
        if self.arm_mode == "dual":
            self._arm_state = np.concatenate([
                executed_action[:7], executed_action[8:15]
            ])
            self._gripper_state = np.array([
                np.clip(executed_action[7], 0.0, self.gripper_action_max),
                np.clip(executed_action[15], 0.0, self.gripper_action_max),
            ])
        else:
            self._arm_state = executed_action[:7].copy()
            self._gripper_state = np.array([
                np.clip(executed_action[7], 0.0, self.gripper_action_max)
            ])

        self._steps_since_reset += 1

        pub_status = {"published": True, "joint_subs": 1, "gripper_subs": 1}
        return {"executed_action": executed_action, "publish_status": pub_status}

    def send_action_chunk(
        self,
        action_chunk: Any,
        *,
        action_index: Optional[int] = None,
        execute_steps: Optional[int] = None,
        publish_hz: Optional[float] = None,
        return_observations: bool = False,
    ) -> Dict[str, Any]:
        """Execute a whole action chunk (fake -- no real timing).

        Same interface as the real env's ``send_action_chunk``.
        """
        action_index = self.exec_action_index if action_index is None else int(action_index)
        execute_steps = self.exec_steps if execute_steps is None else int(execute_steps)

        chunk = np.asarray(action_chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        horizon = chunk.shape[0]

        start_idx = min(max(action_index, 0), horizon - 1)
        end_idx = min(start_idx + execute_steps, horizon)
        exec_chunk = chunk[start_idx:end_idx]

        executed_actions: List[np.ndarray] = []
        observations: List[Dict[str, Any]] = []
        dones: List[bool] = []
        successes: List[bool] = []
        rewards: List[float] = []
        masks: List[float] = []

        for i in range(exec_chunk.shape[0]):
            step_result = self.step(exec_chunk[i])
            executed_actions.append(step_result["executed_action"])

            if return_observations:
                observations.append(self.get_observation())
                done, success, reward, mask = self.get_info_for_step()
                dones.append(bool(done))
                successes.append(bool(success))
                rewards.append(float(reward))
                masks.append(float(mask))
                if done:
                    break

        result: Dict[str, Any] = {
            "executed_actions": np.asarray(executed_actions) if executed_actions else np.zeros((0, self.action_dim)),
            "publish_status": {"published": True, "joint_subs": 1, "gripper_subs": 1},
            "start_idx": start_idx,
            "end_idx": start_idx + len(executed_actions),
        }
        if return_observations:
            result.update(
                observations=observations,
                dones=dones,
                successes=successes,
                rewards=rewards,
                masks=masks,
            )
        return result

    # ── Termination / reward ─────────────────────────────────────────

    def get_info_for_step(self) -> Tuple[bool, bool, float, float]:
        """Return ``(done, success, reward, mask)``.

        In fake env, episodes terminate only by auto_reset_steps.
        """
        time_stop = (
            self.auto_reset_steps > 0
            and self._steps_since_reset >= self.auto_reset_steps
        )

        done = bool(time_stop)
        success = False
        reward = 0.0
        mask = 0.0 if done else 1.0

        self.done, self.success, self.reward = done, success, reward

        if done:
            logger.info(
                "[FAKE] Episode done (steps=%d, auto_reset_steps=%d).",
                self._steps_since_reset, self.auto_reset_steps,
            )
        return done, success, reward, mask

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """No-op for fake env (no hardware to release)."""
        logger.info("%s.close() invoked [FAKE - nothing to tear down].", type(self).__name__)
        self._ep_count += 1

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ─── Concrete envs ─────────────────────────────────────────────────────

class FrankaFakeDualArmEnv(_FrankaFakeEnvBase):
    """Fake dual-arm Franka env: 14 joints + 2 grippers (action/state dim 16)."""

    ARM_MODE = "dual"
    JOINT_DIM = 14
    GRIPPER_DIM = 2

    def __init__(
        self,
        camera_topics: Optional[Dict[str, str]] = None,
        joint_state_topic: str = "",
        gripper_state_topic: str = "",
        joint_cmd_topic: str = "",
        gripper_cmd_topic: str = "",
        left_arm_joint_names: Optional[List[str]] = None,
        right_arm_joint_names: Optional[List[str]] = None,
        left_gripper_names: Optional[List[str]] = None,
        right_gripper_names: Optional[List[str]] = None,
        reset_joints: Any = None,
        reset_grippers: Any = None,
        language_instruction: str = "",
        image_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
        auto_reset_steps: int = 0,
        video_dir: Optional[str] = None,
        env_usage: str = "train",
        wrist_image_key: str = "left_wrist_image",
        gripper_mode: str = "binary",
        gripper_state_scale: float = GRIPPER_STATE_SCALE,
        gripper_action_max: float = GRIPPER_ACTION_MAX,
        **kwargs: Any,
    ) -> None:
        arm_joint_names = list(left_arm_joint_names or [f"joint_{i}" for i in range(7)]) + \
                          list(right_arm_joint_names or [f"joint_{i+7}" for i in range(7)])
        gripper_names = list(left_gripper_names or ["left_gripper"]) + \
                        list(right_gripper_names or ["right_gripper"])

        super().__init__(
            camera_topics=camera_topics or {"cam0": "left_side", "cam1": "right_side", "cam2": "left_wrist", "cam3": "right_wrist"},
            joint_state_topic=joint_state_topic,
            gripper_state_topic=gripper_state_topic,
            joint_cmd_topic=joint_cmd_topic,
            gripper_cmd_topic=gripper_cmd_topic,
            arm_joint_names=arm_joint_names,
            gripper_names=gripper_names,
            wrist_image_key=wrist_image_key,
            reset_joints=reset_joints,
            reset_grippers=reset_grippers,
            language_instruction=language_instruction,
            image_size=image_size,
            auto_reset_steps=auto_reset_steps,
            video_dir=video_dir,
            env_usage=env_usage,
            gripper_mode=gripper_mode,
            gripper_state_scale=gripper_state_scale,
            gripper_action_max=gripper_action_max,
            extra_kwargs=kwargs,
        )


class FrankaFakeSingleArmEnv(_FrankaFakeEnvBase):
    """Fake single-arm Franka env: 7 joints + 1 gripper (action/state dim 8).

    Parameters
    ----------
    side:
        ``"left"`` or ``"right"``. For naming consistency with real env.
    """

    ARM_MODE = "single"
    JOINT_DIM = 7
    GRIPPER_DIM = 1

    def __init__(
        self,
        camera_topics: Optional[Dict[str, str]] = None,
        joint_state_topic: str = "",
        gripper_state_topic: str = "",
        joint_cmd_topic: str = "",
        gripper_cmd_topic: str = "",
        side: str = "left",
        arm_joint_names: Optional[List[str]] = None,
        gripper_names: Optional[List[str]] = None,
        all_arm_joint_names: Optional[List[str]] = None,
        all_gripper_names: Optional[List[str]] = None,
        reset_joints: Any = None,
        reset_grippers: Any = None,
        language_instruction: str = "",
        image_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
        auto_reset_steps: int = 0,
        video_dir: Optional[str] = None,
        env_usage: str = "train",
        gripper_mode: str = "binary",
        gripper_state_scale: float = GRIPPER_STATE_SCALE,
        gripper_action_max: float = GRIPPER_ACTION_MAX,
        **kwargs: Any,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side: str = side

        super().__init__(
            camera_topics=camera_topics or {"cam0": "left_side", "cam1": "right_side", "cam2": "left_wrist", "cam3": "right_wrist"},
            joint_state_topic=joint_state_topic,
            gripper_state_topic=gripper_state_topic,
            joint_cmd_topic=joint_cmd_topic,
            gripper_cmd_topic=gripper_cmd_topic,
            arm_joint_names=list(arm_joint_names or [f"joint_{i}" for i in range(7)]),
            gripper_names=list(gripper_names or ["gripper"]),
            wrist_image_key=f"{side}_wrist_image",
            all_arm_joint_names=all_arm_joint_names,
            all_gripper_names=all_gripper_names,
            reset_joints=reset_joints,
            reset_grippers=reset_grippers,
            language_instruction=language_instruction,
            image_size=image_size,
            auto_reset_steps=auto_reset_steps,
            video_dir=video_dir,
            env_usage=env_usage,
            gripper_mode=gripper_mode,
            gripper_state_scale=gripper_state_scale,
            gripper_action_max=gripper_action_max,
            extra_kwargs=kwargs,
        )


# Backward-compatible alias
FrankaFakeEnv = FrankaFakeDualArmEnv


__all__ = [
    "FrankaFakeEnv",
    "FrankaFakeDualArmEnv",
    "FrankaFakeSingleArmEnv",
]
