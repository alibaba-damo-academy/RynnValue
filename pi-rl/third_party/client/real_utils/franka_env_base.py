"""Base Franka environment class with ROS2 lifecycle management.

Provides :class:`_FrankaEnvBase` — the shared machinery (ROS2 spin loop,
keyboard listener, video recording, manual reset, step / get_info_for_step)
for both single- and dual-arm Franka envs.

Subclasses must define the class-level ``ARM_MODE`` / ``JOINT_DIM`` /
``GRIPPER_DIM`` attributes and pass flat ``arm_joint_names`` /
``gripper_names`` lists into ``__init__``.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from client.real_utils.franka_constants import (
    DEFAULT_RESET_JOINT_TOLERANCE,
    DEFAULT_RESET_TIMEOUT,
    GRIPPER_ACTION_MAX,
    GRIPPER_MODE,
    GRIPPER_STATE_SCALE,
    TARGET_IMAGE_SIZE,
)
from client.real_utils.franka_ros2_node import FrankaROS2Node

# ROS2 imports guarded — base needs rclpy for lifecycle management.
try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    _ROS2_AVAILABLE = True
except Exception as _ros_import_err:  # pragma: no cover - depends on host
    rclpy = None  # type: ignore[assignment]
    SingleThreadedExecutor = None  # type: ignore[assignment]
    _ROS2_AVAILABLE = False
    _ROS2_IMPORT_ERROR = _ros_import_err

# Optional video deps
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class _FrankaEnvBase:
    """Shared machinery (ROS2 lifecycle, keyboard, video, reset/step) for both
    single- and dual-arm Franka envs.

    Subclasses must define the class-level ``ARM_MODE`` / ``JOINT_DIM`` /
    ``GRIPPER_DIM`` attributes and pass flat ``arm_joint_names`` /
    ``gripper_names`` lists into ``__init__``. They also choose which wrist
    camera to surface via the ``wrist_image_key`` argument.
    """

    # Subclasses override these.
    ARM_MODE: str = "base"
    JOINT_DIM: int = 0
    GRIPPER_DIM: int = 0

    def __init__(
        self,
        *,
        camera_topics: Optional[Dict[str, str]],
        joint_state_topic: str,
        gripper_state_topic: str,
        joint_cmd_topic: str,
        gripper_cmd_topic: str,
        arm_joint_names: List[str],
        gripper_names: List[str],
        wrist_image_key: str,
        reset_joints: Any = None,
        reset_grippers: Any = None,
        language_instruction: str = "",
        image_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
        auto_reset_steps: int = 0,
        reset_joint_tolerance: float = DEFAULT_RESET_JOINT_TOLERANCE,
        reset_timeout_sec: float = DEFAULT_RESET_TIMEOUT,
        video_dir: Optional[str] = None,
        env_usage: str = "train",
        gripper_state_scale: float = GRIPPER_STATE_SCALE,
        gripper_action_max: float = GRIPPER_ACTION_MAX,
        gripper_mode: str = GRIPPER_MODE,
        all_arm_joint_names: Optional[List[str]] = None,
        all_gripper_names: Optional[List[str]] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not _ROS2_AVAILABLE:
            raise RuntimeError(
                "ROS2 (rclpy) is not available; FrankaEnv cannot be instantiated. "
                f"Original import error: {_ROS2_IMPORT_ERROR!r}"
            )

        self._config_kwargs = dict(extra_kwargs or {})
        self.env_usage = env_usage
        self.video_dir = video_dir or None

        # Gripper config (aligned with franka_client_sync.py)
        self.gripper_state_scale: float = float(gripper_state_scale)
        self.gripper_action_max: float = float(gripper_action_max)
        self.gripper_mode: str = gripper_mode

        # Action-chunk execution config (mirrors franka_client_sync.py CLI args
        # --frequency / --execute_steps / --action_index). Used by
        # ``send_action_chunk`` to publish a whole inferred chunk at a fixed rate,
        # decoupled from the caller's control loop, so the low-level controller
        # follows the chunk smoothly instead of dwelling on every intermediate
        # waypoint (the cause of the mid-chunk "back-and-forth" oscillation).
        self.exec_publish_hz: float = float(self._config_kwargs.get("exec_publish_hz", 10.0))
        self.exec_steps: int = int(self._config_kwargs.get("exec_steps", 20))
        self.exec_action_index: int = int(self._config_kwargs.get("exec_action_index", 0))

        # Dimensions ---------------------------------------------------------
        self.arm_mode: str = self.ARM_MODE
        self.joint_dim: int = self.JOINT_DIM
        self.gripper_dim: int = self.GRIPPER_DIM
        self.action_dim: int = self.joint_dim + self.gripper_dim
        self.state_dim: int = self.action_dim
        self.wrist_image_key: str = wrist_image_key

        # Validate joint name lists ------------------------------------------
        arm_joint_names = list(arm_joint_names)
        gripper_names = list(gripper_names)
        if len(arm_joint_names) != self.joint_dim:
            raise ValueError(
                f"{type(self).__name__} expects {self.joint_dim} arm joint names, "
                f"got {len(arm_joint_names)}."
            )
        if len(gripper_names) != self.gripper_dim:
            raise ValueError(
                f"{type(self).__name__} expects {self.gripper_dim} gripper names, "
                f"got {len(gripper_names)}."
            )

        # Publish names: full joint/gripper lists for ROS2 publishing.
        # For dual-arm, these equal arm_joint_names/gripper_names.
        # For single-arm on shared dual driver, these include BOTH arms so that
        # publish messages contain all 14 joints + 2 grippers (non-controlled
        # side holds current position), aligned with franka_client_sync.py.
        self._publish_arm_joint_names: List[str] = list(
            all_arm_joint_names if all_arm_joint_names else arm_joint_names
        )
        self._publish_gripper_names: List[str] = list(
            all_gripper_names if all_gripper_names else gripper_names
        )

        # Config knobs -------------------------------------------------------
        self.language_instruction: str = language_instruction or ""
        self.image_size: Tuple[int, int] = tuple(image_size)
        self.auto_reset_steps: int = int(auto_reset_steps or 0)
        self.reset_joint_tolerance: float = float(reset_joint_tolerance)
        self.reset_timeout_sec: float = float(reset_timeout_sec)

        # Reset joints/grippers are stored for reference but NOT used for
        # automatic driving -- reset() waits for human repositioning.
        if reset_joints is not None:
            self.reset_joints = np.asarray(reset_joints, dtype=np.float64).reshape(-1)
        else:
            self.reset_joints = np.zeros(self.joint_dim, dtype=np.float64)

        if reset_grippers is not None:
            self.reset_grippers = np.asarray(reset_grippers, dtype=np.float64).reshape(-1)
        else:
            self.reset_grippers = np.zeros(self.gripper_dim, dtype=np.float64)

        # Episode state ------------------------------------------------------
        self._steps_since_reset: int = 0
        self._ep_count: int = 0
        self.done: bool = False
        self.success: bool = False
        self.reward: float = 0.0
        self.info: Dict[str, Any] = {}

        # Keyboard signals (set by background thread)
        self._kb_lock = threading.Lock()
        self._kb_success: bool = False
        self._kb_done: bool = False
        self._kb_stop = threading.Event()
        self._kb_thread: Optional[threading.Thread] = None

        # Manual reset synchronisation
        self._reset_event = threading.Event()
        self._waiting_for_reset: bool = False

        # Video frame buffer
        self._frame_buffer: List[np.ndarray] = []

        # ROS2 ---------------------------------------------------------------
        if camera_topics is None:
            raise ValueError("camera_topics dict is required")

        self._owns_rclpy = False
        if not rclpy.ok():
            logger.info("rclpy not initialized; calling rclpy.init() (env owns rclpy lifecycle).")
            rclpy.init()
            self._owns_rclpy = True
        else:
            logger.info("rclpy already initialized; reusing existing context (env does NOT own shutdown).")

        logger.info(
            "Creating FrankaROS2Node: cameras=%s joint_state=%s gripper_state=%s "
            "joint_cmd=%s gripper_cmd=%s publish_joints=%d publish_grippers=%d image_size=%s",
            list(camera_topics.keys()), joint_state_topic, gripper_state_topic,
            joint_cmd_topic, gripper_cmd_topic,
            len(self._publish_arm_joint_names), len(self._publish_gripper_names),
            tuple(self.image_size),
        )
        self._node = FrankaROS2Node(
            camera_topics=camera_topics,
            joint_state_topic=joint_state_topic,
            gripper_state_topic=gripper_state_topic,
            joint_cmd_topic=joint_cmd_topic,
            gripper_cmd_topic=gripper_cmd_topic,
            arm_joint_names=self._publish_arm_joint_names,
            gripper_names=self._publish_gripper_names,
            image_size=self.image_size,
        )

        # Use a dedicated SingleThreadedExecutor so that any other thread or
        # library code in the same process calling ``rclpy.spin_once`` /
        # ``rclpy.spin_until_future_complete`` (which all share the *global*
        # executor) cannot collide with our spin loop.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        logger.info(
            "Created dedicated SingleThreadedExecutor for env (id=%s).",
            id(self._executor),
        )

        self._spin_stop = threading.Event()
        self._spin_thread = threading.Thread(
            target=self._spin_loop, name="franka-ros2-spin", daemon=True
        )
        logger.info("Starting ROS2 spin thread '%s'.", self._spin_thread.name)
        self._spin_thread.start()

        # Wait for first sensor data so the very first reset/observation does
        # not return zeros.
        logger.info(
            "Waiting up to %.1fs for first full sensor snapshot ...",
            self.reset_timeout_sec,
        )
        self._wait_for_sensor_data(timeout=self.reset_timeout_sec)

        # Capture initial joint/gripper state so the non-controlled side of a
        # single-arm env holds the startup pose rather than drifting with live
        # readings on every step.
        _, _init_arm, _init_grip = self._node.snapshot()
        self._init_arm: np.ndarray = _init_arm.copy()
        self._init_grip: np.ndarray = _init_grip.copy()
        logger.info(
            "Captured initial state snapshot: arm_shape=%s grip_shape=%s",
            self._init_arm.shape, self._init_grip.shape,
        )

        # Start manual success/reset keyboard listener.
        self._start_keyboard_listener()

        logger.info(
            "%s ready (arm_mode=%s, action_dim=%d, usage=%s, "
            "video_dir=%s, image_size=%s, auto_reset_steps=%d)",
            type(self).__name__, self.arm_mode, self.action_dim, env_usage,
            self.video_dir, self.image_size, self.auto_reset_steps,
        )

    # ── ROS2 spin loop ───────────────────────────────────────────────

    def _spin_loop(self) -> None:
        tid = threading.get_ident()
        logger.info(
            "[spin] loop started (tid=%s, node=%s, executor=%s).",
            tid, type(self._node).__name__, type(self._executor).__name__,
        )
        n_iter = 0
        try:
            while not self._spin_stop.is_set() and rclpy.ok():
                try:
                    self._executor.spin_once(timeout_sec=0.1)
                except ValueError as e:
                    logger.error(
                        "[spin] ValueError in dedicated executor.spin_once "
                        "(tid=%s, iter=%d): %s -- this should not happen with "
                        "a private SingleThreadedExecutor; check whether the "
                        "node was added to another executor too.",
                        tid, n_iter, e,
                    )
                    time.sleep(0.05)
                n_iter += 1
                if n_iter % 200 == 0:
                    logger.debug("[spin] alive (tid=%s, iter=%d).", tid, n_iter)
        except Exception:  # pragma: no cover - defensive
            logger.exception("[spin] loop crashed (tid=%s, iter=%d)", tid, n_iter)
        finally:
            logger.info(
                "[spin] loop exited (tid=%s, iter=%d, stop=%s, rclpy_ok=%s).",
                tid, n_iter, self._spin_stop.is_set(),
                rclpy.ok() if rclpy is not None else None,
            )

    def _wait_for_sensor_data(self, timeout: float) -> None:
        t0 = time.time()
        last_log = t0
        while not self._node.has_all_data():
            if time.time() - t0 > timeout:
                imgs, arm, grip = self._node.snapshot()
                expected_cameras = set(self._node._camera_topics.values())
                received_cameras = set(imgs.keys())
                missing_cameras = expected_cameras - received_cameras

                missing_parts = []
                if missing_cameras:
                    missing_parts.append(f"cameras {sorted(missing_cameras)}")
                with self._node._lock:
                    if self._node._arm_pos is None:
                        missing_parts.append("joint_states")
                    if self._node._gripper_pos is None:
                        missing_parts.append("gripper_states")

                raise RuntimeError(
                    f"Franka sensor data not ready after {timeout:.1f}s — cannot start training.\n"
                    f"  Missing: {', '.join(missing_parts) if missing_parts else 'unknown'}\n"
                    f"  Received images: {sorted(received_cameras)}\n"
                    f"  Expected images: {sorted(expected_cameras)}\n"
                    f"  arm_pos shape: {arm.shape} (None={self._node._arm_pos is None})\n"
                    f"  gripper_pos shape: {grip.shape} (None={self._node._gripper_pos is None})\n"
                    f"Please check:\n"
                    f"  1. All camera ROS2 topics are publishing\n"
                    f"  2. Joint state topic is publishing\n"
                    f"  3. Gripper state topic is publishing\n"
                    f"  4. Topic names in task config match the actual ROS2 topics"
                )
            # Heartbeat every 2s so user sees progress.
            if time.time() - last_log > 2.0:
                imgs, arm, grip = self._node.snapshot()
                logger.info(
                    "[wait_sensor] still waiting (%.1fs): images=%s arm_pos=%s gripper_pos=%s.",
                    time.time() - t0, list(imgs.keys()), arm.shape, grip.shape,
                )
                last_log = time.time()
            time.sleep(0.05)
        logger.info("All Franka sensor data ready (%.2fs).", time.time() - t0)

    # ── Keyboard listener ───────────────────────────────────────────

    def _start_keyboard_listener(self) -> None:
        """Spawn a background thread that watches stdin for 's' / 'r' keys."""
        if not sys.stdin or not sys.stdin.isatty():
            logger.info("stdin is not a TTY; manual success/reset disabled.")
            return

        self._kb_thread = threading.Thread(
            target=self._keyboard_loop, name="franka-keyboard", daemon=True,
        )
        self._kb_thread.start()

    def _keyboard_loop(self) -> None:
        while not self._kb_stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                line = sys.stdin.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                ch = line.strip().lower()

                # During manual reset wait, ANY input confirms reset complete.
                if self._waiting_for_reset:
                    self._reset_event.set()
                    logger.info("Reset confirmed by operator.")
                    continue

                # Normal episode: 's' = success, 'r' = done/failed
                if ch == "s":
                    with self._kb_lock:
                        self._kb_success = True
                        self._kb_done = True
                    logger.info("Manual success signal received.")
                elif ch == "r":
                    with self._kb_lock:
                        self._kb_done = True
                    logger.info("Manual reset signal received.")
            except Exception:  # pragma: no cover - defensive
                logger.exception("Keyboard listener error")
                time.sleep(0.1)

    def _clear_keyboard_signals(self) -> None:
        with self._kb_lock:
            self._kb_success = False
            self._kb_done = False

    def _read_keyboard_signals(self) -> Tuple[bool, bool]:
        with self._kb_lock:
            return self._kb_done, self._kb_success

    # ── Observation ──────────────────────────────────────────────────

    def get_observation(self) -> Dict[str, Any]:
        """Return the current observation dict.

        Returned keys:
        - ``"state"``: dict with arm/gripper info
        - ``"left_side"``, ``"right_side"``, ``"left_wrist"``, ``"right_wrist"``
        - ``"prompt"``: str language instruction
        """
        images, arm, grip = self._node.snapshot()

        def _img(key: str) -> np.ndarray:
            img = images.get(key)
            if img is None:
                raise KeyError(
                    f"Observation image '{key}' not available. Received camera keys "
                    f"{sorted(images.keys())}. The task config's camera_topics values "
                    f"must include '{key}' (expected left_side/right_side/left_wrist/"
                    f"right_wrist, matching franka_client_sync.py). Check that the "
                    f"camera_topics mapping is correct and the topic is publishing."
                )
            return img.astype(np.uint8, copy=False)

        # Internal keys match franka_client_sync.py CAMERA_TOPICS values
        left_side = _img("left_side")
        right_side = _img("right_side")
        left_wrist = _img("left_wrist")
        right_wrist = _img("right_wrist")

        # Build state dict aligned with franka_client_sync.py
        if self.arm_mode == "dual":
            # The driver's gripper names are cross-wired w.r.t. physical left/right
            # (same as single arm): the physical left gripper reading is grip[1],
            # the physical right gripper reading is grip[0].
            observation_state = {
                "left_arm": arm[:7].astype(np.float32),
                "left_gripper": float(grip[1]),   # raw mm (0~252), physical left = grip[1]
                "right_arm": arm[7:14].astype(np.float32),
                "right_gripper": float(grip[0]),  # raw mm (0~252), physical right = grip[0]
            }
        else:
            # Single arm
            observation_state = {
                "arm": arm[:7].astype(np.float32),
                "gripper": float(grip[1]),  # raw mm (0~252)
            }

        obs: Dict[str, Any] = {
            "state": observation_state,
            "left_side": left_side,
            "right_side": right_side,
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
            "prompt": self.language_instruction,
        }

        # Buffer the main camera frame for video recording.
        if self.video_dir:
            try:
                self._frame_buffer.append(left_side.copy())
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to buffer video frame")

        return obs

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        """Wait for the human operator to manually reset the robot, then return obs."""
        # Flush any in-progress episode video first.
        self._maybe_save_episode_video()

        # Episode bookkeeping
        self._steps_since_reset = 0
        self._frame_buffer = []
        self.done = False
        self.success = False
        self.reward = 0.0
        self.info = {}
        self._clear_keyboard_signals()

        # --- Wait for human to manually reposition the robot -----------------
        if self._kb_thread is not None:
            self._reset_event.clear()
            self._waiting_for_reset = True
            logger.info("=" * 60)
            logger.info("WAITING FOR MANUAL RESET")
            logger.info("Move the robot to the start position, then press Enter.")
            logger.info("=" * 60)
            self._reset_event.wait()
            self._waiting_for_reset = False
            self._clear_keyboard_signals()
            logger.info("Reset confirmed. Starting new episode.")
        else:
            logger.info("Non-interactive mode: pausing 2s for robot to settle.")
            time.sleep(2.0)

        return self.get_observation()

    # ── Step ─────────────────────────────────────────────────────────

    def _action_to_flat(self, action: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Convert one interleaved env action into the flat ``[joints..., grippers...]``
        vector expected by :meth:`FrankaROS2Node.publish_action`.

        Action format aligned with franka_client_sync.py:
          - 16-dim dual arm: [left_joints(7), left_grip(1), right_joints(7), right_grip(1)]
          - 8-dim single arm: [joints(7), grip(1)]

        Returns ``(flat_action, executed_action)``.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] < self.action_dim:
            raise ValueError(
                f"Action must be {self.action_dim}-d for {type(self).__name__}, "
                f"got shape {a.shape}"
            )
        # Sanitise NaN / Inf -> 0.0
        a = np.where(np.isfinite(a), a, 0.0)

        executed_action = a[: self.action_dim].copy()

        # Convert interleaved action format to flat [joints, grippers] for
        # FrankaROS2Node.publish_action, applying gripper conversion.
        if self.arm_mode == "dual":
            left_joints = executed_action[:7]
            left_gripper_raw = float(executed_action[7])
            right_joints = executed_action[8:15]
            right_gripper_raw = float(executed_action[15])

            left_gripper_cmd = float(np.clip(left_gripper_raw, 0.0, self.gripper_action_max))
            right_gripper_cmd = float(np.clip(right_gripper_raw, 0.0, self.gripper_action_max))
            # The hardware driver expects a 0~100 percentage while the policy
            # outputs 0~252 mm, so scale it (same as the single-arm branch)
            left_gripper_cmd = (left_gripper_cmd / 252.0) * 100.0
            right_gripper_cmd = (right_gripper_cmd / 252.0) * 100.0

            flat_action = np.concatenate([
                left_joints, right_joints,
                # The driver's gripper names are cross-wired w.r.t. physical left/right
                # (same as single arm): the physical left command goes to the
                # "right_gripper" name slot (index 1), the physical right command
                # goes to the "left_gripper" name slot (index 0).
                np.array([right_gripper_cmd, left_gripper_cmd]),
            ])
        else:
            # Single arm on shared dual driver (aligned with franka_client_sync.py):
            # Publish full 14 joints + 2 grippers; non-controlled side holds current.
            joints = executed_action[:7]
            gripper_raw = float(executed_action[7])
            gripper_cmd = float(np.clip(gripper_raw, 0.0, self.gripper_action_max))
            # The hardware driver expects a 0~100 percentage while the policy
            # outputs 0~252 mm, so scale it
            gripper_cmd = (gripper_cmd / 252.0) * 100.0

            # Build full 14-joint array: [left(7), right(7)]
            # Use the initial (startup) state for the non-controlled right side
            # so it stays at its home pose rather than tracking live readings.
            right_joints = (
                self._init_arm[7:14] if self._init_arm.shape[0] >= 14 else np.zeros(7)
            )
            all_joints = np.concatenate([joints, right_joints])

            # Build full 2-gripper array against names [left_gripper, right_gripper].
            # On this robot the driver's "left_gripper" name physically actuates the
            # RIGHT gripper, so the left arm we control maps to the "right_gripper"
            # name slot: the policy command goes there, and the "left_gripper" slot
            # is hardcoded to 100 (fully open). Mirrors franka_client_sync.py.
            hold_gripper = 100.0
            all_grippers = np.array([hold_gripper, gripper_cmd])

            flat_action = np.concatenate([all_joints, all_grippers])

        return flat_action, executed_action

    def step(self, action: Any) -> Dict[str, Any]:
        """Execute one action and return a dict with ``executed_action``.

        Action format aligned with franka_client_sync.py:
          - 16-dim dual arm: [left_joints(7), left_grip(1), right_joints(7), right_grip(1)]
          - 8-dim single arm: [joints(7), grip(1)]
        """
        flat_action, executed_action = self._action_to_flat(action)

        # Publish and verify
        pub_status = self._node.publish_action(flat_action)

        self._steps_since_reset += 1
        with np.printoptions(precision=4, suppress=True, linewidth=200):
            logger.info(
                "[step %d] action(dim=%d)=%s | published=%s joint_subs=%s gripper_subs=%s",
                self._steps_since_reset, self.action_dim, executed_action,
                pub_status.get("published"),
                pub_status.get("joint_subs"),
                pub_status.get("gripper_subs"),
            )
        if not pub_status.get("published", False):
            logger.error(
                "[step %d] publish_action FAILED: %s",
                self._steps_since_reset, pub_status.get("error"),
            )
        elif pub_status.get("joint_subs", 0) == 0 or pub_status.get("gripper_subs", 0) == 0:
            logger.warning(
                "[step %d] publish OK but no subscriber on cmd topics "
                "(joint_subs=%s, gripper_subs=%s) -- controller likely not running.",
                self._steps_since_reset,
                pub_status.get("joint_subs"),
                pub_status.get("gripper_subs"),
            )
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
        """Execute a whole inferred action chunk at a fixed publish rate.

        This is the env-side port of ``franka_client_sync.py``'s
        ``send_action_chunk`` + chunk-slicing logic. It publishes
        ``execute_steps`` waypoints (starting at ``action_index``) at
        ``publish_hz`` Hz, decoupling the robot publish rate from the caller's
        control loop. At a high ``publish_hz`` the low-level controller only
        ever chases the *latest* (≈chunk-end) target, so the arm makes one
        smooth committed motion per inference instead of dwelling on — and
        reproducing — every intermediate waypoint (the cause of the mid-chunk
        back-and-forth oscillation seen with per-step 30 Hz execution).

        Parameters default to the env's ``exec_*`` config (which mirror the
        reference client's ``--action_index`` / ``--execute_steps`` /
        ``--frequency``). ``execute_steps`` is clamped to the chunk horizon.

        For online RL, pass ``return_observations=True`` so the caller can
        reconstruct one replay transition per executed waypoint.

        If ``return_observations`` is true, also collects one post-action
        observation/info tuple after each executed waypoint. This gives online
        training a per-action trajectory while still sending the chunk in one
        RPC. Observation collection happens inside the publish loop, so the
        effective waypoint rate will include camera/state read time.

        Returns a dict with ``executed_actions`` (N, action_dim), the last
        ``publish_status``, and the resolved ``start_idx`` / ``end_idx``.
        """
        action_index = self.exec_action_index if action_index is None else int(action_index)
        execute_steps = self.exec_steps if execute_steps is None else int(execute_steps)
        publish_hz = self.exec_publish_hz if publish_hz is None else float(publish_hz)

        chunk = np.asarray(action_chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        horizon = chunk.shape[0]

        start_idx = min(max(action_index, 0), horizon - 1)
        end_idx = min(start_idx + execute_steps, horizon)
        exec_chunk = chunk[start_idx:end_idx]

        dt = 1.0 / publish_hz if publish_hz > 0 else 0.0

        executed_actions: List[np.ndarray] = []
        observations: List[Dict[str, Any]] = []
        dones: List[bool] = []
        successes: List[bool] = []
        rewards: List[float] = []
        masks: List[float] = []
        last_status: Dict[str, Any] = {}
        for i in range(exec_chunk.shape[0]):
            flat_action, executed_action = self._action_to_flat(exec_chunk[i])
            last_status = self._node.publish_action(flat_action)
            executed_actions.append(executed_action)
            self._steps_since_reset += 1
            if dt > 0.0:
                time.sleep(dt)
            if return_observations:
                observations.append(self.get_observation())
                done, success, reward, mask = self.get_info_for_step()
                dones.append(bool(done))
                successes.append(bool(success))
                rewards.append(float(reward))
                masks.append(float(mask))
                if done:
                    break

        if not last_status.get("published", True):
            logger.error(
                "[send_action_chunk] publish_action FAILED on last waypoint: %s",
                last_status.get("error"),
            )
        elif last_status and (
            last_status.get("joint_subs", 0) == 0 or last_status.get("gripper_subs", 0) == 0
        ):
            logger.warning(
                "[send_action_chunk] publish OK but no subscriber on cmd topics "
                "(joint_subs=%s, gripper_subs=%s) -- controller likely not running.",
                last_status.get("joint_subs"), last_status.get("gripper_subs"),
            )

        result = {
            "executed_actions": np.asarray(executed_actions) if executed_actions else np.zeros((0, self.action_dim)),
            "publish_status": last_status,
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
        """Return ``(done, success, reward, mask)``."""
        kb_done, kb_success = self._read_keyboard_signals()

        time_stop = (
            self.auto_reset_steps > 0
            and self._steps_since_reset >= self.auto_reset_steps
        )

        success = bool(kb_success)
        done = bool(kb_done or time_stop)

        if done and not success and time_stop and not kb_done:
            logger.info(
                "Auto-reset: %d steps reached (limit=%d).",
                self._steps_since_reset, self.auto_reset_steps,
            )

        reward = 1.0 if success else 0.0
        mask = 0.0 if done else 1.0

        self.done, self.success, self.reward = done, success, reward

        if done:
            logger.info(
                "Episode done (success=%s, time_stop=%s, manual=%s).",
                success, time_stop, kb_done,
            )

        return done, success, reward, mask

    # ── Video ────────────────────────────────────────────────────────

    def _maybe_save_episode_video(self) -> None:
        if not self.video_dir or not self._frame_buffer:
            return
        try:
            os.makedirs(self.video_dir, exist_ok=True)
            path = os.path.join(
                self.video_dir, f"episode_{self._ep_count:04d}.mp4"
            )
            frames = self._frame_buffer

            if imageio is not None:
                with imageio.get_writer(path, fps=10) as writer:  # type: ignore[attr-defined]
                    for f in frames:
                        writer.append_data(f)
            elif cv2 is not None:  # pragma: no cover - fallback
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, 10.0, (w, h))
                for f in frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
            else:  # pragma: no cover
                logger.warning(
                    "Neither imageio nor cv2 available; skipping video save."
                )
                return
            logger.info("Saved episode video: %s (%d frames)", path, len(frames))
        except Exception:
            logger.exception("Failed to save episode video")
        finally:
            self._frame_buffer = []
            self._ep_count += 1

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop background threads and tear down the ROS2 node."""
        logger.info("%s.close() invoked; shutting down.", type(self).__name__)
        try:
            self._maybe_save_episode_video()
        except Exception:  # pragma: no cover
            logger.exception("Failed to flush pending episode video on close.")

        # Stop keyboard listener.
        self._kb_stop.set()
        if self._kb_thread is not None:
            logger.info("Joining keyboard thread (timeout=1.0s).")
            self._kb_thread.join(timeout=1.0)
            if self._kb_thread.is_alive():
                logger.warning("Keyboard thread did not exit within 1.0s.")
            self._kb_thread = None

        # Stop ROS2 spin and destroy node.
        self._spin_stop.set()
        if getattr(self, "_spin_thread", None) is not None:
            logger.info("Joining ROS2 spin thread (timeout=2.0s).")
            self._spin_thread.join(timeout=2.0)
            if self._spin_thread.is_alive():
                logger.warning("ROS2 spin thread did not exit within 2.0s.")

        # Tear down dedicated executor first (after spin thread stopped).
        executor = getattr(self, "_executor", None)
        if executor is not None:
            try:
                if self._node is not None:
                    executor.remove_node(self._node)
                logger.info("Shutting down dedicated executor.")
                executor.shutdown()
            except Exception:  # pragma: no cover
                logger.exception("Failed to shut down dedicated executor.")
            self._executor = None

        try:
            if self._node is not None:
                logger.info("Destroying ROS2 node.")
                self._node.destroy_node()
        except Exception:  # pragma: no cover
            logger.exception("Failed to destroy ROS2 node.")
        self._node = None  # type: ignore[assignment]

        if self._owns_rclpy:
            try:
                logger.info("Calling rclpy.shutdown() (env owned rclpy lifecycle).")
                rclpy.shutdown()
            except Exception:  # pragma: no cover
                logger.exception("rclpy.shutdown() raised.")
            self._owns_rclpy = False
        logger.info("%s.close() done.", type(self).__name__)

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass
