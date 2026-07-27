"""ROS2 node for Franka robot sensor subscription and action publishing.

This module provides :class:`FrankaROS2Node`, which handles all low-level
ROS2 communication (camera image subscriptions, joint/gripper state
subscriptions, joint/gripper command publishing).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from client.real_utils.franka_constants import TARGET_IMAGE_SIZE
from client.real_utils.franka_images import resize_with_pad, ros_image_to_numpy

# ROS2 imports are guarded so this module remains importable on a training
# machine that does not have ROS2 installed.
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as ROSImage
    from sensor_msgs.msg import JointState

    _ROS2_AVAILABLE = True
except Exception as _ros_import_err:  # pragma: no cover - depends on host
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment, misc]
    ROSImage = None  # type: ignore[assignment]
    JointState = None  # type: ignore[assignment]
    _ROS2_AVAILABLE = False
    _ROS2_IMPORT_ERROR = _ros_import_err


class FrankaROS2Node(Node):  # type: ignore[misc]
    """ROS2 node that subscribes to Franka sensors and publishes joint/gripper commands.

    Takes flat ``arm_joint_names`` / ``gripper_names`` lists; sizing for both
    single- and dual-arm setups is handled by the caller.
    """

    def __init__(
        self,
        camera_topics: Dict[str, str],
        joint_state_topic: str,
        gripper_state_topic: str,
        joint_cmd_topic: str,
        gripper_cmd_topic: str,
        arm_joint_names: List[str],
        gripper_names: List[str],
        image_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
        node_name: str = "franka_expo_ft_env",
        qos: int = 10,
    ) -> None:
        super().__init__(node_name)

        self._camera_topics = dict(camera_topics)
        self._image_size = tuple(image_size)
        self._arm_joint_names = list(arm_joint_names)
        self._gripper_names = list(gripper_names)
        self._n_arm_joints = len(self._arm_joint_names)
        self._n_grippers = len(self._gripper_names)

        # Publishers
        self._joint_pub = self.create_publisher(JointState, joint_cmd_topic, qos)
        self._gripper_pub = self.create_publisher(JointState, gripper_cmd_topic, qos)

        # Sensor cache (protected by _lock)
        self._lock = threading.Lock()
        self._images: Dict[str, np.ndarray] = {}
        self._arm_pos: Optional[np.ndarray] = None
        self._gripper_pos: Optional[np.ndarray] = None

        # Camera subscriptions
        for topic, obs_key in self._camera_topics.items():
            self.create_subscription(
                ROSImage, topic,
                lambda msg, key=obs_key: self._image_cb(msg, key),
                qos,
            )
            self.get_logger().info(f"Camera sub: {topic} -> {obs_key}")

        # Joint / gripper subscriptions
        self.create_subscription(JointState, joint_state_topic, self._joint_cb, qos)
        self.create_subscription(JointState, gripper_state_topic, self._gripper_cb, qos)

        self.get_logger().info("FrankaROS2Node ready - waiting for sensor data ...")

    # ── Callbacks ────────────────────────────────────────────────────

    def _image_cb(self, msg: "ROSImage", obs_key: str) -> None:
        try:
            img_rgb = ros_image_to_numpy(msg)
            img_resized = resize_with_pad(img_rgb, self._image_size)
            with self._lock:
                self._images[obs_key] = img_resized
        except Exception as e:  # pragma: no cover - defensive
            self.get_logger().error(f"Image cb error ({obs_key}): {e}")

    def _joint_cb(self, msg: "JointState") -> None:
        with self._lock:
            self._arm_pos = np.array(msg.position, dtype=np.float64)

    def _gripper_cb(self, msg: "JointState") -> None:
        with self._lock:
            self._gripper_pos = np.array(msg.position, dtype=np.float64)

    # ── Data access ──────────────────────────────────────────────────

    def has_all_data(self) -> bool:
        """True once every camera, joint state, and gripper state has arrived."""
        with self._lock:
            n_img = len(self._images)
            has_arm = self._arm_pos is not None
            has_grip = self._gripper_pos is not None
        return (
            n_img == len(self._camera_topics)
            and has_arm
            and has_grip
        )

    def snapshot(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Return a thread-safe copy of (images, arm_pos, gripper_pos).

        Returns zero-filled arrays for any field that has not yet been received
        so that callers (e.g. ``get_observation`` from a freshly created env)
        do not crash. Use ``has_all_data`` to check readiness. Default sizes
        match the configured arm/gripper joint name lists, so single-arm and
        dual-arm setups both work transparently.
        """
        n_arm = max(self._n_arm_joints, 1)
        n_grip = max(self._n_grippers, 1)
        with self._lock:
            images = {k: v.copy() for k, v in self._images.items()}
            arm = self._arm_pos.copy() if self._arm_pos is not None else np.zeros(n_arm, np.float64)
            grip = self._gripper_pos.copy() if self._gripper_pos is not None else np.zeros(n_grip, np.float64)
        return images, arm, grip

    # ── Action publishing ────────────────────────────────────────────

    @staticmethod
    def _make_joint_state(names: List[str], positions: Iterable[float]) -> "JointState":
        msg = JointState()
        msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        msg.name = names
        msg.position = [float(p) for p in positions]
        return msg

    def publish_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Publish a flat action of length ``n_arm_joints + n_grippers``.

        Returns a status dict so callers can verify the publish actually went
        through. Note that ROS2 publishers are fire-and-forget; the strongest
        guarantee we can give from the publisher side is:

        - the call did not raise ("published" == True)
        - and at least one matched subscriber exists at the time of publish
          (``joint_subs`` / ``gripper_subs`` > 0).

        Returned keys: ``published``, ``error``, ``joint_subs``,
        ``gripper_subs``, ``joint_positions``, ``gripper_positions``.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        n_joints = self._n_arm_joints
        n_grippers = self._n_grippers
        expected = n_joints + n_grippers
        if a.shape[0] < expected:
            raise ValueError(
                f"Expected action of dim >= {expected} "
                f"({n_joints} joints + {n_grippers} grippers), got {a.shape[0]}"
            )

        joint_positions = a[:n_joints].tolist()
        gripper_positions = a[n_joints:n_joints + n_grippers].tolist()

        # Snapshot matched-subscriber counts BEFORE publishing -- if it is 0,
        # nobody downstream will receive the command (controller likely down).
        try:
            joint_subs = int(self._joint_pub.get_subscription_count())
        except Exception:  # pragma: no cover - defensive
            joint_subs = -1
        try:
            gripper_subs = int(self._gripper_pub.get_subscription_count())
        except Exception:  # pragma: no cover - defensive
            gripper_subs = -1

        status: Dict[str, Any] = {
            "published": False,
            "error": None,
            "joint_subs": joint_subs,
            "gripper_subs": gripper_subs,
            "joint_positions": joint_positions,
            "gripper_positions": gripper_positions,
        }

        try:
            joint_msg = self._make_joint_state(
                names=self._arm_joint_names,
                positions=joint_positions,
            )
            self._joint_pub.publish(joint_msg)

            gripper_msg = self._make_joint_state(
                names=self._gripper_names,
                positions=gripper_positions,
            )
            self._gripper_pub.publish(gripper_msg)
            status["published"] = True
        except Exception as e:
            status["error"] = repr(e)
            self.get_logger().error(f"publish_action failed: {e!r}")
        return status
