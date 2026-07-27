# adapted from openpi
"""Franka synchronous inference client (direct ROS2 subscription + WebSocket)

Observations are obtained the same way as infer_dream (direct ROS2 subscription);
communication uses WebSocket (openpi_client) with synchronous blocking inference + execution.

Semantic conventions (aligned with the model inference side):
  - No delta: the model predicts absolute joint positions directly; the client sends them as-is
  - Gripper raw mm: both state input and action output are raw mm (0–252), no normalization
  - 4 cameras: left_side, right_side, left_wrist, right_wrist sent as independent keys
  - 8-dim native: state/action = arm7 + gripper1; padding is handled inside the model

Usage:
    python franka_client_sync.py --host localhost --port 8000
    python franka_client_sync.py --host localhost --port 8000 \
        --execute_steps 30 --action_index 0
"""

import argparse
import logging
import os
import threading
import time

import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image as ROSImage

from openpi_client import websocket_client_policy
from realtime_plotter import start_plotter, send_plot_data, stop_plotter

try:
    import imageio
except ImportError:
    imageio = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Configuration constants ──────────────────────────────────────────

TARGET_IMAGE_SIZE = (224, 224)   # (W, H)
ACTION_RATE_HZ = 30.0
TASK_PROMPT = "flip the steak"

# ROS2 topics
CAMERA_TOPICS = {
    "/camera/d435_a/color/image_raw":      "left_side",       # left view
    "/camera/d435_c/color/image_raw":      "right_side",      # right view
    "/camera/d405_b/color/image_rect_raw": "left_wrist",      # left wrist
    "/camera/d405_a/color/image_rect_raw": "right_wrist",     # right wrist
}

JOINT_STATE_TOPIC   = "/dual_franka_driver/joint_states"
GRIPPER_STATE_TOPIC = "/dual_franka_driver/gripper_states"

JOINT_CMD_TOPIC   = "/dual_franka_planner/joint_command"
GRIPPER_CMD_TOPIC = "/dual_franka_planner/gripper_command"

LEFT_ARM_JOINT_NAMES  = [f"left_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINT_NAMES = [f"right_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_NAMES    = ["left_gripper"]
RIGHT_GRIPPER_NAMES   = ["right_gripper"]

# Gripper physical scales:
#   state gripper: 0~252 mm (physical travel reported by the driver)
#   action gripper: 0~382 (model action output range, max value in collected data)
GRIPPER_STATE_MAX = 252.0   # max state feedback value / max robot command value
GRIPPER_ACTION_MAX = 252.0  # max value of the model's action output
GRIPPER_THRESHOLD = 150.0   # binarization threshold: < this → 0 (closed), >= this → 252 (open)


def policy_gripper_to_cmd(value: float) -> float:
    """Model gripper action → robot gripper command (direct pass-through)."""
    return float(value)


# ─── Image utilities ──────────────────────────────────────────────────

def _ros_image_to_numpy(msg: ROSImage) -> np.ndarray:
    """ROS2 sensor_msgs/Image → (H, W, 3) uint8 RGB ndarray."""
    enc = msg.encoding.lower()
    channel_map = {
        "rgb8": 3, "bgr8": 3,
        "rgba8": 4, "bgra8": 4,
        "mono8": 1,
    }
    channels = channel_map.get(enc, 3)
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)

    if enc == "bgr8":
        img = img[:, :, ::-1].copy()
    elif enc == "bgra8":
        img = img[:, :, [2, 1, 0]].copy()
    elif enc == "rgba8":
        img = img[:, :, :3].copy()
    elif enc == "mono8":
        img = np.stack([img[:, :, 0]] * 3, axis=-1)

    return img


def _resize_with_pad(img: np.ndarray, size: tuple = TARGET_IMAGE_SIZE) -> np.ndarray:
    """(H, W, 3) uint8 → resize with pad → (H', W', 3) uint8; keep aspect ratio, center with black padding."""
    target_w, target_h = size
    pil_img = Image.fromarray(img)
    orig_w, orig_h = pil_img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

    padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))
    return np.array(padded)


# ─── ROS2 node ─────────────────────────────────────────────────────────

class FrankaNode(Node):
    """
    Integrated ROS2 node:
        - Subscribes to 3 cameras, joint states, gripper states
        - Publishes joint commands and gripper commands
    """

    def __init__(self):
        super().__init__("franka_sync_node")

        # ── Publishers ─────────────────────────────────────────────
        self._joint_pub = self.create_publisher(
            JointState, JOINT_CMD_TOPIC, 10)
        self._gripper_pub = self.create_publisher(
            JointState, GRIPPER_CMD_TOPIC, 10)

        # ── Sensor caches (protected by _lock) ────────────────────
        self._lock = threading.Lock()
        self._images: dict[str, np.ndarray] = {}       # key → (H, W, 3) uint8
        self._arm_pos: np.ndarray | None = None         # (14,)
        self._gripper_pos: np.ndarray | None = None     # (2,)
        self._gripper_received = False                  # flag set when first gripper state is received
        self._action_count = 0                          # count of published actions
        self._init_arm_pos: np.ndarray | None = None    # initial joint positions (set on first joint_states message)

        # ── Camera subscriptions ──────────────────────────────────
        for topic, obs_key in CAMERA_TOPICS.items():
            self.create_subscription(
                ROSImage, topic,
                lambda msg, key=obs_key: self._image_cb(msg, key),
                10,
            )
            self.get_logger().info(f"Camera sub: {topic} → {obs_key}")

        # ── Joint/gripper subscriptions ──────────────────────────
        self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._joint_cb, 10)
        self.create_subscription(
            JointState, GRIPPER_STATE_TOPIC, self._gripper_cb, 10)

        self.get_logger().info("FrankaNode (sync) ready — waiting for sensor data …")

    # ── Callbacks ─────────────────────────────────────────────────

    def _image_cb(self, msg: ROSImage, obs_key: str):
        try:
            img_rgb = _ros_image_to_numpy(msg)
            img_resized = _resize_with_pad(img_rgb)
            with self._lock:
                self._images[obs_key] = img_resized
        except Exception as e:
            self.get_logger().error(f"Image cb error ({obs_key}): {e}")

    def _joint_cb(self, msg: JointState):
        with self._lock:
            self._arm_pos = np.array(msg.position, dtype=np.float64)
            if self._init_arm_pos is None:
                self._init_arm_pos = self._arm_pos.copy()

    def _gripper_cb(self, msg: JointState):
        with self._lock:
            self._gripper_pos = np.array(msg.position, dtype=np.float64)
            if not self._gripper_received:
                self._gripper_received = True
                self.get_logger().info(
                    f"Gripper state received: raw={self._gripper_pos.tolist()}"
                )

    # ── Data queries ──────────────────────────────────────────────

    def has_all_data(self) -> bool:
        with self._lock:
            n_img = len(self._images)
            has_arm = self._arm_pos is not None
            has_grip = self._gripper_pos is not None
        ok = (n_img == len(CAMERA_TOPICS)) and has_arm and has_grip
        if not ok:
            missing = []
            if n_img < len(CAMERA_TOPICS):
                with self._lock:
                    got = list(self._images.keys())
                missing.append(f"cameras({n_img}/{len(CAMERA_TOPICS)} got={got})")
            if not has_arm:
                missing.append("joint_states")
            if not has_grip:
                missing.append("gripper_states")
            self.get_logger().info(
                f"Waiting: {', '.join(missing)}", throttle_duration_sec=2.0)
        else:
            # Once all sensors are ready, periodically print current gripper raw mm state
            with self._lock:
                gripper = self._gripper_pos.copy()
            self.get_logger().info(
                f"Gripper state OK: raw_mm={gripper.tolist()}",
                throttle_duration_sec=5.0,
            )
        return ok

    def get_obs(self, prompt: str = TASK_PROMPT) -> dict:
        """Build an observation dict in the format expected by the model (8-dim native).

        Keys expected by the model:
            observation.state              (8,) float32  [arm7, gripper_raw_mm]
            observation.images.left_side   (224,224,3) uint8
            observation.images.left_wrist  (224,224,3) uint8
            observation.images.right_side  (224,224,3) uint8
            observation.images.right_wrist (224,224,3) uint8
            prompt                         str

        Gripper is passed as raw mm (0–252); the model's PadStatesAndActions handles padding.
        """
        with self._lock:
            arm = self._arm_pos.copy()
            gripper = self._gripper_pos.copy()
            images = {k: v.copy() for k, v in self._images.items()}

        # state (8,): [arm7_joints, gripper_raw_mm]
        state = np.concatenate([
            arm[:7],
            [float(gripper[1])],  # raw mm, no ÷252 normalization
        ]).astype(np.float32)
        
        # print state with round 3
        print("State:", np.round(state, 3).tolist())
        return {
            "observation.state": state,
            "observation.images.left_side": images["left_side"],
            "observation.images.left_wrist": images["left_wrist"],
            "observation.images.right_side": images["right_side"],
            "observation.images.right_wrist": images["right_wrist"],
            "prompt": prompt,
        }

    # ── Action publishing ─────────────────────────────────────────

    @staticmethod
    def _make_joint_state(names: list, positions: list) -> JointState:
        msg = JointState()
        msg.header.stamp = rclpy.clock.Clock().now().to_msg()
        msg.name = names
        msg.position = [float(p) for p in positions]
        return msg

    def publish_action(self, action: np.ndarray):
        """Publish a single action step; supports 8-dim single-arm or 16-dim dual-arm.

        8-dim (single arm mode):
          [0:7]  → left arm joints
          [7]    → left gripper
        16-dim (dual arm mode):
          [0:7]   → left arm joints
          [7]     → left gripper
          [8:15]  → right arm joints
          [15]    → right gripper
        """
        action = np.asarray(action).reshape(-1)
        dim = action.shape[0]

        with self._lock:
            arm = self._init_arm_pos.copy() if self._init_arm_pos is not None else np.zeros(14)
            gripper = self._gripper_pos.copy() if self._gripper_pos is not None else np.zeros(2)

        if dim == 8:
            # Single arm: control the left arm only; right arm holds its current state
            left_joints  = action[:7].tolist()
            right_joints = arm[7:14].tolist()
            # Model output gripper is already raw mm (0–252); clip directly
            left_gripper  = policy_gripper_to_cmd(float(action[7]))
            # Right gripper holds its current state (also raw mm)
            right_gripper = float(gripper[1])

            joint_msg = self._make_joint_state(
                names=LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES,
                positions=left_joints + right_joints,
            )
            self._joint_pub.publish(joint_msg)

            # Hardware driver expects order: [right_gripper, left_gripper]
            gripper_msg = self._make_joint_state(
                names=LEFT_GRIPPER_NAMES + RIGHT_GRIPPER_NAMES,
                positions=[100, (left_gripper / 252.0) * 100],
            )
            self._gripper_pub.publish(gripper_msg)
            
        elif dim == 16:
            # Dual arm
            left_joints  = action[:7].tolist()
            right_joints = action[8:15].tolist()
            left_gripper  = policy_gripper_to_cmd(float(action[7]))
            right_gripper = policy_gripper_to_cmd(float(action[15]))
            
            joint_msg = self._make_joint_state(
                names=LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES,
                positions=left_joints + right_joints,
            )
            self._joint_pub.publish(joint_msg)

            gripper_msg = self._make_joint_state(
                names=LEFT_GRIPPER_NAMES + RIGHT_GRIPPER_NAMES,
                positions=[left_gripper, right_gripper],
            )
            self._gripper_pub.publish(gripper_msg)
        else:
            raise ValueError(f"Unexpected action dim: {action.shape}")

        # self._action_count += 1
        # # Print action/gripper info on the first publish + every 30 steps
        # if self._action_count == 1 or self._action_count % 30 == 0:
        #     if dim == 8:
        #         gripper_info = (
        #             f"left_gripper_cmd={left_gripper:.1f}mm; "
        #             f"right_gripper_hold={right_gripper:.1f}mm"
        #         )
        #     else:
        #         gripper_info = (
        #             f"left_gripper_cmd={left_gripper:.1f}mm; "
        #             f"right_gripper_cmd={right_gripper:.1f}mm"
        #         )
        #     self.get_logger().info(
        #         f"Action #{self._action_count} published: "
        #         f"action_dim={dim}, {gripper_info}"
        #     )

    def send_action_chunk(self, action_chunk: np.ndarray, hz: float = ACTION_RATE_HZ):
        """Publish action_chunk step by step at a fixed rate; shape=(N, 8) or (N, 16)."""
        dt = 1.0 / hz
        for i in range(action_chunk.shape[0]):
            self.publish_action(action_chunk[i])
            time.sleep(dt)


# ─── Keyboard input listener ──────────────────────────────────────────

class KeyboardListener:
    """Background thread listening for terminal keyboard input, used for episode control.

    - 'j': start the current episode
    - 's': end the current episode as success
    - 'f': end the current episode as failure
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._last_key: str = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._listen_loop, name="kb-listener", daemon=True
        )
        self._thread.start()

    def _listen_loop(self):
        import sys
        import select
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                line = sys.stdin.readline().strip().lower()
                if line:
                    with self._lock:
                        self._last_key = line
            except Exception:
                time.sleep(0.1)

    def get_key(self) -> str:
        """Get and clear the most recent key press."""
        with self._lock:
            key = self._last_key
            self._last_key = ""
        return key

    def wait_for_key(self, target: str) -> str:
        """Block until the specified key is pressed."""
        while not self._stop.is_set():
            key = self.get_key()
            if key == target:
                return key
            time.sleep(0.05)
        return ""

    def stop(self):
        self._stop.set()


# ─── Main function ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Franka sync inference client (direct ROS2 subscription) - serial inference and execution"
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--prompt", type=str, default="Pick up the two breads from the table and put them in the basket.")
    parser.add_argument("--task", type=str, default="unknown_task", help="Task name (used for video directory naming)")
    parser.add_argument("--algorithm", type=str, default="unknown",
                        choices=["iql_dense", "iql_sparse", "sft", "unknown"],
                        help="Algorithm type (used for video directory naming and statistics)")
    parser.add_argument("--num_episodes", type=int, default=20, help="Total number of episodes")
    parser.add_argument("--num_steps", type=int, default=8000, help="Max steps per episode, 0=unlimited")
    parser.add_argument("--frequency", type=float, default=10, help="Control frequency in Hz")
    parser.add_argument("--action_index", type=int, default=0)
    parser.add_argument("--execute_steps", type=int, default=16, help="Steps executed per chunk (also the re-inference interval)")
    parser.add_argument(
        "--video_dir",
        type=str,
        default="eval_videos",
        help="Directory to save per-episode videos",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="plots",
        help="Directory to save plot images",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable real-time plotting",
    )
    args = parser.parse_args()

    # Video directory: eval_videos/<task>/<algorithm>/
    args.video_dir = os.path.join(args.video_dir, args.task, args.algorithm)

    # Start the real-time plotting process
    plot_queue = None
    plot_proc = None
    if not args.no_plot:
        plot_queue, plot_proc = start_plotter(plot_dir=args.plot_dir)
        logger.info("Real-time plotting process started")

    # ── 1. Initialize ROS2 ────────────────────────────────────────
    rclpy.init()
    node = FrankaNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Keyboard listener
    kb = KeyboardListener()

    try:
        # ── 2. Wait for sensors to be ready ──────────────────────
        logger.info("Waiting for all sensor data …")
        while not node.has_all_data():
            time.sleep(0.2)
        logger.info("All sensor data ready!")

        # ── 3. Create WebSocket client ───────────────────────────
        client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        logger.info(f"Server metadata: {client.get_server_metadata()}")

        # ── 4. Warmup ────────────────────────────────────────────
        logger.info("Warmup ...")
        for _ in range(2):
            obs = node.get_obs(prompt=args.prompt)
            client.infer(obs)

        # ── 5. Episode loop ──────────────────────────────────────
        num_episodes = args.num_episodes
        episode_results = []  # list of (success: bool, steps: int)

        print(f"\n{'━' * 60}")
        print(f"  Task: {args.task} | Algorithm: {args.algorithm} | Total episodes: {num_episodes}")
        print(f"  Video dir: {args.video_dir}")
        print(f"{'━' * 60}")

        for episode_idx in range(1, num_episodes + 1):
            # ── Show running statistics ──
            if episode_results:
                successes = sum(1 for s, _ in episode_results if s)
                total_steps_list = [st for _, st in episode_results]
                avg_success = successes / len(episode_results) * 100
                avg_steps = np.mean(total_steps_list)
                std_steps = np.std(total_steps_list)
                print(f"\n{'═' * 60}")
                print(f"  History: {successes}/{len(episode_results)} succeeded "
                      f"| success rate {avg_success:.1f}% | avg steps {avg_steps:.1f} ± {std_steps:.1f}")
                print(f"{'═' * 60}")

            print(f"\n{'━' * 60}")
            print(f"  Episode {episode_idx}/{num_episodes} — press 'j' to start")
            print(f"{'━' * 60}")

            # Wait for 'j' to start
            while True:
                key = kb.get_key()
                if key == "j":
                    break
                time.sleep(0.05)

            print(f"\n▶ Episode {episode_idx} started! (press 's'=success / 'f'=failure to end)")
            episode_steps = 0
            episode_done = False
            episode_success = False
            # Video frame buffers: third-person view (left_side) + first-person views (left_wrist / right_wrist)
            frames_third_person = []
            frames_first_person = []
            frames_right_wrist = []

            # ── Inference + control main loop ─────────────────────
            while not episode_done:
                episode_steps += 1

                obs = node.get_obs(prompt=args.prompt)

                t0 = time.time()
                result = client.infer(obs)
                latency_ms = (time.time() - t0) * 1000
                action_chunk = result["actions"]  # (horizon, 8)

                horizon = action_chunk.shape[0]
                start_idx = min(args.action_index, horizon - 1)
                end_idx = min(start_idx + args.execute_steps, horizon)
                exec_chunk = action_chunk[start_idx:end_idx]

                if episode_steps <= 3 or episode_steps % 10 == 0:
                    cur_state = obs["observation.state"]
                    logger.info(
                        f"[Ep {episode_idx} Step {episode_steps}] Infer {latency_ms:.1f}ms | "
                        f"action {action_chunk.shape} | "
                        f"exec [{start_idx}:{end_idx}] ({len(exec_chunk)} steps)\n"
                        f"  cur_state  = {np.round(cur_state, 3)}\n"
                        f"  step_first = {exec_chunk[0].round(3)}\n"
                        f"  step_last  = {exec_chunk[-1].round(3)}"
                    )
                else:
                    logger.info(
                        f"[Ep {episode_idx} Step {episode_steps}] Infer {latency_ms:.1f}ms | "
                        f"exec [{start_idx}:{end_idx}] ({len(exec_chunk)} steps)"
                    )

                # ── Send data to the plotting process ──
                if plot_queue is not None:
                    send_plot_data(plot_queue, obs["observation.state"], action_chunk)

                node.send_action_chunk(exec_chunk, hz=args.frequency)

                # ── Buffer video frames ──
                frames_third_person.append(obs["observation.images.left_side"])
                frames_first_person.append(obs["observation.images.left_wrist"])
                if "observation.images.right_wrist" in obs:
                    frames_right_wrist.append(obs["observation.images.right_wrist"])

                # ── Check keyboard: 's' success / 'f' failure ──
                key = kb.get_key()
                if key == "s":
                    episode_done = True
                    episode_success = True
                elif key == "f":
                    episode_done = True
                    episode_success = False

                # Max step limit
                if args.num_steps > 0 and episode_steps >= args.num_steps:
                    episode_done = True

            # ── Episode finished ──
            result_str = "✓ Success" if episode_success else "✗ Failure"
            episode_results.append((episode_success, episode_steps))

            # ── Save videos ──
            if imageio is not None and (frames_third_person or frames_first_person):
                os.makedirs(args.video_dir, exist_ok=True)
                tag = "success" if episode_success else "fail"
                if frames_third_person:
                    path_3rd = os.path.join(args.video_dir, f"ep{episode_idx:02d}_{tag}_third_person.mp4")
                    with imageio.get_writer(path_3rd, fps=10, format="ffmpeg", codec="libx264",
                                            output_params=["-preset", "ultrafast", "-crf", "23"]) as w:
                        for f in frames_third_person:
                            w.append_data(f)
                    logger.info(f"Saved third-person video: {path_3rd} ({len(frames_third_person)} frames)")
                if frames_first_person:
                    path_1st = os.path.join(args.video_dir, f"ep{episode_idx:02d}_{tag}_first_person.mp4")
                    with imageio.get_writer(path_1st, fps=10, format="ffmpeg", codec="libx264",
                                            output_params=["-preset", "ultrafast", "-crf", "23"]) as w:
                        for f in frames_first_person:
                            w.append_data(f)
                    logger.info(f"Saved first-person video (left): {path_1st} ({len(frames_first_person)} frames)")
                if frames_right_wrist:
                    path_rw = os.path.join(args.video_dir, f"ep{episode_idx:02d}_{tag}_right_wrist.mp4")
                    with imageio.get_writer(path_rw, fps=10, format="ffmpeg", codec="libx264",
                                            output_params=["-preset", "ultrafast", "-crf", "23"]) as w:
                        for f in frames_right_wrist:
                            w.append_data(f)
                    logger.info(f"Saved first-person video (right): {path_rw} ({len(frames_right_wrist)} frames)")
            elif imageio is None and frames_third_person:
                logger.warning("imageio not installed, skipping video saving (pip install imageio[ffmpeg])")

            successes = sum(1 for s, _ in episode_results if s)
            total_steps_list = [st for _, st in episode_results]
            avg_success = successes / len(episode_results) * 100
            avg_steps = np.mean(total_steps_list)
            std_steps = np.std(total_steps_list)

            print(f"\n{'─' * 60}")
            print(f"  Episode {episode_idx} finished: {result_str} | steps: {episode_steps}")
            print(f"  Cumulative: {successes}/{len(episode_results)} succeeded "
                  f"| success rate {avg_success:.1f}% | avg steps {avg_steps:.1f} ± {std_steps:.1f}")
            print(f"{'─' * 60}")

        # ── All episodes completed ──
        print(f"\n{'═' * 60}")
        print(f"  All {num_episodes} episodes completed!")
        successes = sum(1 for s, _ in episode_results if s)
        avg_success = successes / len(episode_results) * 100
        all_steps = [st for _, st in episode_results]
        avg_steps = np.mean(all_steps)
        std_steps = np.std(all_steps)
        print(f"  Final success rate: {successes}/{num_episodes} = {avg_success:.1f}%")
        print(f"  Avg steps: {avg_steps:.1f} ± {std_steps:.1f}")
        print(f"{'═' * 60}\n")

        # Per-episode details
        print("Episode details:")
        for i, (s, st) in enumerate(episode_results, 1):
            print(f"  Episode {i:2d}: {'success' if s else 'failure'} | {st} steps")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        if episode_results:
            successes = sum(1 for s, _ in episode_results if s)
            avg_success = successes / len(episode_results) * 100
            all_steps = [st for _, st in episode_results]
            avg_steps = np.mean(all_steps)
            std_steps = np.std(all_steps)
            print(f"\nStats at interruption: {successes}/{len(episode_results)} succeeded "
                  f"| success rate {avg_success:.1f}% | avg steps {avg_steps:.1f} ± {std_steps:.1f}")
    except Exception:
        logger.exception("Main loop exception")
    finally:
        kb.stop()
        # ── Stop the plotting process ──
        if plot_queue is not None and plot_proc is not None:
            stop_plotter(plot_queue, plot_proc)
        logger.info("Shutting down ...")
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        logger.info("Done.")


if __name__ == "__main__":
    main()
