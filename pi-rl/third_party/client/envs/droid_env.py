import time

import cv2
import numpy as np

from droid.robot_env import RobotEnv
from client.envs.utils import process_image_for_obs
from client.real_utils.vis_utils import raw_frame_from_raw_obs, save_episode_video as save_episode_video_to_disk
from client.real_utils.detector import PickBlocksDetector
from client.real_utils.detector import LightPlugDetector
from client.real_utils.detector import success_detector_manual

    
class DroidEnv(RobotEnv):
    def __init__(
        self,
        action_space = "joint_velocity",
        gripper_action_space = "position",
        auto_reset_steps = 0,
        ignore_auto_reset = False,
        bounds = None,
        reset_joints = None,
        reset_random = False,
        randomize_low = None,
        randomize_high = None,
        image_size = None,
        action_keys = None,
        side_camera_id = None,
        wrist_camera_id = None,
        language_instruction = None,
        control_hz = None,
        video_dir = None,
        camera_intrinsics = None,
        camera_extrinsics = None,
        record_camera = None,
        **kwargs,
    ):
        super().__init__(
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            reset_joints=reset_joints,
            randomize_low=randomize_low,
            randomize_high=randomize_high,
        )
        self.camera_reader.set_trajectory_mode()
        
        self.bounds = bounds
        self.auto_reset_steps = auto_reset_steps
        self.ignore_auto_reset = ignore_auto_reset
        self._steps_since_reset = 0

        self.reset_random = reset_random
        self.image_size = image_size
        self.side_camera_id = side_camera_id
        self.wrist_camera_id = wrist_camera_id
        self.record_camera = record_camera
        self.language_instruction = language_instruction
        self._camera_intrinsics = camera_intrinsics  # optional fallback for action viz (dict cam_id -> 3x3 K)
        self._camera_extrinsics = camera_extrinsics  # optional fallback (dict cam_id -> 6D pose)

        # for handling environment errors   
        self.done, self.success, self.reward, self.info = False, False, 0, {}
        self._last_gripper_velocity = 0.0
        self.prev_obs = None
        self.control_hz = control_hz
        self.video_dir = video_dir
        self._raw_frame_buffer = []
        self._record_frame_buffer = []
        self._ep_count = 0

    def reset(self):
        self._before_reset()
        self._steps_since_reset = 0
        self._raw_frame_buffer = []
        self._record_frame_buffer = []
        super().reset(randomize=self.reset_random)
        return self.get_observation()

    def _before_reset(self):
        """Override in subclasses to e.g. reset detector sequence and move_up before reset."""
        pass

    def get_info_for_step(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.prev_obs
        time_stop = self.auto_reset_due()
        reached_boundary = self.reached_boundary(raw_obs)

        manual = success_detector_manual()
        if manual == "success":
            done, success, manual_stop = True, True, True
        elif manual == "reset":
            done, success, manual_stop = True, False, True
        else:
            manual_stop = False
            success, terminate = self.detect(raw_obs)
            if terminate: print(f"Terminate: {terminate}")
            done = bool(success or terminate or time_stop)

        if done:
            print(f"Done! Success: {success}, Time stop: {time_stop}, Manual stop: {manual_stop}, Reached boundary: {reached_boundary}")
            if self.video_dir and self._raw_frame_buffer:
                save_episode_video_to_disk(
                    self._raw_frame_buffer, self.video_dir, self._ep_count
                )
                self._raw_frame_buffer = []
            if self.video_dir and self._record_frame_buffer:
                save_episode_video_to_disk(
                    self._record_frame_buffer, self.video_dir, self._ep_count,
                    prefix="record",
                )
                self._record_frame_buffer = []
            if self.video_dir:
                self._ep_count += 1
        self.done, self.success, self.reward, self.info = done, success, 1.0 if success else 0.0, {}
        reward = 1.0 if success else 0.0
        mask = 0.0 if done else 1.0
        return done, success, reward, mask

    def detect(self, raw_obs):
        """Override in subclasses. Base: no detector, always False."""
        raise NotImplementedError("detect not implemented for base DroidEnv")

    def get_raw_observation(self):
        raw_obs = super().get_observation()
        self.prev_obs = raw_obs
        return raw_obs

    def transform_observation(self, raw_obs):
        # match the input of DroidDataset
        side_img = process_image_for_obs(raw_obs["image"][self.side_camera_id], bgr_to_rgb=True, image_size=self.image_size)
        wrist_img = process_image_for_obs(raw_obs["image"][self.wrist_camera_id], bgr_to_rgb=True, image_size=self.image_size)
        data_dict = {
            "exterior_image_1_left": side_img,
            "exterior_image_2_left": side_img,
            "wrist_image_left": wrist_img,
            "cartesian_position": raw_obs["robot_state"]["cartesian_position"],
            "gripper_position": raw_obs["robot_state"]["gripper_position"],
            "prompt": self.language_instruction,
        }
        if "camera_intrinsics" in raw_obs and "camera_extrinsics" in raw_obs:
            intr = dict(raw_obs["camera_intrinsics"])
            if self.image_size is not None:
                for cam_id in [self.side_camera_id, self.wrist_camera_id]:
                    K = intr.get(cam_id)
                    if K is not None and cam_id in raw_obs["image"]:
                        img = raw_obs["image"][cam_id]
                        h_orig, w_orig = img.shape[:2]
                        sx, sy = self.image_size[1] / w_orig, self.image_size[0] / h_orig
                        K = np.array(K, dtype=np.float64).copy()
                        K[0, 0], K[1, 1] = K[0, 0] * sx, K[1, 1] * sy
                        K[0, 2], K[1, 2] = K[0, 2] * sx, K[1, 2] * sy
                        intr[cam_id] = K
            data_dict["camera_intrinsics"] = intr
            data_dict["camera_extrinsics"] = raw_obs["camera_extrinsics"]
        return data_dict

    def get_observation(self):
        raw_obs = self.get_raw_observation()
        if self.video_dir:
            frame = raw_frame_from_raw_obs(raw_obs, self.side_camera_id, self.wrist_camera_id)
            if frame is not None:
                self._raw_frame_buffer.append(frame)
            if self.record_camera and self.record_camera in raw_obs.get("image", {}):
                rec = np.asarray(raw_obs["image"][self.record_camera], dtype=np.uint8)
                if rec.shape[-1] == 4:
                    rec = cv2.cvtColor(rec, cv2.COLOR_BGRA2RGB)
                else:
                    rec = cv2.cvtColor(rec, cv2.COLOR_BGR2RGB)
                self._record_frame_buffer.append(rec)
        return self.transform_observation(raw_obs)

    def reached_boundary(self, raw_obs):
        pos = np.asarray(raw_obs["robot_state"]["cartesian_position"][:3], dtype=np.float64)
        lows, highs = self.bounds[:, 0], self.bounds[:, 1]
        reached = bool((pos <= lows).any() or (pos >= highs).any())
        return reached

    def step(self, action):
        self._steps_since_reset += 1
        
        action = np.asarray(action, dtype=np.float64)
        # to handle different action spaces
        action = action[:self.DoF]

        if self.bounds is not None and self.prev_obs is not None and len(action) >= 3:
            # Block motion that would push further out of cartesian bounds.
            pos = np.asarray(self.prev_obs["robot_state"]["cartesian_position"][:3], dtype=np.float64)
            lows = np.asarray(self.bounds[:, 0], dtype=np.float64)
            highs = np.asarray(self.bounds[:, 1], dtype=np.float64)
            for axis in range(3):
                if (pos[axis] <= lows[axis] and action[axis] < 0) or (
                    pos[axis] >= highs[axis] and action[axis] > 0
                ):
                    action[axis] = 0.0
        executed_action = np.array(action, dtype=np.float64)

        action_info = super().step(action)
        a = np.asarray(action).reshape(-1)
        self._last_gripper_velocity = float(action_info.get("gripper_velocity", a[-1] if len(a) > 0 else 0.0))
        action_info["executed_action"] = executed_action
        return action_info

    def auto_reset_due(self):
        if self.ignore_auto_reset:
            return False
        due = (self._steps_since_reset >= self.auto_reset_steps)
        if due: print(f"Timeout...")
        return due

    @property
    def steps_since_reset(self):
        return self._steps_since_reset

    def close(self):
        try:
            super().close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

class PickBlocksEnv(DroidEnv):
    """Pick-only: success from robot state (partial gripper close, lifted z, gripper closing)."""

    def __init__(
        self,
        move_up_steps=2,
        move_up_velocity=1,
        success_reset_randomize_magnitude=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._move_up_steps = int(move_up_steps)
        self._move_up_velocity = float(move_up_velocity)
        self.success_reset_randomize_magnitude = float(success_reset_randomize_magnitude)
        self.camera_reader.set_trajectory_mode()
        self.pick_detector = None
        self._init_detector()

    def _before_reset(self):
        if self.pick_detector is not None:
            self.pick_detector.reset_sequence()
        # if not getattr(self, "success", False):
        #     return
        # Success: go to reset position with random x/y offset via cartesian_noise (non-blocking)
        mag = self.success_reset_randomize_magnitude
        cartesian_noise = np.array([
            np.random.uniform(-mag, mag),
            np.random.uniform(-mag, mag),
            0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float64)
        # cartesian_noise = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._robot.update_joints(self.reset_joints, velocity=False, blocking=True, cartesian_noise=cartesian_noise)
        print("reset with random x/y offset: ", cartesian_noise)

        # open gripper
        self._robot.update_gripper(0, velocity=False, blocking=True)
        time.sleep(1)
        # move up
        super().move_up(steps=1, velocity=2)
        time.sleep(1)

    def detect(self, raw_obs):
        robot_state = raw_obs["robot_state"]
        return bool(
            self.pick_detector.detect(
                gripper_position=robot_state["gripper_position"],
                cartesian_position=robot_state["cartesian_position"],
                gripper_velocity=getattr(self, "_last_gripper_velocity", None),
            )
        ), False

    def _init_detector(self):
        self.pick_detector = PickBlocksDetector()

class Light2Env(DroidEnv):
    """Light task 2: success when yellow plug is no longer visible (unplugged).

    Uses LightPlugDetector to count yellow pixels in a bounding box ROI.
    Success when yellow pixel count drops below max_yellow_pixels for
    consecutive_frames in a row.
    """

    def __init__(
        self,
        detector_image_key="38651013_left",
        max_yellow_pixels=100,
        consecutive_frames=5,
        detect_size=384,
        pre_reset_joints=None,
        success_reset_randomize_magnitude=0.06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.detector_image_key = detector_image_key
        self.max_yellow_pixels = int(max_yellow_pixels)
        self.consecutive_frames = int(consecutive_frames)
        self.detect_size = detect_size
        self.pre_reset_joints = np.array(pre_reset_joints) if pre_reset_joints is not None else None
        self.success_reset_randomize_magnitude = float(success_reset_randomize_magnitude)
        self.light_detector = None
        self._init_detector()

    def _init_detector(self):
        self.light_detector = LightPlugDetector(
            detector_image_key=self.detector_image_key,
            max_yellow_pixels=self.max_yellow_pixels,
            consecutive_frames=self.consecutive_frames,
            detect_size=self.detect_size,
        )

    def _before_reset(self):
        if self.light_detector is not None:
            self.light_detector.reset_sequence()
        self._robot.update_gripper(1, velocity=False, blocking=True)
        vel = np.array([-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for _ in range(3):
            self.update_robot(vel, action_space="cartesian_velocity", gripper_action_space="velocity", blocking=True)
        if self.pre_reset_joints is not None:
            mag = self.success_reset_randomize_magnitude
            cartesian_noise = np.array([
                np.random.uniform(-mag, mag),
                np.random.uniform(-mag, mag),
                0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float64)
            self._robot.update_joints(self.pre_reset_joints, velocity=False, blocking=True, cartesian_noise=cartesian_noise)
            print("pre-reset with random x/y offset: ", cartesian_noise)
        self._robot.update_gripper(0, velocity=False, blocking=True)
        time.sleep(1)

    def detect(self, raw_obs):
        if self.light_detector is not None:
            return self.light_detector.detect(raw_obs)
        return False, False


__all__ = [
    "DroidEnv",
    "PickBlocksEnv",
    "Light2Env",
]