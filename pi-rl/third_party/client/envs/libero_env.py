import math
import pathlib

import cv2
import numpy as np

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from client.envs.utils import process_image_for_obs


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class LiberoEnv:
    """LIBERO simulation environment with the same interface as DroidEnv.

    Wraps ``OffScreenRenderEnv`` and translates observations into the dict
    format consumed by the training pipeline (matching the keys produced by
    ``DroidEnv.transform_observation``).
    """

    def __init__(
        self,
        bddl_file_name=None,
        task_suite_name="libero_spatial",
        task_id=0,
        image_size=None,
        env_resolution=256,
        auto_reset_steps=300,
        language_instruction=None,
        video_dir=None,
        seed=7,
        num_steps_wait=10,
        **kwargs,
    ):
        if bddl_file_name is None:
            from libero.libero.benchmark import get_benchmark
            benchmark = get_benchmark(task_suite_name)()
            task = benchmark.get_task(task_id)
            bddl_file_name = str(
                pathlib.Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            )
            if language_instruction is None:
                language_instruction = task.language
            self._init_states = benchmark.get_task_init_states(task_id)
        else:
            self._init_states = None

        self._env_resolution = env_resolution
        env_args = {
            "bddl_file_name": bddl_file_name,
            "camera_heights": env_resolution,
            "camera_widths": env_resolution,
        }
        self._env = OffScreenRenderEnv(**env_args)
        self._env.seed(seed)

        self.image_size = image_size
        self.auto_reset_steps = auto_reset_steps
        self.language_instruction = language_instruction or ""
        self.video_dir = video_dir
        self.num_steps_wait = num_steps_wait

        self.done = False
        self.success = False
        self.reward = 0.0
        self.info = {}
        self._steps_since_reset = 0
        self._episode_idx = 0
        self._raw_frame_buffer = []
        self._ep_count = 0

    def reset(self):
        self._steps_since_reset = 0
        self._raw_frame_buffer = []
        obs = self._env.reset()

        if self._init_states is not None:
            idx = self._episode_idx % len(self._init_states)
            obs = self._env.set_init_state(self._init_states[idx])
            self._episode_idx += 1

        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self._env.step([0.0] * 6 + [-1.0])

        self.done = False
        self.success = False
        self.reward = 0.0
        return self._transform_observation(obs)

    def _transform_observation(self, obs):
        agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

        agentview = process_image_for_obs(agentview, image_size=self.image_size)
        wrist = process_image_for_obs(wrist, image_size=self.image_size)

        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
        eef_axisangle = _quat2axisangle(eef_quat.copy())
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
        state = np.concatenate([eef_pos, eef_axisangle, gripper_qpos])

        return {
            "image": agentview,
            "wrist_image": wrist,
            "state": state,
            "prompt": self.language_instruction,
        }

    def get_observation(self):
        obs = self._env.env._get_observations()
        data = self._transform_observation(obs)
        if self.video_dir and obs is not None:
            frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            self._raw_frame_buffer.append(frame)
        return data

    def step(self, action):
        self._steps_since_reset += 1
        action = np.asarray(action, dtype=np.float64)[:7]
        obs, reward, done, info = self._env.step(action.tolist())
        executed_action = action.copy()
        return {"executed_action": executed_action}

    def get_info_for_step(self, raw_obs=None):
        success = self._env.check_success()
        time_stop = self._steps_since_reset >= self.auto_reset_steps
        done = bool(success or time_stop)

        if done and self.video_dir and self._raw_frame_buffer:
            self._save_video()

        self.done = done
        self.success = success
        self.reward = 1.0 if success else 0.0
        reward = self.reward
        mask = 0.0 if done else 1.0
        return done, success, reward, mask

    def _save_video(self):
        import os
        os.makedirs(self.video_dir, exist_ok=True)
        path = os.path.join(self.video_dir, f"ep_{self._ep_count:04d}.mp4")
        h, w = self._raw_frame_buffer[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        for frame in self._raw_frame_buffer:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        self._raw_frame_buffer = []
        self._ep_count += 1

    def detect(self, raw_obs):
        return self._env.check_success(), False

    def close(self):
        try:
            self._env.close()
            del self._env
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
