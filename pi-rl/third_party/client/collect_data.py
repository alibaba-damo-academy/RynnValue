import os
import shutil
import time
from typing import Dict

import cv2
import imageio
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from droid.misc.time import time_ms

from client.real_utils.spacemouse import SpaceMousePolicy
from droid.trajectory_utils.trajectory_writer import TrajectoryWriter

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "save_root",
    "data/pick_and_place_cube",
    "Root directory where collected trajectories will be stored.",
)
flags.DEFINE_integer(
    "num_episodes",
    0,
    "Number of successful trajectories to collect. Set to 0 to run indefinitely.",
)
config_flags.DEFINE_config_file(
    "task_config",
    "configs/task/pick.py",
    "File path to the task configuration.",
    lock_config=False,
)
flags.DEFINE_bool(
    "test_detector",
    False,
    "If True, never return when done; reset env and keep the collection loop running (for testing the detector).",
)
# Saved MP4 resolution (width, height); low-res to save disk and encoding time
flags.DEFINE_integer("video_save_width", 320, "Width of saved MP4 frames.")
flags.DEFINE_integer("video_save_height", 240, "Height of saved MP4 frames.")


def smallest_missing_id(dir_path: str) -> int:
    os.makedirs(dir_path, exist_ok=True)
    ids = set()
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if os.path.isdir(p) and name.isdigit():
            ids.add(int(name))
    i = 0
    while i in ids:
        i += 1
    return i


def collect_trajectory(
    env,
    controller,
    save_filepath=None,
    recording_folderpath=False,
    test_detector=False,
):
    controller.reset_state()
    env.camera_reader.set_trajectory_mode()

    traj_writer = None
    if save_filepath:
        traj_writer = TrajectoryWriter(save_filepath, metadata=None, save_images=False)

    # Stream MP4 and HDF5 to avoid holding full episode in memory
    mp4_writers = {}
    video_dir = None

    t_reset0 = time.perf_counter()
    env.reset()
    print("[between-episode] env.reset()={:.2f}s (start of episode)".format(time.perf_counter() - t_reset0))

    start_recording = False
    _episode_success = None

    try:
        while True:
            time_start = time_ms()
            controller_info = controller.get_info()
            control_timestamps = {"step_start": time_ms(), "step_end": time_ms()}
            t_after_controller = time_ms()

            read_camera_start = time_ms()
            obs = env.get_raw_observation()
            read_camera_end = time_ms()
            t_before_transform = time_ms()
            saved_obs = env.transform_observation(obs)
            t_after_transform = time_ms()
            done, success, _, _ = env.get_info_for_step(obs)
            t_after_obs = time_ms()

            obs["controller_info"] = controller_info

            control_timestamps["policy_start"] = time_ms()
            action, controller_action_info = controller.forward(obs, include_info=True)

            control_timestamps["sleep_start"] = time_ms()
            comp_time = time_ms() - control_timestamps["step_end"]
            sleep_left = (1 / env.control_hz) - (comp_time / 1000)

            if sleep_left > 0:
                time.sleep(sleep_left)

            control_timestamps["control_start"] = time_ms()
            action_info = env.step(action)

            control_timestamps["step_end"] = time_ms()
            action_info.update(controller_action_info)

            obs["timestamp"]["control"] = control_timestamps
            # HDF5: saved_observation + action only (smaller files; training ignores raw observation).
            timestep = {"saved_observation": saved_obs, "action": action_info}

            if traj_writer is not None and start_recording:
                traj_writer.write_timestep(timestep)
                vid_size = (FLAGS.video_save_width, FLAGS.video_save_height)
                for key in mp4_writers:
                    f = np.asarray(saved_obs[key], dtype=np.uint8)
                    if f.ndim == 2:
                        f = np.stack([f, f, f], axis=-1)
                    if (f.shape[1], f.shape[0]) != vid_size:
                        f = cv2.resize(f, vid_size, interpolation=cv2.INTER_LINEAR)
                    mp4_writers[key].append_data(f)

            if (not start_recording) and controller_info.get("movement_enabled", False) and recording_folderpath:
                env.camera_reader.start_recording(recording_folderpath)
                start_recording = True
                video_dir = os.path.join(os.path.dirname(recording_folderpath), "recordings", "MP4")
                os.makedirs(video_dir, exist_ok=True)
                # Open MP4 writers and stream frames (no in-memory buffer)
                for key, val in saved_obs.items():
                    if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[-1] == 3:
                        out_path = os.path.join(video_dir, f"{key}.mp4")
                        mp4_writers[key] = imageio.get_writer(
                            out_path, fps=30, format="ffmpeg", codec="libx264",
                            output_params=["-preset", "ultrafast", "-crf", "28"],
                        )
                print("start recording (streaming traj + MP4; low memory)")

            t_after_timestep = time_ms()

            if done:
                if test_detector:
                    print(f"Done (success={success}); test_detector=True")
                    continue
                _episode_success = success
                result = {
                    "success": success,
                    "failure": not success,
                    "info": controller_info,
                }
                return result
            time_end = time_ms()
            # Block timings (ms)
            ts = control_timestamps
            print(
                "timing ms:",
                "controller=", t_after_controller - time_start,
                "read_camera=", read_camera_end - read_camera_start,
                "transform_obs=", t_after_transform - t_before_transform,
                "get_info=", t_after_obs - t_after_transform,
                "transform_get_info_total=", t_after_obs - read_camera_end,
                "policy=", ts["sleep_start"] - ts["policy_start"],
                "sleep=", ts["control_start"] - ts["sleep_start"],
                "env_step=", ts["step_end"] - ts["control_start"],
                "timestep_append=", time_end - t_after_timestep,
                "| total=", time_end - time_start,
            )
    finally:
        t0 = time.perf_counter()
        try:
            if recording_folderpath:
                env.camera_reader.stop_recording()
        except Exception as e:
            print("Warning: stop_recording error:", e)
        t_stop_rec = time.perf_counter()

        for key, writer in mp4_writers.items():
            try:
                writer.close()
            except Exception:
                pass

        if _episode_success is False:
            if traj_writer is not None:
                try:
                    traj_writer.close()
                except Exception:
                    pass
            print("[between-episode] stop_recording={:.2f}s (failure — skipped hdf5 flush)".format(
                t_stop_rec - t0))
        else:
            t_mp4 = time.perf_counter()
            try:
                if traj_writer is not None:
                    traj_writer.close(metadata=controller_info if "controller_info" in locals() else None)
            except Exception as e:
                print("Warning: traj_writer close error:", e)
            t_hdf5 = time.perf_counter()
            print(
                "[between-episode] stop_recording={:.2f}s close_mp4={:.2f}s close_hdf5={:.2f}s total={:.2f}s".format(
                    t_stop_rec - t0, t_mp4 - t_stop_rec, t_hdf5 - t_mp4, t_hdf5 - t0
                )
            )

def run_and_route_one(
    env, controller, base_dir
) -> Dict[str, object]:
    tmp_root = os.path.join(base_dir, "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    session_name = f"session_{int(time.time())}"
    tmp_dir = os.path.join(tmp_root, session_name)
    images_dir = os.path.join(tmp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    save_filepath = os.path.join(tmp_dir, "traj.hdf5")

    print("Temp session dir:", tmp_dir)
    print("Temp traj file:", save_filepath)
    print("Temp recording path:", images_dir)

    print("Start collecting")
    result = collect_trajectory(
        env,
        controller=controller,
        save_filepath=save_filepath,
        recording_folderpath=images_dir,
        test_detector=FLAGS.test_detector,
    )

    success = result.get("success", False)
    print(f"Outcome: {'success' if success else 'failure'}")

    if not success:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Failure — discarded temp data.")
        return {"dest_dir": None, "id": None, "result": result}

    outcome_root = os.path.join(base_dir, "success")
    new_id = smallest_missing_id(outcome_root)
    dest_dir = os.path.join(outcome_root, str(new_id))
    os.makedirs(outcome_root, exist_ok=True)

    print("Assigning id:", new_id)
    print("Destination:", dest_dir)

    t_move0 = time.perf_counter()
    shutil.move(tmp_dir, dest_dir)
    print("[between-episode] move={:.2f}s".format(time.perf_counter() - t_move0))
    print(
        "Result saved:",
        os.path.join(dest_dir, "traj.hdf5"),
        os.path.join(dest_dir, "images"),
        os.path.join(dest_dir, "recordings", "MP4"),
    )
    return {"dest_dir": dest_dir, "id": new_id, "result": result}


def main(_):
    base_dir = FLAGS.save_root
    os.makedirs(base_dir, exist_ok=True)

    # use cartesian velocity and velocity for collecting data
    FLAGS.task_config.action_space = "cartesian_velocity"
    FLAGS.task_config.gripper_action_space = "velocity"
    
    env = FLAGS.task_config.env(**FLAGS.task_config)
    # Teleop collection: end episodes on success/detector/manual/bounds only — not step budget.
    env.ignore_auto_reset = True
    task_config = FLAGS.task_config
    controller = SpaceMousePolicy(
        max_lin_vel=task_config.collect_max_lin_vel,
        max_rot_vel=task_config.collect_max_rot_vel,
    )

    if FLAGS.test_detector:
        print("test_detector=True: running collection loop without saving; on done will reset and continue (Ctrl+C to stop).")
        collect_trajectory(
            env,
            controller=controller,
            save_filepath=None,
            recording_folderpath=False,
            test_detector=True,
        )
        return

    episode = 0
    successful_episodes = 0
    while True:
        episode += 1

        print(f"\n{'=' * 60}")
        print(f"Starting trajectory collection #{episode} (Successful: {successful_episodes}/{FLAGS.num_episodes if FLAGS.num_episodes > 0 else '∞'})")
        print(f"{'=' * 60}\n")

        result = run_and_route_one(env, controller, base_dir)
        
        if result["result"]["success"]:
            successful_episodes += 1
            print(f"\n{'=' * 60}")
            print(f"Completed successful trajectory #{successful_episodes} (Total attempts: {episode})")
            print(f"{'=' * 60}\n")

        if FLAGS.num_episodes > 0 and successful_episodes >= FLAGS.num_episodes:
            print(f"\n{'=' * 60}")
            print(f"Reached target of {FLAGS.num_episodes} successful episodes after {episode} total attempts")
            print(f"{'=' * 60}\n")
            break


if __name__ == "__main__":
    app.run(main)
