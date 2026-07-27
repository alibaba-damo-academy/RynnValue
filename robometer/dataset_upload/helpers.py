#!/usr/bin/env python3
"""
Helper functions for Robometer model dataset conversion.
Contains utility functions for processing frames, saving images, and managing data.
"""

import os
import subprocess as sp
import uuid

import cv2
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


def save_frame_as_image(frame_data: np.ndarray, output_path: str) -> str:
    """Save a frame as a JPG image."""
    # Convert from HDF5 format to PIL Image
    if frame_data.dtype != np.uint8:
        frame_data = (frame_data * 255).astype(np.uint8)

    image = Image.fromarray(frame_data)
    image.save(output_path, "JPEG", quality=95)
    return output_path


def downsample_frames(frames: np.ndarray | list, max_frames: int = 32) -> np.ndarray | list:
    """Downsample frames to at most max_frames using linear interpolation."""
    # If max_frames is -1, don't downsample
    if max_frames == -1:
        return frames

    if len(frames) <= max_frames:
        return frames

    # Use linear interpolation to downsample
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)

    # keep unique frames
    unique_indices = np.unique(indices)

    # Handle both list and numpy array inputs
    if isinstance(frames, list):
        return [frames[i] for i in unique_indices]
    else:
        return frames[unique_indices]


def motion_aware_downsample(frames: np.ndarray, max_frames: int = 32) -> np.ndarray:
    if len(frames) <= max_frames:
        return frames
    T = len(frames)
    resize_long_side = 256
    min_gap = 1

    def _prep(f):
        if resize_long_side:
            h, w = f.shape[:2]
            scale = resize_long_side / max(h, w)
            if scale < 1.0:
                f = cv2.resize(f, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gray = [_prep(f) for f in frames]

    scores = np.zeros(T, dtype=np.float32)
    fb_args = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }
    for i in range(T - 1):
        flow = cv2.calcOpticalFlowFarneback(gray[i], gray[i + 1], None, **fb_args)
        scores[i + 1] = np.linalg.norm(flow, axis=-1).mean()

    keep = {0, T - 1}
    if max_frames > 2:
        for idx in np.argsort(scores)[::-1]:
            if len(keep) >= max_frames:
                break
            if all(abs(idx - k) >= min_gap for k in keep):
                keep.add(int(idx))

    return frames[sorted(keep)]


def create_trajectory_video(
    frames,
    output_dir: str,
    max_frames: int = -1,
    fps: int = 10,
    shortest_edge_size: int = 240,
    center_crop: bool = False,
) -> str:
    """Create a trajectory video from frames and save as MP4 file."""
    # Handle numpy array of frames (traditional case)
    if not isinstance(frames, np.ndarray):
        frames = np.array(frames)

    # Downsample frames
    frames = downsample_frames(frames, max_frames)

    # Get video dimensions from first frame
    if len(frames) == 0:
        raise ValueError("No frames provided for video creation")

    height, width = frames[0].shape[:2]

    # First, optionally center crop to min(height, width)
    if center_crop:
        # Calculate crop coordinates for center crop
        crop_h = min(height, width)
        y_start = max((height - crop_h) // 2, 0)
        x_start = max((width - crop_h) // 2, 0)
        frames = frames[y_start : y_start + crop_h, x_start : x_start + crop_h]
        height, width = frames[0].shape[:2]

    # Figure out target dimensions for all frames
    if height != width:
        scale_factor = shortest_edge_size / min(height, width)
        target_height = int(height * scale_factor)
        target_width = int(width * scale_factor)
    else:
        target_height = height
        target_width = width

    # Create sequence directory and video file path
    video_path = os.path.join(output_dir, "trajectory.mp4")
    print(f"Saving video to: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (target_width, target_height))

    if not video_writer.isOpened():
        raise Exception("Could not create video writer with any codec")

    # Write frames to video
    for frame in frames:
        # Ensure frame is in uint8 format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # Resize frame to target dimensions if needed
        if frame.shape[:2] != (target_height, target_width):
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        video_writer.write(frame)

    # Release video writer
    video_writer.release()

    return video_path


def create_trajectory_video_optimized(
    frames,
    video_path: str,
    max_frames: int = -1,
    fps: int = 10,
    shortest_edge_size: int = 240,
    center_crop: bool = False,
) -> str:
    """
    Creates a web-optimized trajectory video using a memory-efficient FFmpeg pipe.

    Args:
        frames (list or np.ndarray): A list or array of frames (as RGB, uint8 arrays).
        output_dir (str): Directory to save the video.
        max_frames (int): Maximum number of frames to include in the video.
        fps (int): Frames per second for the output video.
        shortest_edge_size (int): The target size for the shortest edge of the video.
        center_crop (bool): If True, center crop frames to a square before resizing.

    Returns:
        str: The path to the created video file.
    """
    # print(f"Saving optimized video to: {video_path}")
    if os.path.exists(video_path):
        # print(f"Video already exists at: {video_path}, skipping video creation")
        return video_path

    # If frames is callable, call it to get the actual frames
    if callable(frames):
        frames = frames()  # Load frames on-demand
    else:
        frames = frames  # Already loaded frames (legacy datasets)

    if frames is None:
        return None
    if len(frames) == 0:
        raise ValueError("No frames provided for video creation")

    # Downsample frames by selecting indices, which is memory-cheap
    processed_frames = downsample_frames(frames, max_frames)

    # Get dimensions from the first frame
    first_frame = processed_frames[0]
    height, width = first_frame.shape[:2]

    # Determine crop and target dimensions before starting the loop
    if center_crop:
        crop_size = min(height, width)
        y_start = max((height - crop_size) // 2, 0)
        x_start = max((width - crop_size) // 2, 0)
        # After cropping, the frame is a square
        height, width = crop_size, crop_size

    if shortest_edge_size is not None:
        scale_factor = shortest_edge_size / min(height, width)
        target_width = int(width * scale_factor)
        target_height = int(height * scale_factor)

        # Ensure dimensions are even, as required by some codecs like H.264
        target_width = target_width if target_width % 2 == 0 else target_width + 1
        target_height = target_height if target_height % 2 == 0 else target_height + 1
    else:
        target_height, target_width = height, width

    # FFmpeg command for creating a web-optimized H.264 video
    # This pipes raw video frames from stdin
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{target_width}x{target_height}",  # Final size of frames sent to pipe
        "-pix_fmt",
        "bgr24",  # OpenCV provides BGR frames
        "-r",
        str(fps),
        "-i",
        "-",  # Input comes from stdin
        "-an",  # No audio
        "-c:v",
        "libx264",  # Use the H.264 codec
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",  # Pixel format for maximum web compatibility
        "-movflags",
        "+faststart",  # CRITICAL: For web streaming
        video_path,
    ]

    # Start the FFmpeg subprocess
    process = sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)

    # Check if process started successfully
    if process.poll() is not None:
        stderr = process.stderr.read().decode()
        print(f"FFmpeg failed to start. Command: {' '.join(command)}")
        print(f"Error: {stderr}")
        raise RuntimeError("FFmpeg process failed to start")

    for i, frame in enumerate(processed_frames):
        # Ensure frame is in uint8 format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # Apply transformations one frame at a time
        if center_crop:
            frame = frame[y_start : y_start + crop_size, x_start : x_start + crop_size]

        # Resize frame to target dimensions
        if frame.shape[0] != target_height or frame.shape[1] != target_width:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Convert RGB to BGR for FFmpeg pipe
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write the raw frame data to the process's stdin
        try:
            process.stdin.write(frame.tobytes())
        except BrokenPipeError as e:
            stderr = process.stderr.read().decode()
            print(f"BrokenPipeError writing frame. FFmpeg stderr: {stderr}")
            print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
            raise RuntimeError(f"Failed to write frame to FFmpeg: {e}")

    # Close the pipe and finish the process
    process.stdin.close()
    process.wait()

    # Check for errors
    stderr = process.stderr.read().decode()
    if process.returncode != 0:
        print("FFmpeg Error:")
        print(stderr)
        raise RuntimeError("FFmpeg process failed to encode the video.")

    # print("Video created successfully.")
    return video_path


def create_trajectory_sequence(
    frames: list[str], output_dir: str, sequence_name: str, max_frames: int = -1
) -> list[str]:
    """Create a trajectory sequence from frames and save as images."""

    sequence_dir = os.path.join(output_dir, sequence_name)
    os.makedirs(sequence_dir, exist_ok=True)

    # Downsample frames
    frames = downsample_frames(frames, max_frames)

    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = os.path.join(sequence_dir, f"frame_{i:02d}.jpg")
        saved_path = save_frame_as_image(frame, frame_path)
        frame_paths.append(saved_path)

    return frame_paths


def generate_unique_id() -> str:
    """Generate a unique UUID for dataset entries."""
    return str(uuid.uuid4())


def create_hf_trajectory(
    traj_dict: dict,
    video_path: str,
    lang_vector: np.ndarray,
    max_frames: int = -1,
    dataset_name: str = "",
    use_video: bool = True,
    fps: int = 10,
    shortest_edge_size: int = 240,
    center_crop: bool = False,
    hub_repo_id: str | None = None,
) -> dict:
    """Create a HuggingFace dataset trajectory with unified frame loading."""

    # Handle frames - could be np.array, callable, or missing
    frames_data = traj_dict.get("frames")
    if frames_data is None:
        raise ValueError("Trajectory must contain 'frames'")

    video_path = create_trajectory_video_optimized(
        frames_data, video_path, max_frames, fps, shortest_edge_size, center_crop
    )

    if video_path is None:
        print(f"Skipping trajectory {traj_dict.get('id', 'UNKNOWN')} because frames are None")
        return None

    # Get identifiers and fields
    id = traj_dict.get("id", generate_unique_id())
    task_description = traj_dict["task"]
    is_robot: bool = bool(traj_dict.get("is_robot", False))
    quality_label: str = str(traj_dict.get("quality_label", "successful"))
    preference_group_id = traj_dict.get("preference_group_id", None)
    preference_rank = traj_dict.get("preference_rank", None)
    partial_success = traj_dict.get("partial_success", None)
    data_source = traj_dict.get("data_source", dataset_name)

    # Create dataset trajectory
    trajectory = {
        "id": id,
        "task": task_description,
        "lang_vector": lang_vector,  # Pre-computed language vector
        "data_source": data_source,
        "frames": video_path,
        "is_robot": is_robot,
        "quality_label": quality_label,
        "preference_group_id": preference_group_id,
        "preference_rank": preference_rank,
        "partial_success": partial_success,
    }

    return trajectory


def load_sentence_transformer_model() -> SentenceTransformer:
    """Load the sentence transformer model for language embeddings."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def create_output_directory(output_dir: str) -> None:
    """Create the output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def flatten_task_data(task_data: dict[str, list[dict]]) -> list[dict]:
    """Flatten task data into a list of trajectories."""
    all_trajectories = []
    for task_name, trajectories in task_data.items():
        for trajectory in trajectories:
            trajectory["task_name"] = task_name
            all_trajectories.append(trajectory)
    return all_trajectories
