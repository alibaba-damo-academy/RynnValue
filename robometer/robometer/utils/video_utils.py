import base64
import io
import os
import random
from typing import Any, Optional, Union

import cv2
import numpy as np
from PIL import Image

from robometer.data.datasets.helpers import load_frames_from_npz


def extract_frames_from_video(video_path: str, fps: int = 1) -> np.ndarray:
    """
    Extract frames from video file at specified FPS.

    Args:
        video_path: Path to the .mp4 file
        fps: Frames per second to extract (default: 1)

    Returns:
        numpy array of shape (num_frames, H, W, C) with extracted frames
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate frame interval for target FPS
    frame_interval = max(1, int(video_fps / fps))

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_count += 1

    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return np.array(frames)


def _ensure_numpy_frames(frames: Any, frames_shape: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Ensure frames are a numpy array of shape (T, H, W, C).

    Accepts bytes (with shape), numpy array, list of numpy frames, or single frame.
    """
    if frames is None:
        return np.empty((0,))

    # Bytes -> numpy using provided shape
    if isinstance(frames, (bytes, bytearray)):
        if frames_shape is None:
            # Fallback: interpret as uint8 flat array (cannot reshape reliably)
            arr = np.frombuffer(frames, dtype=np.uint8)
            return arr
        if isinstance(frames_shape, list):
            frames_shape = tuple(frames_shape)
        try:
            return np.frombuffer(frames, dtype=np.uint8).reshape(frames_shape)
        except Exception:
            return np.frombuffer(frames, dtype=np.uint8)

    # Already a numpy array
    if isinstance(frames, np.ndarray):
        return frames

    # List of numpy arrays
    if isinstance(frames, list) and all(isinstance(f, np.ndarray) for f in frames):
        return np.stack(frames, axis=0)

    # Unsupported (e.g., file paths) – return as empty; upstream should handle
    return np.empty((0,))


def decode_frames_b64(frames_b64: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for s in frames_b64:
        buf = io.BytesIO(base64.b64decode(s))
        img = Image.open(buf).convert("RGB")
        images.append(img)
    return images


def frames_to_base64_images(frames: Any, frames_shape: tuple[int, int, int, int] | None = None) -> list[str]:
    """Convert frames to a list of base64-encoded JPEG strings.

    Frames can be ndarray (T,H,W,C), bytes + shape, list of ndarray, or a single frame.
    """
    arr = _ensure_numpy_frames(frames, frames_shape)
    if arr.size == 0:
        return []

    # Normalize to (T, H, W, C)
    if arr.ndim == 3:  # single frame (H,W,C)
        arr = arr[None, ...]
    elif arr.ndim != 4:
        # Unknown shape: cannot encode reliably
        return []

    encoded: list[str] = []
    for i in range(arr.shape[0]):
        frame = arr[i]
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return encoded


def add_text_overlay(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int] = (10, 10),
    font_scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    bg_color: Optional[tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Add text overlay to a frame.

    Args:
        frame: Frame in (H, W, C) format, uint8, RGB
        text: Text to add
        position: (x, y) position of text (bottom-left corner of text)
        font_scale: Font scale
        color: Text color (RGB format, will be converted to BGR for cv2)
        thickness: Text thickness
        bg_color: Optional background color (RGB format, will be converted to BGR for cv2)

    Returns:
        Frame with text overlay (H, W, C), RGB
    """
    frame_with_text = frame.copy()

    # Ensure frame has 3 channels
    if frame_with_text.ndim != 3 or frame_with_text.shape[2] != 3:
        raise ValueError(f"Expected frame with shape (H, W, 3), got {frame_with_text.shape}")

    # Convert RGB to BGR for cv2 (cv2 uses BGR)
    frame_bgr = cv2.cvtColor(frame_with_text, cv2.COLOR_RGB2BGR)

    # Convert colors from RGB to BGR for cv2
    color_bgr = (color[2], color[1], color[0])
    bg_color_bgr = (bg_color[2], bg_color[1], bg_color[0]) if bg_color is not None else None

    # Get text size for background box
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # Draw background box if specified
    if bg_color_bgr is not None:
        cv2.rectangle(
            frame_bgr,
            (position[0] - 5, position[1] - text_height - baseline - 5),
            (position[0] + text_width + 5, position[1] + 5),
            bg_color_bgr,
            -1,
        )

    # Draw text (position is bottom-left corner)
    cv2.putText(frame_bgr, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, thickness, cv2.LINE_AA)

    # Convert back to RGB
    frame_with_text = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    return frame_with_text


def create_video_grid_with_progress(
    video_frames_list: list[Optional[np.ndarray]],
    trajectory_progress_data: list[Union[list[float], list[int], np.ndarray, None]],
    grid_size: tuple[int, int] = (3, 3),
    max_videos: int = 9,
    is_discrete_mode: bool = False,
) -> Optional[np.ndarray]:
    """
    Create a grid of videos with progress information overlaid on each video.

    Args:
        video_frames_list: List of videos, each in (T, C, H, W) format or None
        trajectory_progress_data: List of progress_pred values, one per video.
                                  Each element is a list/array of progress predictions.
        grid_size: Tuple of (rows, cols) for the grid
        max_videos: Maximum number of videos to sample
        is_discrete_mode: Whether predictions are in discrete mode

    Returns:
        Grid video in (T, C, H, W) format, or None if insufficient valid videos
    """
    # Filter out None videos
    valid_videos = [(idx, v) for idx, v in enumerate(video_frames_list) if v is not None]

    grid_cells = grid_size[0] * grid_size[1]
    if len(valid_videos) == 0:
        return None

    # Sample available videos (up to grid_cells)
    num_to_sample = min(grid_cells, len(valid_videos))
    sampled_videos = random.sample(valid_videos, num_to_sample)

    # Get corresponding progress data (assume alignment)
    valid_items = [(v_idx, v, trajectory_progress_data[v_idx]) for v_idx, v in sampled_videos]

    # Find maximum time dimension across valid videos
    max_time = max(v.shape[0] for _, v, _ in valid_items) if valid_items else 1

    # Resize and normalize videos to same size for grid
    processed_videos = []
    target_h, target_w = 128, 128  # Target size for each cell in grid

    for _, video, progress_pred in valid_items:
        # Pad video to max_time by repeating the last frame
        if video.shape[0] < max_time:
            padding = np.repeat(video[-1:], max_time - video.shape[0], axis=0)
            video = np.concatenate([video, padding], axis=0)

        # Pad progress_pred to match video length if needed
        if progress_pred is not None and isinstance(progress_pred, (list, np.ndarray)):
            pred_arr = np.array(progress_pred) if isinstance(progress_pred, list) else progress_pred
            if len(pred_arr) < max_time:
                padding = np.repeat([pred_arr[-1]], max_time - len(pred_arr), axis=0)
                progress_pred = np.concatenate([pred_arr, padding], axis=0)
            else:
                progress_pred = pred_arr

        # Convert (T, C, H, W) to (T, H, W, C) for processing
        video = video.transpose(0, 2, 3, 1)

        # Resize each frame and add progress overlay
        resized_frames = []
        for t, frame in enumerate(video):
            # Ensure uint8 format
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frame_resized = cv2.resize(frame, (target_w, target_h))

            # Add progress text overlay in bottom right corner
            if progress_pred is not None:
                # Get progress value for this frame (index by t)
                if isinstance(progress_pred, (list, np.ndarray)):
                    pred_elem = progress_pred[t]
                    if is_discrete_mode:
                        # Discrete mode: apply argmax to get predicted bin index
                        pred_val = float(np.argmax(pred_elem))
                    else:
                        # Continuous mode: convert to float directly
                        pred_val = float(pred_elem)
                else:
                    pred_val = float(progress_pred)

                # Format text (only show predicted value)
                progress_text = f"P:{pred_val:.2f}"

                # Calculate position (bottom right, with padding)
                text_x = target_w - 60
                text_y = target_h - 10

                # Add text with background
                frame_resized = add_text_overlay(
                    frame_resized,
                    progress_text,
                    position=(text_x, text_y),
                    font_scale=0.4,
                    color=(255, 255, 255),  # White text
                    thickness=1,
                    bg_color=(0, 0, 0),  # Black background
                )

            resized_frames.append(frame_resized)

        # Convert back to (T, C, H, W)
        video_resized = np.array(resized_frames).transpose(0, 3, 1, 2)
        processed_videos.append(video_resized)

    # Fill remaining grid cells with black videos
    num_black_videos = grid_cells - len(processed_videos)
    for _ in range(num_black_videos):
        # Create black video: (T, C, H, W) with all zeros (black)
        black_video = np.zeros((max_time, 3, target_h, target_w), dtype=np.uint8)
        processed_videos.append(black_video)

    # Arrange videos in grid
    grid_frames = []
    for t in range(max_time):
        grid_rows = []
        for row in range(grid_size[0]):
            row_frames = []
            for col in range(grid_size[1]):
                vid_idx = row * grid_size[1] + col
                frame = processed_videos[vid_idx][t]  # (C, H, W)
                # Convert to (H, W, C) for concatenation
                frame = frame.transpose(1, 2, 0)
                row_frames.append(frame)
            # Concatenate horizontally
            row_concat = np.concatenate(row_frames, axis=1)  # (H, total_W, C)
            grid_rows.append(row_concat)
        # Concatenate vertically
        grid_frame = np.concatenate(grid_rows, axis=0)  # (total_H, total_W, C)
        # Convert back to (C, H, W)
        grid_frame = grid_frame.transpose(2, 0, 1)
        grid_frames.append(grid_frame)

    # Stack frames: (T, C, H, W)
    grid_video = np.stack(grid_frames, axis=0)
    return grid_video


def create_frame_pair_with_progress(
    eval_result: dict, target_h: int = 224, target_w: int = 224, is_discrete_mode: bool = False
) -> Optional[np.ndarray]:
    """
    Create a horizontal row of frames from a trajectory with progress annotations.
    The number of frames is determined by the length of progress_pred or target_progress.
    Each row represents one trajectory, with frames displayed horizontally.

    Args:
        eval_result: Evaluation result dict with video_path, progress_pred, target_progress
        target_h: Target height for frames
        target_w: Target width for frames

    Returns:
        Single row image with frames horizontally concatenated in (C, H, W) format, or None if unavailable
    """
    video_path = eval_result.get("video_path")
    if video_path is None:
        return None

    # Get progress values to determine number of frames
    progress_pred = eval_result.get("progress_pred")
    target_progress = eval_result.get("target_progress")

    # Determine number of frames from progress_pred or target_progress length
    if isinstance(progress_pred, (list, np.ndarray)) and len(progress_pred) > 0:
        num_frames = len(progress_pred)
    elif isinstance(target_progress, (list, np.ndarray)) and len(target_progress) > 0:
        num_frames = len(target_progress)
    else:
        # Fallback: if both are scalars or None, default to 1 frame
        num_frames = 1

    # Load frames from video
    frames = load_frames_from_npz(video_path)
    frames = frames.transpose(0, 3, 1, 2)  # (T, C, H, W)

    if frames.shape[0] < 1:
        return None

    # Process frames - use evenly spaced indices to sample frames from video
    # This handles cases where video has more frames than progress_pred indicates
    processed_frames = []
    for i in range(num_frames):
        # Calculate frame index - evenly space across available frames
        if num_frames == 1:
            frame_idx = 0
        else:
            frame_idx = int((i / (num_frames - 1)) * (frames.shape[0] - 1)) if frames.shape[0] > 1 else 0
        frame_idx = min(frame_idx, frames.shape[0] - 1)

        frame = frames[frame_idx]  # (C, H, W)
        frame_hwc = frame.transpose(1, 2, 0)  # (H, W, C)
        if frame_hwc.dtype != np.uint8:
            frame_hwc = np.clip(frame_hwc * 255, 0, 255).astype(np.uint8)
        frame_resized = cv2.resize(frame_hwc, (target_w, target_h))

        # Get progress values for this frame index
        if isinstance(progress_pred, (list, np.ndarray)):
            pred_elem = progress_pred[i]
            if is_discrete_mode:
                # Discrete mode: apply argmax to get predicted bin index
                pred_val = float(np.argmax(pred_elem))
            else:
                # Continuous mode: convert to float directly
                pred_val = float(pred_elem)
        else:
            pred_val = float(progress_pred) if progress_pred is not None else 0.0

        if isinstance(target_progress, (list, np.ndarray)):
            # Target is already a discrete bin index (integer) in discrete mode, or continuous in continuous mode
            target_val = float(target_progress[i])
        else:
            target_val = float(target_progress) if target_progress is not None else 0.0

        # Add progress annotation to frame
        progress_text = f"P:{pred_val:.2f} T:{target_val:.2f}"
        text_x = target_w - 110
        text_y = target_h - 10
        frame_resized = add_text_overlay(
            frame_resized,
            progress_text,
            position=(text_x, text_y),
            font_scale=0.4,
            color=(255, 255, 255),
            thickness=1,
            bg_color=(0, 0, 0),
        )

        processed_frames.append(frame_resized)

    # Concatenate all frames horizontally
    if len(processed_frames) == 0:
        return None
    combined_frame = np.concatenate(processed_frames, axis=1)  # (H, num_frames*W, C)

    # Add text label above the frame row (task and quality_label/partial_success)
    task = eval_result.get("task")
    quality_label = eval_result.get("quality_label")
    partial_success = eval_result.get("partial_success")

    # Build label text parts (before wrapping)
    label_parts = []
    if task is not None:
        label_parts.append(f"Task: {task}")
    if partial_success is not None:
        # RoboArena: use partial_success
        label_parts.append(f"Partial: {partial_success:.2f}")
    elif quality_label is not None:
        # Non-RoboArena: use quality_label
        label_parts.append(f"Quality: {quality_label}")

    if label_parts:
        # Get available width for text (frame width - padding on both sides)
        available_width = combined_frame.shape[1] - 20  # 10px padding on each side
        font_scale = 0.5
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Helper function to wrap text to fit available width
        def wrap_text(text, max_width):
            """Wrap text into multiple lines that fit within max_width."""
            words = text.split()
            lines = []
            current_line = words[0] if words else ""

            for word in words[1:]:
                # Check if adding the next word exceeds width
                test_line = current_line + " " + word
                (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

                if text_width <= max_width:
                    current_line = test_line
                else:
                    # Current line is full, start a new line
                    lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            return lines

        # Wrap each label part and collect all lines
        label_lines = []
        for part in label_parts:
            wrapped = wrap_text(part, available_width)
            label_lines.extend(wrapped)

        # Calculate label height based on number of lines
        line_height = 20  # Height per line
        line_spacing = 5  # Spacing between lines
        label_height = len(label_lines) * line_height + (len(label_lines) - 1) * line_spacing + 10  # Add padding

        # Ensure minimum label height
        min_label_height = 30
        if label_height < min_label_height:
            label_height = min_label_height

        label_frame = np.ones((label_height, combined_frame.shape[1], 3), dtype=np.uint8) * 255  # White background

        # Add each line of text
        for i, label_text in enumerate(label_lines):
            y_position = label_height - 5 - (len(label_lines) - 1 - i) * (line_height + line_spacing)
            label_frame = add_text_overlay(
                label_frame,
                label_text,
                position=(10, y_position),  # Left-aligned
                font_scale=font_scale,
                color=(0, 0, 0),  # Black text
                thickness=thickness,
                bg_color=None,  # No background needed (already white)
            )

        # Concatenate label above the frame row
        combined_frame = np.concatenate([label_frame, combined_frame], axis=0)  # (H + label_height, num_frames*W, C)

    # Convert back to (C, H, W) format (no time dimension)
    combined_frame_chw = combined_frame.transpose(2, 0, 1)  # (C, H + label_height, num_frames*W)

    return combined_frame_chw


def create_policy_ranking_grid(
    eval_results: list[dict],
    grid_size: tuple[int, int] = (2, 2),
    max_samples: int = 4,
    border_width: int = 4,
    is_discrete_mode: bool = False,
) -> Optional[np.ndarray]:
    """
    Create a vertical stack of trajectory rows, each showing all frames horizontally.
    Each row represents one trajectory with all its frames displayed horizontally.

    Args:
        eval_results: List of evaluation results with video_path, progress_pred, target_progress
        grid_size: Not used anymore (kept for compatibility), but ignored
        max_samples: Maximum number of samples (trajectories) to display (default: 4)
        border_width: Width of border between rows in pixels

    Returns:
        Stacked image in (H, W, C) format, uint8, RGB, or None if unavailable
    """
    # Filter results with valid video_paths
    valid_results = [r for r in eval_results if r.get("video_path") is not None]

    if len(valid_results) == 0:
        return None

    # Limit to max_samples
    num_to_sample = min(max_samples, len(valid_results))

    # Sample random results
    sampled_results = random.sample(valid_results, num_to_sample)

    # Create frame rows for each sampled result (each row is one trajectory)
    frame_rows = []
    target_h, target_w = 224, 224

    for result in sampled_results:
        frame_row = create_frame_pair_with_progress(result, target_h, target_w, is_discrete_mode=is_discrete_mode)
        if frame_row is not None:
            frame_rows.append(frame_row)

    if len(frame_rows) == 0:
        return None

    # Get the maximum width across all rows to align them
    max_width = max(row.shape[2] for row in frame_rows)  # (C, H, W) -> W dimension
    max_height = max(row.shape[1] for row in frame_rows)  # (C, H, W) -> H dimension

    # Pad all rows to the same width and height
    aligned_rows = []
    border_color = np.array([128, 128, 128], dtype=np.uint8)  # Gray border

    for row in frame_rows:
        # Convert to (H, W, C) for processing
        row_hwc = row.transpose(1, 2, 0)  # (H, W, C)
        current_height, current_width = row_hwc.shape[0], row_hwc.shape[1]

        # Pad width if necessary (pad on the right)
        if current_width < max_width:
            width_padding = max_width - current_width
            right_padding = np.ones((current_height, width_padding, 3), dtype=np.uint8) * 255  # White padding
            row_hwc = np.concatenate([row_hwc, right_padding], axis=1)

        # Pad height if necessary (pad on the bottom)
        if current_height < max_height:
            height_padding = max_height - current_height
            bottom_padding = np.ones((height_padding, row_hwc.shape[1], 3), dtype=np.uint8) * 255  # White padding
            row_hwc = np.concatenate([row_hwc, bottom_padding], axis=0)

        aligned_rows.append(row_hwc)

    # Stack rows vertically with borders between them
    stacked_rows = []
    for i, row in enumerate(aligned_rows):
        stacked_rows.append(row)
        # Add horizontal border below (except for last row)
        if i < len(aligned_rows) - 1:
            h_border = np.tile(border_color, (border_width, row.shape[1], 1))
            stacked_rows.append(h_border)

    # Concatenate vertically
    if len(stacked_rows) == 0:
        return None

    grid_frame = np.concatenate(stacked_rows, axis=0)  # (total_H, max_W, C)

    # Return as static image in (H, W, C) format, uint8
    if grid_frame.dtype != np.uint8:
        grid_frame = np.clip(grid_frame, 0, 255).astype(np.uint8)

    return grid_frame
