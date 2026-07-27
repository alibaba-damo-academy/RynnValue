#!/usr/bin/env python3
"""
Video helper utilities for robust loading of videos with codec handling and
backend fallbacks. Centralized here to be reused across dataset loaders.
"""

import gc
import os
import shutil
import subprocess
import tempfile
import time
from signal import SIGTERM
from typing import Optional

import cv2
import numpy as np

# Configuration constants
DEFAULT_SUBPROCESS_TIMEOUT = 30  # seconds for ffprobe/ffmpeg operations
DEFAULT_FRAME_READ_TIMEOUT = 60  # seconds for reading all frames from a video
MAX_FRAMES_SANITY_CHECK = 10000  # maximum number of frames to prevent infinite loops


def _ffprobe_codec_name(path: str, timeout: int = DEFAULT_SUBPROCESS_TIMEOUT) -> str | None:
    """Return codec_name for the first video stream using ffprobe, or None on failure."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nk=1:nw=1",
                path,
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout,
        )
        codec_name = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else None
        return codec_name
    except subprocess.TimeoutExpired:
        print(f"Warning: ffprobe timed out after {timeout}s for {path}")
        return None
    except Exception as e:
        print(f"Warning: ffprobe failed for {path}: {e}")
        return None


def _reencode_to_h264(input_path: str, timeout: int = DEFAULT_SUBPROCESS_TIMEOUT) -> str | None:
    """Re-encode input video to H.264 yuv420p if ffmpeg is available. Returns output path or None."""
    if shutil.which("ffmpeg") is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
        output_path = tmp_out.name
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_path,
            ],
            check=True,
            timeout=timeout,
        )
        return output_path
    except subprocess.TimeoutExpired:
        print(f"Warning: ffmpeg re-encode timed out after {timeout}s for {input_path}")
        try:
            os.unlink(output_path)
        except Exception:
            pass
        return None
    except Exception as e:
        print(f"Warning: ffmpeg re-encode failed for {input_path}: {e}")
        try:
            os.unlink(output_path)
        except Exception:
            pass
        return None


def _open_with_best_backend(path: str) -> cv2.VideoCapture | None:
    """Try multiple OpenCV backends and return an opened capture or None."""
    backends: list[int] = [getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY), cv2.CAP_ANY]
    for backend in backends:
        try:
            cap_try = cv2.VideoCapture(path, backend)
            if cap_try.isOpened():
                # Validate we can read at least one frame
                ret, test_frame = cap_try.read()
                if ret and test_frame is not None:
                    cap_try.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return cap_try
            cap_try.release()
        except Exception:
            try:
                cap_try.release()
            except Exception:
                pass
    return None


def load_video_frames(
    video_input,
    timeout: Optional[int] = DEFAULT_FRAME_READ_TIMEOUT,
    max_frames: int = MAX_FRAMES_SANITY_CHECK,
) -> np.ndarray:
    """Load video frames (RGB uint8) from a file path (str/Path) or video bytes.

    - For byte inputs, detect AV1 and re-encode to H.264 for compatibility
    - Uses OpenCV with FFMPEG backend when available
    - Returns numpy array of shape (T, H, W, 3) in RGB order
    - Protected by timeout and maximum frame count to prevent hangs

    Args:
        video_input: Path to video file or video bytes
        timeout: Maximum time in seconds to spend reading frames (None to disable)
        max_frames: Maximum number of frames to read (sanity check against infinite loops)

    Raises:
        TimeoutError: If frame reading exceeds the timeout
        ValueError: If video cannot be opened or no frames extracted
    """
    temp_files_to_cleanup: list[str] = []
    cap: cv2.VideoCapture | None = None
    video_path_str = str(video_input) if isinstance(video_input, (str, os.PathLike)) else "bytes"

    if isinstance(video_input, (str, os.PathLike)):
        video_path = str(video_input)
        decodable_path = video_path
        # If file is AV1, re-encode to H.264 so OpenCV can decode (many platforms lack AV1)
        codec_name = _ffprobe_codec_name(video_path)
        if codec_name == "av1":
            h264_path = _reencode_to_h264(video_path, timeout=120)
            if h264_path is not None:
                temp_files_to_cleanup.append(h264_path)
                decodable_path = h264_path
        cap = _open_with_best_backend(decodable_path)
    else:
        # Save bytes to temp file first
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(video_input)
            temp_input_path = temp_file.name
        temp_files_to_cleanup.append(temp_input_path)

        # If the input is AV1, transcode to H.264 for compatibility
        codec_name = _ffprobe_codec_name(temp_input_path)
        decodable_path = temp_input_path
        if codec_name == "av1":
            h264_path = _reencode_to_h264(temp_input_path)
            if h264_path is not None:
                temp_files_to_cleanup.append(h264_path)
                decodable_path = h264_path
        cap = _open_with_best_backend(decodable_path)

    try:
        frames: list[np.ndarray] = []
        if cap is None or not cap.isOpened():
            raise ValueError(
                f"Could not open video file {video_path_str} with available backends. "
                "If the source is AV1, install AV1 support or enable ffmpeg re-encode."
            )

        start_time = time.time()
        frame_count = 0

        while True:
            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Video frame reading timed out after {timeout}s for {video_path_str}. "
                    f"Read {frame_count} frames before timeout."
                )

            # Check max frames sanity limit
            if frame_count >= max_frames:
                print(
                    f"Warning: Reached maximum frame limit ({max_frames}) for {video_path_str}. "
                    f"This might indicate a corrupted or unusually long video."
                )
                frames = []
                gc.collect()
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB for consistency
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            frame_count += 1
        cap.release()

        # Clean up temp file(s) if we created any
        for path in temp_files_to_cleanup:
            try:
                os.unlink(path)
            except Exception:
                pass

        if len(frames) == 0:
            raise ValueError(f"No frames could be extracted from video {video_path_str}")

        return np.array(frames)

    except Exception as e:
        if cap is not None:
            cap.release()
        # Clean up temp files in case of error
        for path in temp_files_to_cleanup:
            try:
                os.unlink(path)
            except Exception:
                pass
        # Add more context to the error
        if isinstance(e, (TimeoutError, ValueError)):
            raise
        raise RuntimeError(f"Error loading video {video_path_str}: {e}") from e
