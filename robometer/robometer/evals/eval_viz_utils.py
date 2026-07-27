#!/usr/bin/env python3
"""
Utility functions for visualization in RBM evaluations.
"""

from typing import Optional
import os
import logging
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import decord

logger = logging.getLogger(__name__)


def create_combined_progress_success_plot(
    progress_pred: np.ndarray,
    num_frames: int,
    success_binary: Optional[np.ndarray] = None,
    success_probs: Optional[np.ndarray] = None,
    success_labels: Optional[np.ndarray] = None,
    is_discrete_mode: bool = False,
    title: Optional[str] = None,
    loss: Optional[float] = None,
    pearson: Optional[float] = None,
) -> plt.Figure:
    """Create a combined plot with progress, success binary, and success probabilities.

    This function creates a unified plot with 1 subplot (progress only) or 3 subplots
    (progress, success binary, success probs), similar to the one used in compile_results.py.

    Args:
        progress_pred: Progress predictions array
        num_frames: Number of frames
        success_binary: Optional binary success predictions
        success_probs: Optional success probability predictions
        success_labels: Optional ground truth success labels
        is_discrete_mode: Optional; unused, kept for API compatibility
        title: Optional title for the plot (if None, auto-generated from loss/pearson)
        loss: Optional loss value to display in title
        pearson: Optional pearson correlation to display in title

    Returns:
        matplotlib Figure object
    """
    # Determine if we should show success plots
    has_success_binary = success_binary is not None and len(success_binary) == len(progress_pred)

    if has_success_binary:
        # Three subplots: progress, success (binary), success_probs
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.5))
        ax = axs[0]  # Progress subplot
        ax2 = axs[1]  # Success subplot (binary)
        ax3 = axs[2]  # Success probs subplot
    else:
        # Single subplot: progress only
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax2 = None
        ax3 = None

    # Plot progress
    ax.plot(progress_pred, linewidth=2)
    ax.set_ylabel("Progress")

    # Build title
    if title is None:
        title_parts = ["Progress"]
        if loss is not None:
            title_parts.append(f"Loss: {loss:.3f}")
        if pearson is not None:
            title_parts.append(f"Pearson: {pearson:.2f}")
        title = ", ".join(title_parts)
    fig.suptitle(title)

    # Set y-limits and ticks (always continuous since discrete is converted before this function)
    ax.set_ylim(0, 1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(y_ticks)

    # Setup success binary subplot
    if ax2 is not None:
        ax2.step(range(len(success_binary)), success_binary, where="post", linewidth=2, label="Predicted", color="blue")
        # Add ground truth success labels as green line if available
        if success_labels is not None and len(success_labels) == len(success_binary):
            ax2.step(
                range(len(success_labels)),
                success_labels,
                where="post",
                linewidth=2,
                label="Ground Truth",
                color="green",
            )
        ax2.set_ylabel("Success (Binary)")
        ax2.set_ylim(-0.05, 1.05)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_yticks([0, 1])
        ax2.legend()

    # Setup success probs subplot if available
    if ax3 is not None and success_probs is not None:
        ax3.plot(range(len(success_probs)), success_probs, linewidth=2, label="Success Prob", color="purple")
        # Add ground truth success labels as green line if available
        if success_labels is not None and len(success_labels) == len(success_probs):
            ax3.step(
                range(len(success_labels)),
                success_labels,
                where="post",
                linewidth=2,
                label="Ground Truth",
                color="green",
                linestyle="--",
            )
        ax3.set_ylabel("Success Probability")
        ax3.set_ylim(-0.05, 1.05)
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax3.legend()

    plt.tight_layout()
    return fig


def extract_frames(video_path: str, fps: float = 1.0, max_frames: int = 64) -> np.ndarray:
    """Extract frames from video file as numpy array (T, H, W, C).

    Supports both local file paths and URLs (e.g., HuggingFace Hub URLs).
    Uses the provided ``fps`` to control how densely frames are sampled from
    the underlying video, but caps the total number of frames at ``max_frames``
    to prevent memory issues.

    Args:
        video_path: Path to video file or URL
        fps: Frames per second to extract (default: 1.0)
        max_frames: Maximum number of frames to extract (default: 64). This prevents
            memory issues with long videos or high FPS settings.

    Returns:
        numpy array of shape (T, H, W, C) containing extracted frames, or None if error
    """
    if video_path is None:
        return None

    if isinstance(video_path, tuple):
        video_path = video_path[0]

    # Check if it's a URL or local file
    is_url = video_path.startswith(("http://", "https://"))
    is_local_file = os.path.exists(video_path) if not is_url else False

    if not is_url and not is_local_file:
        logger.warning(f"Video path does not exist: {video_path}")
        return None

    try:
        # decord.VideoReader can handle both local files and URLs
        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)

        # Determine native FPS; fall back to a reasonable default if unavailable
        try:
            native_fps = float(vr.get_avg_fps())
        except Exception:
            native_fps = 1.0

        # If user-specified fps is invalid or None, default to native fps
        if fps is None or fps <= 0:
            fps = native_fps

        # Compute how many frames we want based on desired fps
        # num_frames ≈ total_duration * fps = total_frames * (fps / native_fps)
        if native_fps > 0:
            desired_frames = int(round(total_frames * (fps / native_fps)))
        else:
            desired_frames = total_frames

        # Clamp to [1, total_frames]
        desired_frames = max(1, min(desired_frames, total_frames))

        # IMPORTANT: Cap at max_frames to prevent memory issues
        # This is critical when fps is high or videos are long
        if desired_frames > max_frames:
            logger.warning(
                f"Requested {desired_frames} frames but capping at {max_frames} "
                f"to prevent memory issues (video has {total_frames} frames at {native_fps:.2f} fps, "
                f"requested extraction at {fps:.2f} fps)"
            )
            desired_frames = max_frames

        # Evenly sample indices to match the desired number of frames
        if desired_frames == total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int).tolist()

        frames_array = vr.get_batch(frame_indices).asnumpy()  # Shape: (T, H, W, C)
        del vr
        return frames_array
    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {e}")
        return None
