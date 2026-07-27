#!/usr/bin/env python3
"""VLAC baseline for progress prediction.

VLAC (Vision-Language-Action-Critic) is a general-purpose pair-wise critic model
designed for real-world robot reinforcement learning and data refinement.
It provides robust evaluation capabilities for task progress prediction and
task completion verification based on images and task descriptions.

Reference: https://github.com/InternRobotics/VLAC
Model: https://huggingface.co/InternRobotics/VLAC

Downloading the model:
    # Option 1: Use Hugging Face CLI
    pip install huggingface_hub
    huggingface-cli download InternRobotics/VLAC

    # Option 2: Use Python API (automatic in VLAC.__init__)
    from robometer.evals.baselines.vlac import VLAC
    model = VLAC(model_path="InternRobotics/VLAC")  # Auto-downloads if not found

    # Option 3: Use download function directly
    from robometer.evals.baselines.vlac import download_vlac_model
    model_path = download_vlac_model(local_dir="./models/vlac")
"""

import os
import tempfile
from typing import List, Dict, Optional
import numpy as np
import cv2

from robometer.utils.logger import get_logger

# Disable tqdm globally before importing evo_vlac to prevent progress bars
try:
    import tqdm

    # Monkey-patch tqdm to always disable before evo_vlac imports it
    _original_tqdm = tqdm.tqdm

    def _disabled_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return _original_tqdm(*args, **kwargs)

    tqdm.tqdm = _disabled_tqdm
except ImportError:
    pass  # tqdm not available, nothing to patch

try:
    from evo_vlac import GAC_model
    from evo_vlac.utils.video_tool import compress_video

    VLAC_AVAILABLE = True
except ImportError:
    VLAC_AVAILABLE = False

try:
    from huggingface_hub import snapshot_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

logger = get_logger()


def download_vlac_model(
    repo_id: str = "InternRobotics/VLAC", cache_dir: Optional[str] = None, local_dir: Optional[str] = None
) -> str:
    """
    Download VLAC model from Hugging Face.

    Args:
        repo_id: Hugging Face repository ID (default: "InternRobotics/VLAC")
        cache_dir: Optional cache directory for Hugging Face downloads
        local_dir: Optional local directory to download model to (if None, uses cache)

    Returns:
        Path to the downloaded model directory

    Example:
        >>> model_path = download_vlac_model()
        >>> model_path = download_vlac_model(local_dir="./models/vlac")
    """
    if not HF_AVAILABLE:
        raise ImportError("huggingface_hub is required to download models. Install with: pip install huggingface_hub")

    print(f"Downloading VLAC model from {repo_id}...")
    model_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Use actual files, not symlinks
    )
    print(f"Model downloaded to: {model_path}")
    return model_path


class VLAC:
    """VLAC baseline for progress prediction using pair-wise comparison.

    VLAC uses a pair-wise comparison mechanism to predict task progress
    for each frame in a trajectory. It requires a local model checkpoint
    to be loaded.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        model_type: str = "internvl2",
        temperature: float = 0.5,
        top_k: int = 1,
        batch_size: int = 5,
        skip: int = 5,
        frame_skip: bool = True,
        auto_download: bool = True,
        use_images: bool = False,
    ):
        """
        Initialize VLAC model.

        Args:
            model_path: Path to local VLAC model checkpoint, or Hugging Face repo ID
                       (e.g., "InternRobotics/VLAC"). If repo ID and model doesn't exist
                       locally, will download automatically if auto_download=True.
            device: Device to run model on (e.g., "cuda:0")
            model_type: Model type (default: "internvl2")
            temperature: Temperature for generation
            top_k: Top-k sampling
            batch_size: Batch size for processing
            skip: Pair-wise step size
            frame_skip: Whether to skip frames for efficiency
            auto_download: If True and model_path is a Hugging Face repo ID, automatically
                          download the model if not found locally
            use_images: If True, use image mode (get_trajectory_critic with image files).
                       If False, use video mode (web_trajectory_critic with video file).
        """
        # Check if model_path exists locally first
        if not os.path.exists(model_path):
            # If not found locally and looks like a Hugging Face repo ID (contains "/" but not an absolute path)
            # and auto_download is enabled, download from Hugging Face
            is_hf_repo_id = (
                "/" in model_path
                and not os.path.isabs(model_path)
                and not model_path.startswith(".")
                and not model_path.startswith("~")
            )

            if is_hf_repo_id and auto_download:
                if not HF_AVAILABLE:
                    raise ImportError(
                        "huggingface_hub is required to download models. Install with: pip install huggingface_hub"
                    )
                print(f"Model path '{model_path}' not found locally. Downloading from Hugging Face...")
                model_path = download_vlac_model(repo_id=model_path)
            elif not is_hf_repo_id:
                raise FileNotFoundError(
                    f"Model path '{model_path}' does not exist. "
                    f"To download from Hugging Face, use model_path='InternRobotics/VLAC'"
                )

        self.model_path = model_path
        self.device = device
        self.model_type = model_type
        self.temperature = temperature
        self.top_k = top_k
        self.batch_size = batch_size
        self.skip = skip
        self.frame_skip = frame_skip
        self.use_images = use_images

        # Initialize model
        self.critic = GAC_model(tag="critic")
        self.critic.init_model(model_path=model_path, model_type=model_type, device_map=device)
        self.critic.temperature = temperature
        self.critic.top_k = top_k
        self.critic.set_config()
        self.critic.set_system_prompt()

    def compute_progress(
        self, frames_array: np.ndarray, task_description: str = "", reference_video_path: Optional[str] = None
    ) -> List[Optional[float]]:
        """
        Compute progress predictions for frames using VLAC baseline.

        Args:
            frames_array: (N, H, W, 3) uint8 array from trajectory frames
            task_description: Task description text
            reference_video_path: Optional path to reference video for in-context learning

        Returns:
            List of task progress values in [0, 1] range for each frame.
            Values are normalized from VLAC output (which may be [0, 100] or [0, 1]).
            If no predictions, returns list of zeros.
        """
        if frames_array is None or frames_array.size == 0:
            return []

        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as tmpdir:
            if self.use_images:
                # Image mode: save frames as image files and use get_trajectory_critic
                image_list = []
                for i, frame in enumerate(frames_array):
                    # Ensure frame is uint8
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    # Save frame as image
                    image_path = os.path.join(tmpdir, f"frame_{i:05d}.jpg")
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(image_path, frame_bgr)
                    image_list.append(image_path)

                # Handle reference images if provided
                ref_image_list = []
                if reference_video_path:
                    # Load reference video frames
                    ref_cap = cv2.VideoCapture(reference_video_path)
                    ref_idx = 0
                    while True:
                        ret, frame = ref_cap.read()
                        if not ret:
                            break
                        # Save reference frame
                        ref_image_path = os.path.join(tmpdir, f"ref_frame_{ref_idx:05d}.jpg")
                        cv2.imwrite(ref_image_path, frame)
                        ref_image_list.append(ref_image_path)
                        ref_idx += 1
                    ref_cap.release()

                # Run VLAC trajectory critic with images
                # Note: get_trajectory_critic returns (critic_list, value_list) for pair-wise comparisons
                critic_list, value_list = self.critic.get_trajectory_critic(
                    task=task_description,
                    image_list=image_list,
                    ref_image_list=ref_image_list if ref_image_list else None,
                    batch_num=self.batch_size,
                    ref_num=len(ref_image_list) if ref_image_list else 0,
                    rich=True,  # Output decimal values (True for [0,1] range, False for integer percentage)
                    reverse_eval=False,
                )
            else:
                # Video mode: use web_trajectory_critic with video file
                video_path = os.path.join(tmpdir, "trajectory.mp4")
                self._frames_to_video(frames_array, video_path, fps=1.0)

                # Compress video (VLAC expects compressed video)
                compressed_video = os.path.join(tmpdir, "compressed.mp4")
                _, output_fps = compress_video(video_path, compressed_video, fps=1.0)

                # Verify compressed video shape matches original frames
                cap = cv2.VideoCapture(compressed_video)
                compressed_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB for consistency
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    compressed_frames.append(frame_rgb)
                cap.release()
                compressed_frames_array = np.array(compressed_frames)  # [T, H, W, C] in RGB

                # Check shape match - frames_array is [T, H, W, C] format (checked in _frames_to_video)
                assert compressed_frames_array.shape[0] == frames_array.shape[0], (
                    f"Shape mismatch: original {frames_array.shape} vs compressed {compressed_frames_array.shape}"
                )

                # Run VLAC trajectory critic with video
                # Note: web_trajectory_critic returns (result_path, value_list, critic_list, done_list)
                # where value_list contains progress values in [0, 1] if rich=True, or [0, 100] if rich=False
                result_path, value_list, critic_list, done_list = self.critic.web_trajectory_critic(
                    task_description=task_description,
                    main_video_path=compressed_video,
                    reference_video_path=None,
                    batch_num=self.batch_size,
                    ref_num=0,  # Number of reference images from reference video
                    think=False,  # Whether to use Chain-of-Thought reasoning
                    skip=1,  # predict per frame
                    rich=True,  # Output decimal values (True for [0,1] range, False for integer percentage)
                    reverse_eval=False,  # Whether to reverse evaluation (for VROC evaluation)
                    output_path=tmpdir,
                    fps=float(output_fps),
                    frame_skip=self.frame_skip,  # Whether to skip frames for efficiency (if False, evaluate each frame)
                    done_flag=False,  # Whether to output done/task completion flag
                    in_context_done=False,  # Whether to use reference video for done prediction
                    done_threshold=0.9,  # Threshold for task completion
                    video_output=False,  # Whether to output annotated video
                )

            logger.info(f"value_list: {value_list}")

            # Extract progress values from value_list
            # value_list contains progress predictions for each frame
            # VLAC may return values in [0, 100] or [0, 1] range, normalize to [0, 1] if needed
            if value_list and len(value_list) > 0:
                # Convert to list of floats
                progress_list = [float(val) for val in value_list]

                # Check if values need normalization from [0, 100] to [0, 1]
                max_val = max(progress_list) if progress_list else 0.0
                if max_val > 1.0:
                    # Normalize from [0, 100] to [0, 1]
                    progress_list = [val / 100.0 for val in progress_list]

                # Ensure we have predictions for all frames
                num_frames = frames_array.shape[0]
                if len(progress_list) < num_frames:
                    # Pad with last value
                    progress_list.extend([progress_list[-1]] * (num_frames - len(progress_list)))
                elif len(progress_list) > num_frames:
                    # Truncate to match frame count
                    progress_list = progress_list[:num_frames]

                return np.array(progress_list)
            else:
                # Return zeros for all frames if no predictions
                return np.array([0.0] * frames_array.shape[0])

    def _frames_to_video(self, frames_array: np.ndarray, output_path: str, fps: float = 5.0):
        """Convert frames array to video file."""
        height, width = frames_array.shape[1], frames_array.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames_array:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
