#!/usr/bin/env python3
"""
Batch collator for processing list of samples.
"""

import tempfile
import uuid
from pathlib import Path
import numpy as np
import torch
from qwen_vl_utils import process_vision_info

from .base import BaseCollator
from .utils import convert_frames_to_pil_images, pad_list_to_max, write_mp4
from robometer.data.dataset_types import PreferenceSample, ProgressSample
from robometer.data.dataset_category import is_preference_only_ds
from robometer.data.datasets.helpers import DataGenStrat
from typing import List, Dict, Union
from robometer.models.utils import convert_discrete_target_to_continuous
from PIL import Image

MAX_IMAGE_SIDE = 480  # bigger side
MAX_IMAGE_PIXELS = 1024 * 1024  # safety cap (1.0 MP). raise to 1.5MP if stable


def _resize_pil(pil: Image.Image, max_side: int = MAX_IMAGE_SIDE, max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    pil = pil.convert("RGB")
    w, h = pil.size

    # Scale down if max side too large
    scale_side = min(1.0, max_side / float(max(w, h)))

    # Scale down if too many pixels (area cap)
    scale_area = (max_pixels / float(w * h)) ** 0.5 if (w * h) > max_pixels else 1.0

    scale = min(scale_side, scale_area)

    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        pil = pil.resize((nw, nh), resample=Image.BICUBIC)

    return pil


def should_compute_progress(
    quality_label: str,
    data_gen_strategy: str,
    data_source: str = None,
    is_chosen: bool = False,
    partial_success: float | None = None,
) -> float:
    """
    Check if progress should be computed for a trajectory.

    Includes if it is successful or rewound
    but NOT suboptimal or failure. Also masks out progress if data_source is in preference_only category,
    except when the strategy is DIFFERENT_TASK and it's the rejected trajectory (where progress should still be computed, but will be 0.0).
    Also includes trajectories with partial_success.

    Args:
        quality_label: The quality label of the trajectory
        data_gen_strategy: The data generation strategy
        data_source: The data source name (optional)
        is_chosen: Whether this is the chosen trajectory (traj A) in a preference sample
        partial_success: Partial success value (0-1) or None. If present, progress should be computed.

    Returns:
        1.0 if progress should be computed, 0.0 otherwise
    """

    # Mask out progress if data_source is in preference_only category
    if data_source is not None and is_preference_only_ds(data_source):
        # For preference_only datasets:
        # - If it's the chosen trajectory, always mask out (don't compute)
        # - If it's the rejected trajectory with DIFFERENT_TASK strategy, still compute (it will be 0.0)
        return 0.0
        # if is_chosen:
        #    return 0.0
        # elif data_gen_strategy == DataGenStrat.DIFFERENT_TASK.value:
        #    return 1.0
        # else:
        #    return 0.0
    # If partial_success and not is_preference_only_ds, always compute progress
    # predict partial success for roboreward trajectories not preference only
    elif partial_success is not None and "roboreward" in data_source:
        return 1.0
    elif quality_label in ["suboptimal", "failure", "failed"]:
        return 0.0
    elif quality_label == "successful" or data_gen_strategy == DataGenStrat.REWIND.value:
        return 1.0

    return 0.0


def create_padding_mask(frames_shapes: torch.Tensor, max_length: int = None) -> torch.Tensor:
    """
    Create padding mask based on frames_shape.

    Args:
        frames_shapes: Tensor of shape (batch_size, ...) where first dim of each row is num_frames
        max_length: Maximum length for padding. If None, uses max of first dim in frames_shapes

    Returns:
        Tensor of shape (batch_size, max_length) with 1.0 for valid frames, 0.0 for padding
    """
    # Extract num_frames from first dimension of each shape
    if frames_shapes.dim() > 1:
        num_frames = frames_shapes[:, 0].float()
    else:
        num_frames = frames_shapes.float()

    if max_length is None:
        max_length = int(num_frames.max().item())

    # Create range tensor: [0, 1, 2, ..., max_length-1]
    range_tensor = torch.arange(max_length, dtype=torch.float32, device=frames_shapes.device)

    # Broadcast comparison: (batch_size, 1) vs (1, max_length) -> (batch_size, max_length)
    # For each sample, positions < num_frames are valid (1.0), others are padding (0.0)
    masks = (range_tensor.unsqueeze(0) < num_frames.unsqueeze(1)).float()

    return masks


class RBMBatchCollator(BaseCollator):
    def __init__(
        self,
        processor,
        tokenizer=None,
        max_length: int = 1024,
        resized_height: int = 128,
        resized_width: int = 128,
        base_model_id: str = None,
        load_embeddings: bool = False,
        use_multi_image: bool = False,
        prog_pref: bool = False,
        use_per_frame_progress_token: bool = False,
        shuffle_progress_frames: bool = False,
        inference: bool = False,
        **kwargs,
    ):
        super().__init__(
            processor=processor,
            tokenizer=tokenizer,
            max_length=max_length,
            resized_height=resized_height,
            resized_width=resized_width,
            base_model_id=base_model_id,
            load_embeddings=load_embeddings,
            **kwargs,
        )
        self.use_multi_image = use_multi_image
        self.prog_pref = prog_pref

        # Molmo2 only supports multi-image mode, not video
        if "Molmo" in self.base_model_id and not self.use_multi_image:
            raise ValueError(
                "Molmo2 does not support video mode (use_multi_image=False). "
                "Please set data.use_multi_image=True to use Molmo2 with multi-image input."
            )
        self.use_per_frame_progress_token = use_per_frame_progress_token
        # Validate that use_per_frame_progress_token requires use_multi_image
        if self.use_per_frame_progress_token and not self.use_multi_image:
            raise ValueError(
                "use_per_frame_progress_token=True requires use_multi_image=True. "
                "Per-frame progress tokens can only be added in multi-image mode."
            )
        self.shuffle_progress_frames = shuffle_progress_frames
        self.inference = inference

    def _prepare_frames_for_conversation(self, frames: List, prefix: str = "tmp") -> tuple[Union[List, str], dict]:
        """
        Prepare frames for conversation based on use_multi_image flag.

        Args:
            frames: List of PIL Images
            prefix: Prefix for temporary video file (if needed)

        Returns:
            tuple: (video_field, content_extras)
                - video_field: Either list of PIL Images (if use_multi_image) or video file path (str)
                - content_extras: Dictionary with resized_height/width or empty dict
        """
        if self.use_multi_image:
            # # Use images directly - return list of PIL Images
            # if self.resized_height is not None and self.resized_width is not None:
            #     content_extras = {
            #         "resized_height": self.resized_height,
            #         "resized_width": self.resized_width,
            #     }
            # else:
            #     frames = [_resize_pil(frame) for frame in frames]
            #     content_extras = {}
            content_extras = {}
            return frames, content_extras
        elif "Qwen" in self.base_model_id or "Molmo" in self.base_model_id:
            # Qwen and Molmo accept list of PIL Images directly
            if self.resized_height is not None and self.resized_width is not None:
                content_extras = {
                    "resized_height": self.resized_height,
                    "resized_width": self.resized_width,
                }
            else:
                frames = [_resize_pil(frame) for frame in frames]
                content_extras = {}
            return frames, content_extras
        elif "SmolVLM" in self.base_model_id:
            frames = [_resize_pil(frame) for frame in frames]
            # Convert to video file for SmolVLM
            unique_id = uuid.uuid4().hex
            tmp = Path(tempfile.gettempdir()) / f"{prefix}_{unique_id}.mp4"
            write_mp4(frames, tmp, fps=1)
            return str(tmp), {}
        else:
            # Default: return frames as-is
            return frames, {}

    def _add_vision_content_to_list(
        self, content_list: List[Dict], frames_or_video: Union[List, str], content_extras: dict
    ) -> None:
        """
        Add vision content (images or video) to a content list.

        Args:
            content_list: List to append vision content to
            frames_or_video: Either list of PIL Images (if use_multi_image) or video file path (str)
            content_extras: Dictionary with additional content parameters
        """
        if self.use_multi_image:
            # Add each image as a separate entry
            for img in frames_or_video:
                content_list.append({
                    "type": "image",
                    "image": img,
                    **content_extras,
                })
                # Add per-frame progress token after each frame if enabled
                if self.use_per_frame_progress_token:
                    content_list.append({"type": "text", "text": "<|prog_token|>"})
        else:
            # Add video entry
            content_list.append({
                "type": "video",
                "video": frames_or_video,
                "sample_fps": 1.0,
                **content_extras,
            })

    def _process_conversation(
        self, conversations: List[List[Dict]], add_generation_prompt: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process a list of conversations into a batch of inputs.

        Args:
            conversations: List of conversations
            add_generation_prompt: Whether to add generation prompt (otherwise conversation will be closed with </im_end> token)
        Returns:
            Batch of inputs
        """
        if "Qwen" in self.base_model_id or "Molmo" in self.base_model_id:
            # Process all messages in one batch
            texts = [
                self.processor.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    add_vision_id=True,
                    enable_thinking=False,
                    fps=1,
                )
                for msg in conversations
            ]

            is_qwen3 = "Qwen3" in self.base_model_id or "Molmo2" in self.base_model_id

            # For Qwen3, pass image_patch_size to process_vision_info
            process_kwargs = {
                "return_video_kwargs": True,
                "return_video_metadata": is_qwen3,
            }
            if (
                is_qwen3
                and hasattr(self.processor, "image_processor")
                and hasattr(self.processor.image_processor, "patch_size")
            ):
                process_kwargs["image_patch_size"] = self.processor.image_processor.patch_size

            image_inputs, video_inputs, video_kwargs = process_vision_info(conversations, **process_kwargs)

            # For Qwen3, video_inputs is a list of (video, video_metadata) tuples
            # that need to be split before passing to processor
            if is_qwen3 and video_inputs is not None and len(video_inputs) > 0:
                # Check if video_inputs contains tuples (Qwen3 format) or is already split
                if isinstance(video_inputs[0], tuple) and len(video_inputs[0]) == 2:
                    videos, video_metadatas = zip(*video_inputs)
                    videos, video_metadatas = list(videos), list(video_metadatas)
                else:
                    # Already in the correct format
                    videos = video_inputs
                    video_metadatas = None
            else:
                videos = video_inputs if video_inputs else None
                video_metadatas = None

            # Process through the processor in one batch
            processor_kwargs = {
                "text": texts,
                "images": image_inputs,
                "padding": True,
                "truncation": False,
                "max_length": self.max_length,
                "return_tensors": "pt",
                "do_resize": False,
            }

            # Only add videos if they exist
            if videos is not None:
                processor_kwargs["videos"] = videos

            # Add video_metadata and video_kwargs for Qwen3
            # Note: video_kwargs may contain keys like 'videos_kwargs' that need special handling
            if is_qwen3:
                if video_metadatas is not None:
                    processor_kwargs["video_metadata"] = video_metadatas
                # Pass video_kwargs - these contain important metadata for video processing
                # The processor will handle 'videos_kwargs' internally if present
                if video_kwargs:
                    processor_kwargs.update(video_kwargs)

            batch_inputs = self.processor(**processor_kwargs)
        elif "SmolVLM" in self.base_model_id:
            batch_inputs = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                padding=True,
                truncation=False,
                max_length=self.max_length,
                return_dict=True,
                return_tensors="pt",
                fps=1,  # this should be same as fps for write_mp4
            )
        else:
            raise ValueError(f"Invalid base model id: {self.base_model_id}")

        return batch_inputs

    def _process_progress_batch(self, progress_samples: list[ProgressSample]) -> dict[str, torch.Tensor]:
        """Process a batch of progress samples."""
        # Collect all messages for batch processing
        all_messages = []

        target_progress_overrides: list[list[float] | None] = []

        for sample in progress_samples:
            # Convert frames to appropriate format using stored shapes
            frames = convert_frames_to_pil_images(sample.trajectory.frames, sample.trajectory.frames_shape)
            target_progress = sample.trajectory.target_progress

            # Optionally shuffle frames (except the first frame) and keep target progress aligned
            if self.shuffle_progress_frames and target_progress is not None and not self.inference:
                if len(target_progress) > 1 and len(target_progress) == len(frames):
                    shuffle_indices = np.random.permutation(range(1, len(frames)))
                    frames = [frames[0]] + [frames[idx] for idx in shuffle_indices]
                    target_progress = [target_progress[0]] + [target_progress[idx] for idx in shuffle_indices]
                else:
                    raise ValueError(
                        "Target progress must be a list with at least 1 element and match the number of frames "
                        f"for shuffling, got {len(target_progress)} entries for {len(frames)} frames"
                    )

            target_progress_overrides.append(target_progress)

            video_field, content_extras = self._prepare_frames_for_conversation(frames, prefix="tmp_progress")

            # Create conversation for progress evaluation
            prompt = f"The task for the robot is '{sample.trajectory.task}'. Given the trajectory video, predict the task progress at each frame, how far along the robot is towards completing the task, a float between 0 and 1, where 0 is the starting state and 1 is when the task is completed. If the robot is not performing the same task, predict 0 progress."

            # Build content list
            content_list = [{"type": "text", "text": prompt}]
            self._add_vision_content_to_list(content_list, video_field, content_extras)

            conversation = [
                {
                    "role": "user",
                    "content": content_list,
                }
            ]

            all_messages.append(conversation)

        batch_inputs = self._process_conversation(all_messages)
        all_have_target_progress = all(tp is not None for tp in target_progress_overrides)
        if all_have_target_progress:
            batch_inputs = self._add_progress_meta(
                batch_inputs,
                progress_samples,
                target_progress_override=target_progress_overrides,
            )
        batch_inputs["resample_attempts"] = [sample.resample_attempts for sample in progress_samples]
        return batch_inputs

    def _add_progress_meta(
        self,
        batch_inputs: dict[str, torch.Tensor],
        progress_samples: list[ProgressSample],
        target_progress_override: list[list[float] | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_inputs["sample_type"] = ["progress"] * len(progress_samples)
        batch_inputs["task"] = [sample.trajectory.task for sample in progress_samples]
        batch_inputs["metadata"] = [sample.trajectory.metadata for sample in progress_samples]

        # Pad target progress tensors to max length in last dimension
        if target_progress_override is not None:
            target_progress_list = target_progress_override
        else:
            target_progress_list = [sample.trajectory.target_progress for sample in progress_samples]

        batch_inputs["target_progress"] = pad_list_to_max(target_progress_list)
        batch_inputs["quality_labels"] = [sample.trajectory.quality_label for sample in progress_samples]

        frames_shape_list = [sample.trajectory.frames_shape for sample in progress_samples]
        batch_inputs["frames_shape"] = torch.tensor(frames_shape_list, dtype=torch.int32)

        max_length = batch_inputs["target_progress"].shape[-1]
        batch_inputs["padding_mask"] = create_padding_mask(batch_inputs["frames_shape"], max_length)

        batch_inputs["data_source"] = [sample.trajectory.data_source for sample in progress_samples]
        batch_inputs["partial_success"] = [sample.trajectory.partial_success for sample in progress_samples]
        batch_inputs["data_gen_strategy"] = [sample.data_gen_strategy for sample in progress_samples]
        target_progress_mask = [
            should_compute_progress(
                sample.trajectory.quality_label,
                sample.data_gen_strategy,
                data_source=sample.trajectory.data_source,
                partial_success=sample.trajectory.partial_success,
            )
            for sample in progress_samples
        ]
        batch_inputs["target_progress_mask"] = torch.tensor(target_progress_mask, dtype=torch.float32)

        # Create predict_last_frame masks for trajectories with partial_success
        predict_last_frame_mask_list = [sample.trajectory.predict_last_frame_mask for sample in progress_samples]

        # Add predict_last_frame_mask (padded to max_length)
        batch_inputs["predict_last_frame_mask"] = pad_list_to_max(predict_last_frame_mask_list)

        success_label_list = [sample.trajectory.success_label for sample in progress_samples]
        batch_inputs["success_labels"] = pad_list_to_max(success_label_list)

        return batch_inputs

    def _process_preference_batch(self, preference_samples: list[PreferenceSample]) -> dict[str, torch.Tensor]:
        """Process a batch of preference samples."""
        # Collect all messages for batch processing
        all_messages = []

        # During inference, keep original order (chosen=A, rejected=B)
        # During training, randomly decide whether chosen trajectory goes first or second
        if self.inference:
            # Keep original order: chosen is always A (preference_label=1.0)
            preference_labels = np.ones(len(preference_samples), dtype=np.int32)
        else:
            preference_labels = np.random.randint(0, 2, len(preference_samples))

        # Build batch of conversations
        for i, sample in enumerate(preference_samples):
            # Convert frames to appropriate format using stored shapes
            chosen_frames = convert_frames_to_pil_images(
                sample.chosen_trajectory.frames, sample.chosen_trajectory.frames_shape
            )
            rejected_frames = convert_frames_to_pil_images(
                sample.rejected_trajectory.frames, sample.rejected_trajectory.frames_shape
            )

            chosen_video_field, content_extras = self._prepare_frames_for_conversation(
                chosen_frames, prefix="tmp_chosen"
            )
            rejected_video_field, _ = self._prepare_frames_for_conversation(rejected_frames, prefix="tmp_rejected")

            prompt = f"Given these two trajectories for the task '{sample.chosen_trajectory.task}', evaluate which one makes more progress towards the task. Return A for the first trajectory and B for the second trajectory."

            if self.prog_pref:
                # We ask the model to predict both of the task progress and preference
                task_prompt = f" Also predict the task progress at each frame of the first trajectory, how far along the robot is towards completing the task, a float between 0 and 1, where 0 is the starting state and 1 is when the task is completed. If the robot is not performing the same task, predict 0 progress."
                prompt += task_prompt

            # Determine which trajectory is A and which is B based on preference label
            if preference_labels[i] == 1.0:
                # Chosen trajectory first: task + video A (chosen) + <|split_token|> + video B (rejected) + <|pref_token|>
                traj_a_field = chosen_video_field
                traj_b_field = rejected_video_field
            else:
                # Chosen trajectory second: task + video A (rejected) + <|split_token|> + video B (chosen) + <|pref_token|>
                traj_a_field = rejected_video_field
                traj_b_field = chosen_video_field

            # Build content list
            content_list = [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "This is Trajectory A. "},
            ]
            self._add_vision_content_to_list(content_list, traj_a_field, content_extras)

            content_list.extend([
                {"type": "text", "text": "<|split_token|>"},
                {"type": "text", "text": "This is Trajectory B. "},
            ])
            self._add_vision_content_to_list(content_list, traj_b_field, content_extras)

            content_list.append({"type": "text", "text": "Now predict the preference between the two trajectories."})
            content_list.append({"type": "text", "text": "<|pref_token|>"})

            conversation = [
                {
                    "role": "user",
                    "content": content_list,
                }
            ]
            all_messages.append(conversation)

        batch_inputs = self._process_conversation(all_messages)
        # Use the dynamically generated preference labels based on trajectory order
        batch_inputs["preference_labels"] = torch.tensor(preference_labels, dtype=torch.float32)
        batch_inputs = self._add_preference_meta(batch_inputs, preference_samples)
        return batch_inputs

    def _add_preference_meta(
        self, batch_inputs: dict[str, torch.Tensor], preference_samples: list[PreferenceSample]
    ) -> dict[str, torch.Tensor]:
        batch_inputs["data_source"] = [sample.chosen_trajectory.data_source for sample in preference_samples]
        batch_inputs["sample_type"] = ["preference"] * len(preference_samples)
        batch_inputs["task"] = [sample.chosen_trajectory.task for sample in preference_samples]
        batch_inputs["data_gen_strategy"] = [sample.data_gen_strategy for sample in preference_samples]

        # Determine which trajectory is A and which is B based on preference_label
        # Trajectory A is chosen if preference_label==1.0, otherwise rejected is A
        trajectory_A_list = [
            sample.chosen_trajectory
            if batch_inputs["preference_labels"][i].item() == 1.0
            else sample.rejected_trajectory
            for i, sample in enumerate(preference_samples)
        ]
        trajectory_B_list = [
            sample.rejected_trajectory
            if batch_inputs["preference_labels"][i].item() == 1.0
            else sample.chosen_trajectory
            for i, sample in enumerate(preference_samples)
        ]

        batch_inputs["trajectory_A_quality_label"] = [traj.quality_label for traj in trajectory_A_list]
        batch_inputs["trajectory_A_data_source"] = [traj.data_source for traj in trajectory_A_list]

        trajectory_A_data_gen_strategy = []
        trajectory_B_data_gen_strategy = []
        for i, sample in enumerate(preference_samples):
            if batch_inputs["preference_labels"][i].item() == 1.0:
                trajectory_A_data_gen_strategy.append("subsample_task")
                trajectory_B_data_gen_strategy.append(sample.data_gen_strategy)
            else:
                trajectory_A_data_gen_strategy.append(sample.data_gen_strategy)
                trajectory_B_data_gen_strategy.append("subsample_task")

        batch_inputs["trajectory_A_data_gen_strategy"] = trajectory_A_data_gen_strategy

        # Add target progress for both trajectories using list comprehensions
        target_progress_A = [traj.target_progress for traj in trajectory_A_list]
        # Check if any of the progresses in target_progress_A is None
        if any(p is None for p in target_progress_A):
            return batch_inputs
        target_progress_B = [traj.target_progress for traj in trajectory_B_list]
        target_progress_A_mask = [
            should_compute_progress(
                traj.quality_label,
                strategy,
                data_source=traj.data_source,
                partial_success=traj.partial_success,
            )
            for traj, strategy in zip(trajectory_A_list, trajectory_A_data_gen_strategy)
        ]
        target_progress_B_mask = [
            should_compute_progress(
                traj.quality_label,
                strategy,
                data_source=traj.data_source,
                partial_success=traj.partial_success,
            )
            for traj, strategy in zip(trajectory_B_list, trajectory_B_data_gen_strategy)
        ]

        frames_shape_A = [traj.frames_shape for traj in trajectory_A_list]
        frames_shape_B = [traj.frames_shape for traj in trajectory_B_list]
        batch_inputs["frames_shape_A"] = torch.tensor(frames_shape_A, dtype=torch.int32)
        batch_inputs["frames_shape_B"] = torch.tensor(frames_shape_B, dtype=torch.int32)

        # Create predict_last_frame masks for trajectories with partial_success
        predict_last_frame_mask_A_list = [traj.predict_last_frame_mask for traj in trajectory_A_list]
        predict_last_frame_mask_B_list = [traj.predict_last_frame_mask for traj in trajectory_B_list]

        batch_inputs["target_progress_A"] = pad_list_to_max(target_progress_A)
        batch_inputs["target_progress_B"] = pad_list_to_max(target_progress_B)
        batch_inputs["target_progress_A_mask"] = torch.tensor(target_progress_A_mask, dtype=torch.float32)
        batch_inputs["target_progress_B_mask"] = torch.tensor(target_progress_B_mask, dtype=torch.float32)
        batch_inputs["predict_last_frame_mask_A"] = pad_list_to_max(predict_last_frame_mask_A_list)
        batch_inputs["predict_last_frame_mask_B"] = pad_list_to_max(predict_last_frame_mask_B_list)

        max_length_A = batch_inputs["target_progress_A"].shape[-1]
        max_length_B = batch_inputs["target_progress_B"].shape[-1]
        batch_inputs["padding_mask_A"] = create_padding_mask(batch_inputs["frames_shape_A"], max_length_A)
        batch_inputs["padding_mask_B"] = create_padding_mask(batch_inputs["frames_shape_B"], max_length_B)

        batch_inputs["chosen_data_gen_strategy"] = [DataGenStrat.FORWARD_PROGRESS.value] * len(preference_samples)
        batch_inputs["rejected_data_gen_strategy"] = [sample.data_gen_strategy for sample in preference_samples]
        batch_inputs["chosen_quality_label"] = [sample.chosen_trajectory.quality_label for sample in preference_samples]

        target_progress_chosen = [sample.chosen_trajectory.target_progress for sample in preference_samples]
        target_progress_rejected = [sample.rejected_trajectory.target_progress for sample in preference_samples]
        target_progress_chosen_mask = [
            should_compute_progress(
                sample.chosen_trajectory.quality_label,
                DataGenStrat.FORWARD_PROGRESS.value,
                data_source=sample.chosen_trajectory.data_source,
                is_chosen=True,
                partial_success=sample.chosen_trajectory.partial_success,
            )
            for sample in preference_samples
        ]
        target_progress_rejected_mask = [
            should_compute_progress(
                sample.rejected_trajectory.quality_label,
                sample.data_gen_strategy,
                data_source=sample.rejected_trajectory.data_source,
                is_chosen=False,
                partial_success=sample.rejected_trajectory.partial_success,
            )
            for sample in preference_samples
        ]

        # Create predict_last_frame masks for chosen/rejected trajectories with partial_success
        predict_last_frame_mask_chosen_list = [
            sample.chosen_trajectory.predict_last_frame_mask for sample in preference_samples
        ]
        predict_last_frame_mask_rejected_list = [
            sample.rejected_trajectory.predict_last_frame_mask for sample in preference_samples
        ]

        # Pad target progress tensors to max length in last dimension
        batch_inputs["target_progress_chosen"] = pad_list_to_max(target_progress_chosen)
        batch_inputs["target_progress_rejected"] = pad_list_to_max(target_progress_rejected)
        batch_inputs["target_progress_chosen_mask"] = torch.tensor(target_progress_chosen_mask, dtype=torch.float32)
        batch_inputs["target_progress_rejected_mask"] = torch.tensor(target_progress_rejected_mask, dtype=torch.float32)
        batch_inputs["predict_last_frame_mask_chosen"] = pad_list_to_max(predict_last_frame_mask_chosen_list)
        batch_inputs["predict_last_frame_mask_rejected"] = pad_list_to_max(predict_last_frame_mask_rejected_list)

        batch_inputs["chosen_frames_shape"] = torch.tensor(
            [sample.chosen_trajectory.frames_shape for sample in preference_samples], dtype=torch.int32
        )
        batch_inputs["rejected_frames_shape"] = torch.tensor(
            [sample.rejected_trajectory.frames_shape for sample in preference_samples], dtype=torch.int32
        )
        batch_inputs["resample_attempts"] = [sample.resample_attempts for sample in preference_samples]

        # Aggregate success labels for trajectory A from trajectories
        success_label_A_list = [traj.success_label for traj in trajectory_A_list]
        batch_inputs["success_labels_A"] = pad_list_to_max(success_label_A_list)

        # Add metadata structure for evaluation
        metadata_list = []
        for sample in preference_samples:
            metadata = {
                "chosen_metadata": {
                    "quality_label": sample.chosen_trajectory.quality_label,
                    "data_gen_strategy": "subsample_task",
                    "id": sample.chosen_trajectory.id,
                    "video_path": sample.chosen_trajectory.metadata.get("video_path")
                    if sample.chosen_trajectory.metadata
                    else None,
                    "partial_success": sample.chosen_trajectory.partial_success,
                },
                "rejected_metadata": {
                    "quality_label": sample.rejected_trajectory.quality_label,
                    "data_gen_strategy": sample.data_gen_strategy,
                    "id": sample.rejected_trajectory.id,
                    "video_path": sample.rejected_trajectory.metadata.get("video_path")
                    if sample.rejected_trajectory.metadata
                    else None,
                    "partial_success": sample.rejected_trajectory.partial_success,
                },
            }
            metadata_list.append(metadata)
        batch_inputs["metadata"] = metadata_list

        return batch_inputs
