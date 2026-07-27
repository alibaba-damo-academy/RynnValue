#!/usr/bin/env python3
"""
RBM and ReWiND model class for baseline evaluation.

This class provides a unified interface for loading RBM/ReWiND models from checkpoints
and computing progress and preference predictions.
"""

import time
import torch
import numpy as np
from typing import List, Dict, Any, Union

from robometer.utils.setup_utils import setup_batch_collator
from robometer.utils.save import load_model_from_hf
from robometer.data.dataset_types import ProgressSample, PreferenceSample
from robometer.data.datasets.helpers import create_trajectory_from_dict
from robometer.evals.eval_server import forward_model
from robometer.utils.logger import get_logger, setup_loguru_logging
from robometer.models.utils import convert_bins_to_continuous

logger = get_logger()

setup_loguru_logging("TRACE")


class RBMModel:
    """RBM/ReWiND model for baseline evaluation with unified compute methods."""

    def __init__(self, checkpoint_path: str):
        """Initialize the RBM/ReWiND model wrapper.

        Args:
            checkpoint_path: Path to model checkpoint (HuggingFace repo ID or local path)
                           The config.yaml will be loaded from the checkpoint automatically
        """
        self.checkpoint_path = checkpoint_path

        # Automatically determine device (cuda:0 if available, else cpu)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load model, config, processor, and tokenizer using the helper function
        # This handles loading config.yaml from checkpoint and setting up everything
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        exp_config, tokenizer, processor, model = load_model_from_hf(
            model_path=checkpoint_path,
            device=device,
        )

        # Store loaded components
        self.exp_config = exp_config
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

        # Optimize model for inference
        self.model.eval()  # Set to evaluation mode (disables dropout, batch norm updates, etc.)

        # Enable cuDNN benchmarking for faster inference (if using CUDA)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Determine if this is ReWiND or RBM
        self.is_rewind = "rewind" in exp_config.model.base_model_id.lower()
        logger.info(f"Model type: {'ReWiND' if self.is_rewind else 'RBM'}")

        # Create batch collator using the loaded config
        self.batch_collator = setup_batch_collator(
            processor=processor,
            tokenizer=tokenizer,
            cfg=exp_config,
            is_eval=True,
        )

        logger.info(f"Model loaded successfully on device: {self.device}")

    def compute_progress(
        self, frames_array: Union[List, torch.Tensor, np.ndarray], task_description: str = ""
    ) -> List[float]:
        """Compute progress prediction for a trajectory.

        Args:
            frames_array: Array of frames (can be list, tensor, or numpy array)
            task_description: Task description text

        Returns:
            List of progress values (0-1) for each frame
        """
        # Create a ProgressSample from the inputs
        traj_dict = {
            "frames": frames_array,
            "task": task_description,
            "num_frames": len(frames_array) if hasattr(frames_array, "__len__") else frames_array.shape[0],
        }

        trajectory = create_trajectory_from_dict(traj_dict)
        sample = ProgressSample(trajectory=trajectory)

        # Collate into batch
        batch_inputs = self.batch_collator([sample])

        # Extract progress_inputs from batch_inputs (batch_collator returns nested structure)
        progress_inputs = batch_inputs["progress_inputs"]

        # Move to device
        progress_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in progress_inputs.items()
        }

        # Forward pass with inference mode for additional optimization
        with torch.inference_mode():  # Faster than torch.no_grad() for inference-only code
            model_output, _ = forward_model(self.model, progress_inputs, sample_type="progress")

        # Extract progress logits
        progress_logits = model_output.progress_logits
        if progress_logits is None:
            raise ValueError("No progress logits returned from model")

        # Handle different output formats
        if isinstance(progress_logits, dict):
            # RBM format: {"A": tensor, "B": None}
            progress_tensor = progress_logits.get("A")
        else:
            # Direct tensor
            progress_tensor = progress_logits

        if progress_tensor is None:
            raise ValueError("No progress logits in 'A' key")

        # Convert to list of floats
        progress_values = progress_tensor.squeeze().cpu().tolist()

        # Ensure we have the right length
        if isinstance(progress_values, float):
            progress_values = [progress_values]
        
        logger.info(f"Robometer: Progress score: {progress_values}")

        return progress_values

    def compute_preference(
        self, chosen_images: List, rejected_images: List, task_description: str = ""
    ) -> Dict[str, Any]:
        """Compute preference prediction between two trajectories.

        Args:
            chosen_images: List of images/frames for the chosen trajectory
            rejected_images: List of images/frames for the rejected trajectory
            task_description: Task description text

        Returns:
            Dictionary containing:
            - prediction_prob: Probability that chosen is preferred (0.0 to 1.0)
            - is_correct: True if prediction matches ground truth (always True for chosen)
            - preference_pred: Binary prediction (1.0 if chosen is preferred, 0.0 otherwise)
            - Other metadata
        """
        start_time = time.time()

        # Create trajectories
        chosen_traj = create_trajectory_from_dict({
            "frames": chosen_images,
            "task": task_description,
            "num_frames": len(chosen_images),
        })
        rejected_traj = create_trajectory_from_dict({
            "frames": rejected_images,
            "task": task_description,
            "num_frames": len(rejected_images),
        })

        # Create PreferenceSample
        sample = PreferenceSample(
            chosen_trajectory=chosen_traj,
            rejected_trajectory=rejected_traj,
        )

        # Collate into batch
        batch_inputs = self.batch_collator([sample])

        # Extract preference_inputs from batch_inputs (batch_collator returns nested structure)
        preference_inputs = batch_inputs["preference_inputs"]

        # Move to device
        preference_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in preference_inputs.items()
        }

        # Forward pass with inference mode for additional optimization
        with torch.inference_mode():  # Faster than torch.no_grad() for inference-only code
            model_output, _ = forward_model(self.model, preference_inputs, sample_type="preference")

        # Extract preference logits
        pref_logits = model_output.pref_logits
        if pref_logits is None:
            raise ValueError("No preference logits returned from model")

        # Convert logits to probability
        pref_probs = torch.sigmoid(pref_logits)
        prediction_prob = pref_probs.item()
        preference_pred = 1.0 if prediction_prob > 0.5 else 0.0

        processing_time = time.time() - start_time

        # Build result dict (matching RLVLMF format)
        result = {
            "is_correct": True,  # Chosen is always preferred by construction
            "prediction_prob": float(prediction_prob),
            "preference_pred": float(preference_pred),
            "preference_logits": float(pref_logits.item()) if pref_logits is not None else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task_description,
            "num_chosen_frames": len(chosen_images),
            "num_rejected_frames": len(rejected_images),
            "processing_time_seconds": processing_time,
        }

        return result

    def compute_batched_progress(self, samples: List[ProgressSample]) -> List[List[float]]:
        """Compute progress predictions for a batch of trajectories.

        Args:
            samples: List of ProgressSample objects

        Returns:
            List of progress value lists (each inner list contains 0-1 values for each frame)
        """
        if not samples:
            return []

        # Collate into batch
        batch_inputs = self.batch_collator(samples)

        # Extract progress_inputs from batch_inputs (batch_collator returns nested structure)
        progress_inputs = batch_inputs["progress_inputs"]

        # Move to device
        progress_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in progress_inputs.items()
        }

        # Forward pass with inference mode for additional optimization
        with torch.inference_mode():  # Faster than torch.no_grad() for inference-only code
            time_start = time.time()
            model_output, _ = forward_model(self.model, progress_inputs, sample_type="progress")
            time_end = time.time()
            print(f"Time taken for forward pass: {time_end - time_start} seconds")

        # Extract progress logits
        progress_logits = model_output.progress_logits
        if progress_logits is None:
            raise ValueError("No progress logits returned from model")

        # Handle different output formats
        if isinstance(progress_logits, dict):
            # RBM format: {"A": tensor, "B": None}
            progress_tensor = progress_logits.get("A")
        else:
            # Direct tensor
            progress_tensor = progress_logits

        if progress_tensor is None:
            raise ValueError("No progress logits in 'A' key")

        # progress_tensor shape: [batch_size, num_frames] or [batch_size, 1]
        batch_size = progress_tensor.shape[0]
        results = []

        for i in range(batch_size):
            # Extract progress values for this sample
            if progress_tensor.ndim == 2:
                if progress_tensor.shape[1] == 1:
                    # Single value per sample
                    progress_values = [float(progress_tensor[i, 0].item())]
                else:
                    # Multiple values per sample
                    progress_values = progress_tensor[i].cpu().tolist()
            elif progress_tensor.ndim == 3:
                # Multiple values per sample, discrete multiple bins, convert to continuous
                progress_values = convert_bins_to_continuous(progress_tensor[i]).cpu().tolist()
            else:
                # Unexpected shape
                raise ValueError(f"Unexpected progress_tensor shape: {progress_tensor.shape}")

            # Ensure we have the right length
            if isinstance(progress_values, float):
                progress_values = [progress_values]

            results.append(progress_values)

        return results

    def compute_batched_preference(self, samples: List[PreferenceSample]) -> List[Dict[str, Any]]:
        """Compute preference predictions for a batch of trajectory pairs.

        Args:
            samples: List of PreferenceSample objects

        Returns:
            List of result dictionaries, each containing:
            - prediction_prob: Probability that chosen is preferred (0.0 to 1.0)
            - is_correct: True if prediction matches ground truth (always True for chosen)
            - preference_pred: Binary prediction (1.0 if chosen is preferred, 0.0 otherwise)
            - Other metadata
        """
        if not samples:
            return []

        start_time = time.time()

        # Collate into batch
        batch_inputs = self.batch_collator(samples)

        # Extract preference_inputs from batch_inputs (batch_collator returns nested structure)
        preference_inputs = batch_inputs["preference_inputs"]

        # Move to device
        preference_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in preference_inputs.items()
        }

        # Forward pass with inference mode for additional optimization
        with torch.inference_mode():  # Faster than torch.no_grad() for inference-only code
            model_output, _ = forward_model(self.model, preference_inputs, sample_type="preference")

        # Extract preference logits
        pref_logits = model_output.pref_logits
        if pref_logits is None:
            raise ValueError("No preference logits returned from model")

        # pref_logits shape: [batch_size]
        batch_size = pref_logits.shape[0]
        processing_time = time.time() - start_time

        # Convert logits to probabilities
        pref_probs = torch.sigmoid(pref_logits)
        binary_preds = (pref_probs > 0.5).float()

        results = []
        for i in range(batch_size):
            sample = samples[i]
            prediction_prob = float(pref_probs[i].item())
            preference_pred = float(binary_preds[i].item())
            pref_logit = float(pref_logits[i].item())

            result = {
                "is_correct": True,  # Chosen is always preferred by construction
                "prediction_prob": prediction_prob,
                "preference_pred": preference_pred,
                "preference_logits": pref_logit,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "task": sample.chosen_trajectory.task if sample.chosen_trajectory.task else "",
                "num_chosen_frames": len(sample.chosen_trajectory.frames)
                if sample.chosen_trajectory.frames is not None
                else 0,
                "num_rejected_frames": len(sample.rejected_trajectory.frames)
                if sample.rejected_trajectory.frames is not None
                else 0,
                "processing_time_seconds": processing_time / batch_size,  # Average time per sample
            }
            results.append(result)

        return results
