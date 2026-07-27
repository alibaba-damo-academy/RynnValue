#!/usr/bin/env python3
"""
Standalone script to generate success/fail labels for SOAR dataset using Qwen3-VL.

This script uses STREAMING to process episodes one at a time without loading
the entire dataset into memory. Features:

1. Streams SOAR dataset from TFDS format (memory-efficient)
2. Extracts video frames for each episode
3. Uses Qwen3-VL to classify each episode as success or failure
4. Saves results to a JSON file with periodic checkpoints
5. Supports resuming from checkpoints if interrupted

Requirements:
    pip install transformers torch torchvision pillow tqdm tensorflow-datasets qwen-vl-utils

Usage:
    # Basic usage
    python generate_soar_labels_vlm.py --dataset_path /path/to/soar/rlds --output labels.json

    # With checkpointing every 25 episodes
    python generate_soar_labels_vlm.py --dataset_path /path/to/soar/rlds --output labels.json --checkpoint_interval 25

    # Resume from checkpoint
    python generate_soar_labels_vlm.py --dataset_path /path/to/soar/rlds --output labels.json --resume_from labels_checkpoint.json
"""

# CRITICAL: Configure TensorFlow for CPU-only BEFORE any imports
# This prevents TensorFlow from allocating GPU memory
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings

import argparse
import gc
import json
from pathlib import Path
from typing import Any

# Import TensorFlow FIRST and configure for CPU only
try:
    import tensorflow as tf

    # Force TensorFlow to use CPU only - do this BEFORE other imports
    tf.config.set_visible_devices([], "GPU")
    physical_devices = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow configured for CPU-only mode (found {len(physical_devices)} GPUs but not using them)")

    import tensorflow_datasets as tfds
except ImportError:
    print("Error: tensorflow_datasets is required. Install with: pip install tensorflow-datasets")
    exit(1)

# NOW import PyTorch and other libraries (TensorFlow won't interfere)
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

# Verify PyTorch can see GPUs
if torch.cuda.is_available():
    print(f"PyTorch can access {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch will use CPU (no CUDA available)")


def sample_frames(frames: list[np.ndarray], num_samples: int = 8) -> list[np.ndarray]:
    """Sample evenly spaced frames from a video sequence."""
    if len(frames) <= num_samples:
        return frames

    indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
    return [frames[i] for i in indices]


def frames_to_pil_images(frames: list[np.ndarray]) -> list[Image.Image]:
    """Convert numpy frames to PIL Images."""
    pil_images = []
    for frame in frames:
        if frame.ndim == 2:
            # Grayscale
            frame = np.repeat(frame[:, :, np.newaxis], 3, axis=-1)
        elif frame.shape[-1] == 1:
            # Single channel
            frame = np.repeat(frame, 3, axis=-1)
        elif frame.shape[-1] == 4:
            # RGBA -> RGB
            frame = frame[..., :3]

        # Ensure uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        pil_images.append(Image.fromarray(frame))

    return pil_images


def extract_episode_frames(episode_steps: list[dict], img_key: str = "image_0") -> list[np.ndarray]:
    """Extract frames from episode steps."""
    frames = []
    for step in episode_steps:
        obs = step.get("observation", {}) if isinstance(step, dict) else {}
        if img_key not in obs:
            continue

        frame = obs[img_key]
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[-1] in (1, 3, 4):
                frames.append(frame)

    return frames


def get_task_instruction(episode_steps: list[dict]) -> str | None:
    """Extract language instruction from episode."""
    if not episode_steps:
        return None

    first_step = episode_steps[0]
    if "language_instruction" in first_step:
        val = first_step["language_instruction"]
        return val.decode() if isinstance(val, (bytes, bytearray)) else str(val)

    return None


class Qwen3VLClassifier:
    """Wrapper for Qwen3-VL model to classify robot task success/failure."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "auto", verbose: bool = False):
        """
        Initialize the Qwen3-VL model.

        Args:
            model_name: HuggingFace model name. Options:
                - "Qwen/Qwen3-VL-2B-Instruct" (smaller, faster)
                - "Qwen/Qwen3-VL-8B-Instruct" (default)
                - "Qwen/Qwen3-VL-32B-Instruct" (largest, most accurate)
            device: Device to run on ("cuda", "cpu", or "auto")
            verbose: Print debug information including raw model responses
        """
        print(f"Loading Qwen3-VL model: {model_name}")
        self.verbose = verbose

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device,
            # attn_implementation="flash_attention_2",
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

        # Minimum and maximum pixel values for preprocessing
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1280 * 28 * 28

        print(f"Model loaded on device: {self.model.device}")

    def classify_episode(self, frames: list[Image.Image], task_instruction: str, num_frames: int = 8) -> dict[str, Any]:
        """
        Classify whether a robot task was successful or failed.

        Args:
            frames: List of PIL Images from the episode
            task_instruction: Natural language description of the task
            num_frames: Number of frames to sample from the video

        Returns:
            Dictionary with:
                - prediction: "success" or "failure"
                - confidence: float between 0 and 1
                - reasoning: explanation from the model
        """
        # Sample frames if there are too many
        if len(frames) > num_frames:
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        else:
            sampled_frames = frames

        # Create prompt - simplified for better compliance
        prompt = f"""Analyze this robot manipulation task video.

Task: {task_instruction}

Watch the video and answer:
1. Context: What happened in the video? Explain how the motion change from the first frame to last frame demonstrates the robot solving the task specified by the isntruction or not.
2. Decision: Did the robot succeed or fail?
3. Confidence: How confident are you (0.0 to 1.0)?

Format your response EXACTLY as:
Context: Your explanation of the video and the decision.
Decision: SUCCESS or FAILURE
Confidence: 0.X"""

        # Prepare the message with video frames
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": sampled_frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prepare inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True
        )

        # split the videos and according metadatas
        if video_inputs is not None:
            videos, video_metadatas = zip(*video_inputs)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=videos,
            video_metadata=video_metadatas,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        # Move inputs to device
        inputs = inputs.to(self.model.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Use greedy decoding for consistency
            )

        # Trim input tokens
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Parse response
        return self._parse_response(response)

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse the model's response into structured output."""
        if self.verbose:
            print(f"\n[DEBUG] Raw model response:\n{response}\n")

        lines = response.strip().split("\n")

        prediction = "unknown"
        confidence = 0.5
        reasoning = response  # Default to full response if parsing fails

        for line in lines:
            line_lower = line.lower().strip()

            if line_lower.startswith("decision:"):
                parts = line_lower.split("decision:", 1)
                if len(parts) > 1:
                    decision_text = parts[1].strip()
                    if "success" in decision_text:
                        prediction = "success"
                    elif "failure" in decision_text or "fail" in decision_text:
                        prediction = "failure"

            elif line_lower.startswith("confidence:"):
                parts = line_lower.split("confidence:", 1)
                if len(parts) > 1:
                    conf_text = parts[1].strip()
                    try:
                        # Extract number from text
                        conf_num = "".join(c for c in conf_text if c.isdigit() or c == ".")
                        if conf_num:
                            confidence = float(conf_num)
                            if confidence > 1.0:
                                confidence = confidence / 100.0  # Handle percentage
                    except (ValueError, IndexError):
                        confidence = 0.5

            elif line_lower.startswith("reasoning:"):
                parts = line.split("reasoning:", 1)
                if len(parts) > 1:
                    reasoning = parts[1].strip()

        # Fallback: if no structured response, try to infer from content
        if prediction == "unknown":
            if self.verbose:
                print("[DEBUG] Failed to parse structured response, using fallback inference")
            response_lower = response.lower()
            if "success" in response_lower and "fail" not in response_lower:
                prediction = "success"
                confidence = 0.6
            elif "fail" in response_lower or "failure" in response_lower:
                prediction = "failure"
                confidence = 0.6
            else:
                if self.verbose:
                    print(f"[DEBUG] Could not determine prediction from response")

        if self.verbose:
            print(f"[DEBUG] Parsed: prediction={prediction}, confidence={confidence}\n")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "reasoning": reasoning,
            "raw_response": response,
        }


def get_dataset_splits(dataset_path: str) -> dict[str, Any]:
    """
    Get TFDS dataset splits without loading episodes into memory.

    Returns:
        Dictionary mapping split names to TFDS datasets (iterators)
    """
    root = Path(os.path.expanduser(dataset_path))
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    builder = tfds.builder_from_directory(root)

    dataset_dict = {}
    split_name = "success"
    # Get episode count without loading all episodes
    try:
        ds = builder.as_dataset(split=split_name, shuffle_files=False)
        dataset_dict[split_name] = ds

        # Get episode count without loading all episodes
        try:
            info = builder.info.splits[split_name]
            num_episodes = info.num_examples
            print(f"Found {num_episodes} episodes in '{split_name}' split")
        except:
            print(f"Found '{split_name}' split (episode count unknown)")
    except Exception as e:
        print(f"Warning: Could not load '{split_name}' split: {e}")
        dataset_dict[split_name] = None

    return dataset_dict, builder


def main():
    parser = argparse.ArgumentParser(description="Generate success/fail labels for SOAR dataset using Qwen2-VL")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to SOAR TFDS dataset directory (e.g., /path/to/soar/rlds)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="soar_vlm_labels.json",
        help="Output JSON file path (default: soar_vlm_labels.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct-FP8",
        choices=[
            "Qwen/Qwen3-VL-4B-Instruct-FP8",
            "Qwen/Qwen3-VL-8B-Instruct-FP8",
            "Qwen/Qwen3-VL-32B-Instruct-FP8",
        ],
        help="Qwen3-VL model to use (default: 8B-FP8)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to sample from each video (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: 'cuda', 'cpu', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process per split (default: all)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save checkpoint every N episodes (default: 50, 0 to disable)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint file (JSON with previous results)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug information including raw model responses",
    )

    args = parser.parse_args()

    # Load existing results if resuming
    results = []
    processed_episodes = set()
    start_episode_id = 0

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\nResuming from checkpoint: {args.resume_from}")
        with open(args.resume_from, "r") as f:
            checkpoint_data = json.load(f)
            results = checkpoint_data.get("results", [])
            # Track which episodes we've already processed
            for r in results:
                processed_episodes.add((r["split_name"], r["episode_index"]))
            start_episode_id = len(results)
        print(f"Loaded {len(results)} previously processed episodes")

    # Get dataset splits (streaming, no loading into memory)
    print(f"\nAccessing SOAR dataset from: {args.dataset_path}")
    dataset_dict, builder = get_dataset_splits(args.dataset_path)

    if all(ds is None for ds in dataset_dict.values()):
        print("Error: No splits found in dataset!")
        return

    # Initialize classifier
    classifier = Qwen3VLClassifier(model_name=args.model, device=args.device, verbose=args.verbose)

    # Process episodes in streaming fashion
    episode_id = start_episode_id

    def save_checkpoint(output_path: str, results: list, metadata: dict):
        """Save intermediate checkpoint."""
        checkpoint_data = {
            "metadata": metadata,
            "results": results,
        }
        checkpoint_path = output_path.replace(".json", "_checkpoint.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"  [Checkpoint saved: {len(results)} episodes]")

    for split_name, dataset in dataset_dict.items():
        if dataset is None:
            continue

        print(f"\nProcessing '{split_name}' split...")

        ep_count = 0
        agreements = 0  # Track agreements in this split

        # Create tqdm with postfix for live stats
        pbar = tqdm(dataset, desc=split_name)
        for ep_idx, episode in enumerate(pbar):
            # Check if we've already processed this episode
            if (split_name, ep_idx) in processed_episodes:
                continue

            # Check max episodes limit
            if args.max_episodes and ep_count >= args.max_episodes:
                break

            try:
                # Convert episode to numpy (streaming, only this episode in memory)
                steps_np = list(tfds.as_numpy(episode["steps"]))
            except Exception as e:
                print(f"\nWarning: Failed to load episode {ep_idx} in {split_name}: {e}")
                continue

            # Extract frames
            frames = extract_episode_frames(steps_np)
            if not frames:
                print(f"\nWarning: No frames found for episode {ep_idx} in {split_name}")
                continue

            # Get task instruction
            task = get_task_instruction(steps_np)
            if not task:
                print(f"\nWarning: No task instruction for episode {ep_idx} in {split_name}")
                task = "Complete the manipulation task"

            # Convert to PIL images
            pil_frames = frames_to_pil_images(frames)

            # Classify with VLM
            try:
                classification = classifier.classify_episode(pil_frames, task, num_frames=args.num_frames)

                # Store result
                original_label = "success" if split_name == "success" else "failure"
                predicted_label = classification["prediction"]

                result = {
                    "episode_id": episode_id,
                    "split_name": split_name,
                    "episode_index": ep_idx,
                    "task": task,
                    "num_frames": len(frames),
                    "predicted_label": predicted_label,
                    "confidence": classification["confidence"],
                    "reasoning": classification["reasoning"],
                    "original_label": original_label,
                }

                results.append(result)
                episode_id += 1
                ep_count += 1

                # Track agreement
                if predicted_label == original_label:
                    agreements += 1

                # Update progress bar with live stats
                agreement_pct = (agreements / ep_count * 100) if ep_count > 0 else 0
                avg_conf = sum(r["confidence"] for r in results[-ep_count:]) / ep_count if ep_count > 0 else 0
                pbar.set_postfix({"agree": f"{agreement_pct:.1f}%", "conf": f"{avg_conf:.2f}"})

                # Save checkpoint periodically
                if args.checkpoint_interval > 0 and episode_id % args.checkpoint_interval == 0:
                    save_checkpoint(
                        args.output,
                        results,
                        {
                            "dataset_path": args.dataset_path,
                            "model": args.model,
                            "num_frames": args.num_frames,
                            "total_episodes": len(results),
                            "checkpoint": True,
                        },
                    )

            except Exception as e:
                print(f"\nError processing episode {ep_idx} in {split_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

            # Aggressively free memory after each episode
            del frames
            del pil_frames
            del steps_np
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Close progress bar for this split
        pbar.close()

    # Save results
    print(f"\n\nSaving {len(results)} results to {args.output}")

    output_data = {
        "metadata": {
            "dataset_path": args.dataset_path,
            "model": args.model,
            "num_frames": args.num_frames,
            "total_episodes": len(results),
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results:
        correct = sum(1 for r in results if r["predicted_label"] == r["original_label"])
        total = len(results)
        accuracy = correct / total * 100

        print(f"Total episodes processed: {total}")
        print(f"Agreement with original labels: {correct}/{total} ({accuracy:.1f}%)")

        avg_confidence = sum(r["confidence"] for r in results) / total
        print(f"Average confidence: {avg_confidence:.3f}")

        # Per-split breakdown
        for split_name in ["success", "failure"]:
            split_results = [r for r in results if r["split_name"] == split_name]
            if split_results:
                split_correct = sum(1 for r in split_results if r["predicted_label"] == r["original_label"])
                split_total = len(split_results)
                split_acc = split_correct / split_total * 100
                print(f"\n  {split_name.capitalize()} split:")
                print(f"    Agreement: {split_correct}/{split_total} ({split_acc:.1f}%)")

    print("\n" + "=" * 60)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
