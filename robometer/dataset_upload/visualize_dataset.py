#!/usr/bin/env python3
"""
Dataset Visualization Script for AgiBotWorld datasets.
Loads and displays information about the converted HuggingFace datasets.
Saves both video frame visualizations (PNG) and actual video files (MP4).
"""

import argparse
import os
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from datasets import load_from_disk


def _reencode_video_for_compatibility(video_bytes: bytes) -> bytes:
    """Re-encode video with optimal settings for player compatibility."""
    # Save video bytes to temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_input:
        temp_input.write(video_bytes)
        temp_input_path = temp_input.name

    try:
        # Open video with OpenCV
        cap = cv2.VideoCapture(temp_input_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output file with compatibility-focused encoding
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_output:
            temp_output_path = temp_output.name

        # Try multiple H.264 codec options for maximum compatibility
        codecs_to_try = ["avc1", "h264", "x264", "mp4v"]  # Order of preference
        video_writer = None

        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
                if video_writer.isOpened():
                    break
                video_writer.release()
            except:
                continue

        if not video_writer or not video_writer.isOpened():
            raise Exception("Could not initialize video writer with any codec")

        # Copy all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_writer.write(frame)

        cap.release()
        video_writer.release()

        # Read the re-encoded video
        with open(temp_output_path, "rb") as f:
            compatible_bytes = f.read()

        return compatible_bytes

    finally:
        # Clean up temporary files
        try:
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
        except:
            pass


def extract_video_frames(video_bytes: bytes, num_frames: int = 5) -> list:
    """Extract frames from video bytes for visualization."""
    # Save video bytes to temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_file.write(video_bytes)
        temp_path = temp_file.name

    try:
        # Open video with OpenCV
        cap = cv2.VideoCapture(temp_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print("⚠️ Warning: Could not read frames from video")
            return []

        # Extract evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()
        return frames

    finally:
        # Clean up temporary file
        os.unlink(temp_path)


def visualize_dataset(dataset_path: str, max_samples: int = 3):
    """Visualize a HuggingFace dataset saved to disk."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return

    print(f"📁 Loading dataset from: {dataset_path}")

    try:
        # Load dataset from disk
        dataset = load_from_disk(str(dataset_path))
        print(f"✅ Successfully loaded dataset with {len(dataset)} entries")

        # Display dataset info
        print("\n📊 Dataset Information:")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Features: {list(dataset.features.keys())}")

        # Show sample data
        print(f"\n🔍 Sample Data (showing up to {max_samples} samples):")

        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break

            task_name = sample.get("task", "N/A")
            print(f"\n--- Sample {i + 1} ---")
            print(f"🎯 Task: {task_name}")
            print(f"🤖 Is Robot: {sample.get('is_robot', 'N/A')}")
            print(f"⭐ Optimal: {sample.get('optimal', 'N/A')}")

            # Video info (check both 'video' and 'frames' fields)
            video_data = sample.get("video", b"") or sample.get("frames", [])

            # Handle case where frames is a list containing video bytes
            if isinstance(video_data, list) and len(video_data) > 0:
                video_bytes = video_data[0]  # Take first video from list
            elif isinstance(video_data, bytes):
                video_bytes = video_data
            else:
                video_bytes = b""

            if video_bytes and isinstance(video_bytes, bytes):
                print(f"Video size: {len(video_bytes):,} bytes ({len(video_bytes) / 1024 / 1024:.1f} MB)")

                # Extract and display frames
                frames = extract_video_frames(video_bytes, num_frames=3)
                if frames:
                    print(f"Extracted {len(frames)} frames from video")

                    # Create visualization
                    _fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
                    if len(frames) == 1:
                        axes = [axes]

                    for j, (frame, ax) in enumerate(zip(frames, axes, strict=False)):
                        ax.imshow(frame)
                        ax.set_title(f"Frame {j + 1}")
                        ax.axis("off")

                    plt.suptitle(f"Sample {i + 1}: {task_name}", fontsize=14, wrap=True)
                    plt.tight_layout()

                    # Save visualization
                    output_path = dataset_path.parent / f"sample_{i + 1}_frames.png"
                    plt.savefig(output_path, dpi=150, bbox_inches="tight")
                    print(f"🖼️ Saved frame visualization: {output_path}")
                    plt.close()
                else:
                    print("⚠️ Could not extract frames from video")

                # Save the actual video file with re-encoding for better compatibility
                # Create safe filename from task name
                safe_task_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task_name)
                safe_task_name = safe_task_name.replace(" ", "_")[:50]  # Limit length
                video_output_path = dataset_path.parent / f"sample_{i + 1}_{safe_task_name}.mp4"

                # Re-encode video for maximum compatibility with video players/previews
                try:
                    re_encoded_bytes = _reencode_video_for_compatibility(video_bytes)
                    with open(video_output_path, "wb") as f:
                        f.write(re_encoded_bytes)
                    print(f"🎬 Saved compatible video file: {video_output_path}")
                except Exception as e:
                    # Fallback: save original bytes
                    print(f"⚠️ Re-encoding failed ({e}), saving original video bytes")
                    with open(video_output_path, "wb") as f:
                        f.write(video_bytes)
                    print(f"🎬 Saved video file: {video_output_path}")
            else:
                print("❌ No video data found")

            # Actions info
            actions = sample.get("actions", [])
            if actions:
                actions_array = np.array(actions)
                print(f"Actions shape: {actions_array.shape}")
                print(f"Actions range: [{actions_array.min():.3f}, {actions_array.max():.3f}]")

            # Text embedding info
            text_embedding = sample.get("text_embedding", [])
            if text_embedding:
                embedding_array = np.array(text_embedding)
                print(f"Text embedding shape: {embedding_array.shape}")

        # Dataset statistics
        print("\n📈 Dataset Statistics:")
        tasks = [sample.get("task", "Unknown") for sample in dataset]
        unique_tasks = list(set(tasks))
        print(f"  - Unique tasks: {len(unique_tasks)}")
        for task in unique_tasks:
            count = tasks.count(task)
            print(f"    • {task}: {count} samples")

        # Video size statistics
        video_sizes = []
        for sample in dataset:
            video_data = sample.get("video", b"") or sample.get("frames", [])

            # Handle case where frames is a list containing video bytes
            if isinstance(video_data, list) and len(video_data) > 0 and isinstance(video_data[0], bytes):
                video_sizes.append(len(video_data[0]))
            elif isinstance(video_data, bytes):
                video_sizes.append(len(video_data))
            else:
                video_sizes.append(0)
        if video_sizes:
            total_size = sum(video_sizes)
            avg_size = total_size / len(video_sizes)
            print(f"  - Total video data: {total_size / 1024 / 1024:.1f} MB")
            print(f"  - Average video size: {avg_size / 1024 / 1024:.1f} MB")
            print(
                f"  - Video size range: {min(video_sizes) / 1024 / 1024:.1f} - {max(video_sizes) / 1024 / 1024:.1f} MB"
            )

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")


def main():
    parser = argparse.ArgumentParser(description="Visualize AgiBotWorld datasets")
    parser.add_argument("dataset_path", help="Path to the saved dataset directory")
    parser.add_argument("--max_samples", type=int, default=3, help="Maximum number of samples to visualize")

    args = parser.parse_args()

    print("🎬 AgiBotWorld Dataset Visualizer")
    print("=" * 50)

    visualize_dataset(args.dataset_path, args.max_samples)


if __name__ == "__main__":
    main()
