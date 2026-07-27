"""
Per-trajectory camera_pos identification using a local Qwen3-VL model.

For data_sources where camera viewpoint varies across trajectories (e.g.
rfm_new_mit_franka_rfm), this script sends sample frames from each trajectory
to a local VLM to identify the camera position, then writes results into
extracted_meta_with_descriptions.json with per-trajectory robot_description
and camera_description.

Usage:
    python scripts/identify_robot_camera.py

    # Specify which data_sources need per-trajectory identification
    python scripts/identify_robot_camera.py --data-sources rfm_new_mit_franka_rfm

    # Custom model path
    python scripts/identify_robot_camera.py --model-path /path/to/model
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robometer.configs.constants import (
    DATA_SOURCE_ROBOT_DESCRIPTION,
    DATA_SOURCE_CAMERA_DESCRIPTION,
    MIT_FRANKA_CAMERA_DESCRIPTIONS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "/path/to/Qwen3-VL-8B-Instruct"

DATASETS_ROOT = os.environ.get(
    "ROBOMETER_PROCESSED_DATASETS_PATH",
    "/path/to/processed_datasets",
)
META_PATH = Path(__file__).resolve().parent.parent / "extracted_meta.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "extracted_meta_with_descriptions.json"

DEFAULT_PER_TRAJ_SOURCES = ["rfm_new_mit_franka_rfm"]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LocalVLM:
    def __init__(self, model_path: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading model from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print("Model loaded.")

    @torch.inference_mode()
    def query(self, images: list, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_sample_frames(video_path: str, num_frames: int = 3) -> list:
    """Load evenly-spaced frames from an npz file, return as PIL Images."""
    rel = video_path.replace("./processed_datasets/", "")
    npz_path = os.path.join(DATASETS_ROOT, rel)
    if not os.path.exists(npz_path):
        return []
    data = np.load(npz_path, allow_pickle=True)
    frames = data["frames"]
    n = len(frames)
    if n == 0:
        return []
    indices = np.linspace(0, n - 1, min(num_frames, n), dtype=int)
    return [Image.fromarray(frames[i]) for i in indices]


def build_camera_prompt() -> str:
    return """Look at these frames from a robot manipulation video. Identify the camera viewpoint.

This is a Franka robot. The camera is one of these two types:
  - "wrist": wrist-mounted camera attached to the robot's end-effector. The view moves with the arm, shows close-up of gripper and objects being manipulated.
  - "main": fixed third-person camera showing the workspace from a side/front angle. The view does not move.

Respond ONLY with "wrist" or "main", nothing else."""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-sources", nargs="*", default=None,
        help="Data sources needing per-trajectory identification. "
             "Default: rfm_new_mit_franka_rfm",
    )
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Path to local Qwen3-VL model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: extracted_meta_with_descriptions.json)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of trajectories to process at a time",
    )
    args = parser.parse_args()

    per_traj_sources = set(args.data_sources or DEFAULT_PER_TRAJ_SOURCES)
    output_path = Path(args.output) if args.output else OUTPUT_PATH

    if not META_PATH.exists():
        print(f"ERROR: {META_PATH} not found")
        sys.exit(1)

    with open(META_PATH) as f:
        meta = json.load(f)

    # Count how many need VLM
    vlm_count = sum(1 for item in meta if item["data_source"] in per_traj_sources)
    print(f"Loaded {len(meta)} entries from {META_PATH}")
    print(f"Per-trajectory identification for: {per_traj_sources} ({vlm_count} trajectories)")
    print(f"Model: {args.model_path}")
    print("=" * 60)

    # Only load model if needed
    vlm = LocalVLM(args.model_path) if vlm_count > 0 else None

    prompt = build_camera_prompt()
    results = []
    errors = 0

    for i, item in enumerate(meta):
        ds = item["data_source"]
        entry = dict(item)

        if ds in per_traj_sources:
            frames = load_sample_frames(item["video_path"], num_frames=3)
            if not frames:
                print(f"  [{i+1}/{len(meta)}] SKIP (no frames): {item['id']}")
                entry["robot_description"] = robot_description("franka")
                entry["camera_description"] = None
                results.append(entry)
                continue

            try:
                response = vlm.query(frames, prompt).strip().strip('"').strip("'").lower()
                if "wrist" in response:
                    cam_key = "wrist"
                else:
                    cam_key = "main"

                entry["robot_description"] = DATA_SOURCE_ROBOT_DESCRIPTION.get(ds)
                entry["camera_description"] = MIT_FRANKA_CAMERA_DESCRIPTIONS[cam_key]
                print(f"  [{i+1}/{len(meta)}] {item['id'][:8]}... -> {cam_key} ({entry['camera_description']})")
            except Exception as e:
                print(f"  [{i+1}/{len(meta)}] ERROR {item['id'][:8]}...: {e}")
                entry["robot_description"] = DATA_SOURCE_ROBOT_DESCRIPTION.get(ds)
                entry["camera_description"] = MIT_FRANKA_CAMERA_DESCRIPTIONS["main"]
                errors += 1
        else:
            # Use static mapping
            entry["robot_description"] = DATA_SOURCE_ROBOT_DESCRIPTION.get(ds)
            entry["camera_description"] = DATA_SOURCE_CAMERA_DESCRIPTION.get(ds)

        results.append(entry)

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done. {len(results)} entries written to {output_path}")
    if errors:
        print(f"  ({errors} errors, fell back to default mapping)")

    # Summary
    from collections import Counter
    camera_counts = Counter(r.get("camera_description") for r in results)
    print(f"\nCamera distribution:")
    for desc, count in camera_counts.most_common():
        print(f"  {desc}: {count}")


if __name__ == "__main__":
    main()
