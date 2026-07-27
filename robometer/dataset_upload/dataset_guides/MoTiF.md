# MoTiF Dataset Guide

This guide explains how to integrate and use the MoTiF-1K dataset with the Robometer data pipeline using a FrameLoader (no HuggingFace conversion required).

Source: `https://github.com/Minyoung1005/motif#data-structure`

## Overview

- 1K trajectories across 13 task categories; both human and robot (Stretch) motions
- Visual motion representations provided; we support raw video or frame directories
- We use a simple FrameLoader to load frames on-demand for each trajectory

## Directory Structure

As per the MoTiF README, after unzipping `MotIF.zip` under `./data`, `./data/MotIF` contains at least:

```
./data/MotIF/
  annotations/
  human_motion/
  stretch_motion/
```

Our loader first looks for annotations under `annotations/` to pair sources with language text; if absent, it will scan `human_motion/` and `stretch_motion/` for videos or frame directories.

## Loader

- File: `dataset_upload/dataset_loaders/motif_loader.py`
- Exposes `load_motif_dataset(dataset_path: str) -> dict[str, list[dict]]`
- Each trajectory dictionary contains:
  - `id`: unique id
  - `task`: from annotations if available, otherwise "MoTiF"
  - `frames`: `MotifFrameLoader` that lazily reads frames (video file or directory of images)
  - `is_robot`: inferred from path (`stretch`/`robot` -> True, `human` -> False)
  - `quality_label`: "successful"
  - `partial_success`: 1
  - `data_source`: "motif"

## Configuration (configs/data_gen_configs/motif.yaml)

```yaml
# configs/data_gen_configs/motif.yaml

dataset:
  dataset_path: ./datasets/MotIF
  dataset_name: motif

output:
  output_dir: ./robometer_dataset/motif_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false

hub:
  push_to_hub: false
  hub_repo_id: motif_rfm
```

## Usage Example

```bash
uv run python -m dataset_upload.generate_hf_dataset --config dataset_upload/configs/data_gen_configs/motif.yaml
```

This will:
- Find all zip files in the specified dataset path
- For each zip file, extract the task name and load episodes using the humanoid_everyday dataloader
- Extract RGB images from each episode
- Convert frames to web-optimized videos and create a HuggingFace dataset
- Use the zip filename (without extension) as the task description

## Notes

- Annotations: The loader tries to parse any JSON/JSONL files under `annotations/` to find `(source_path, text)` pairs. Supported keys include `video_path|path|image_dir|frames_dir` and `narration|instruction|task|description|caption`.
- Frame directories: If a directory contains images (e.g., `.jpg`, `.png`), it is treated as a sequence of frames.
- Video support: Common video formats are supported via OpenCV (e.g., `.mp4`, `.mov`).
- If you need to use specific MoTiF visual motion representations (e.g., storyboard, optical flow), point `source_path` to those assets and the FrameLoader will load images in order.
