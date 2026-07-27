# H2R Dataset Guide

This guide explains how to integrate and use the H2R (Human2Robot) dataset with the Robometer training pipeline.

Source: `https://huggingface.co/datasets/dannyXSC/HumanAndRobot`
Paper: `https://arxiv.org/abs/2502.16587`

## Overview

H2R contains paired human and robot videos stored as HDF5 files. Each trajectory provides synchronized human and robot camera streams. The loader reads both streams and standardizes them to RGB `uint8` frame tensors.

## Directory Structure

```
<dataset_path>/
  <task_folder_1>/
    <trajectory_1>.hdf5
    <trajectory_2>.hdf5
  <task_folder_2>/
    <trajectory_1>.hdf5
  ...
```

- The folder name represents the task category. A simple mapping converts folder names to human-readable task strings (see loader).

## HDF5 Format

- The loader expects camera streams under the keys:
  - `/cam_data/human_camera`
  - `/cam_data/robot_camera`
- Each dataset is loaded into memory and converted to RGB if needed.

## Configuration (configs/data_gen_configs/h2r.yaml)

```yaml
# configs/data_gen_configs/h2r.yaml

dataset:
  dataset_path: /path/to/h2r_dataset
  dataset_name: h2r

output:
  output_dir: datasets/h2r_rfm
  max_trajectories: 64   # null for all
  max_frames: 64
  use_video: true
  fps: 30
  shortest_edge_size: 240
  center_crop: false

hub:
  push_to_hub: true
  hub_repo_id: h2r_rfm
```

## Usage

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/h2r.yaml
```

This will:
- Discover `.hdf5` trajectories grouped by task folders
- Load paired human and robot camera frames
- Convert frames to RGB `uint8`
- Produce a HuggingFace dataset

## Data Fields

Each trajectory includes:
- `id`: Unique identifier
- `task`: Human-readable task derived from folder name
- `frames`: A tuple `(human_frames, robot_frames)` when read via the loader
- `is_robot`: `False` for human, `True` for robot
- `quality_label`: "successful"
- `partial_success`: 1
- `data_source`: `h2r`

Note: The converter creates two entries per `.hdf5` file (one for human, one for robot) for training convenience.

## Task Name Mapping

The loader includes a simple mapping from folder names to readable descriptions and falls back to a prettified folder name if no mapping exists. You can extend `FOLDER_TO_TASK_NAME` in `dataset_upload/dataset_loaders/h2r_loader.py`.

## Troubleshooting

- KeyError: Verify that the HDF5 files contain `/cam_data/human_camera` and `/cam_data/robot_camera` datasets.
- Shape errors: Frames must be 4D tensors `(T, H, W, 3)`.
- Performance: Large `.hdf5` files will load into memory; consider limiting `max_trajectories` during testing.
