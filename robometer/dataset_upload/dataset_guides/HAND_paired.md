# HAND_paired_data Dataset Guide

This guide explains how to load and convert a subset of the HAND dataset with the Robometer pipeline from [HAND](https://liralab.usc.edu/handretrieval/).

## Overview

The HAND dataset contains paired robot and human demonstrations for the same tasks. Each task has both robot executions and human hand demonstrations.

### Directory Structure

```
dataset_path/
  blend_carrot/                    # Robot demonstrations
    traj0/
      external_imgs/
        im_0.jpg, im_1.jpg, ...
      over_shoulder_imgs/
        im_0.jpg, im_1.jpg, ...
    traj1/
      ...
  blend_carrot_hand/               # Human demonstrations
    traj0/
      external_imgs/
        im_0.jpg, im_1.jpg, ...
      over_shoulder_imgs/
        im_0.jpg, im_1.jpg, ...
    traj1/
      ...
  close_microwave/
    ...
  close_microwave_hand/
    ...
```

### Key Features

- **Paired Data**: Robot tasks (e.g., `blend_carrot`) and human tasks (e.g., `blend_carrot_hand`) share the same task instruction
- **Task Names**: Folder names are converted to instructions (e.g., `blend_carrot` → "blend carrot")
- **Camera Views**: Each trajectory can have up to 2 camera views:
  - `external_imgs/`: External camera view
  - `over_shoulder_imgs/`: Over-shoulder camera view
- **Multiple Trajectories**: Each task contains multiple demonstrations (`traj0`, `traj1`, etc.)

## Configuration

```yaml
# configs/data_gen_configs/hand_paired.yaml

dataset:
  dataset_path: ./datasets/HAND_paired_data
  dataset_name: hand_paired

output:
  output_dir: ./robometer_dataset/hand_paired_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 4

hub:
  push_to_hub: true
  hub_repo_id: hand_paired_rfm
```

## Loader

- File: `dataset_upload/dataset_loaders/hand_paired_loader.py`
- Function: `load_hand_paired_dataset(dataset_path)`
- Notes:
  - Automatically identifies robot vs. human demonstrations based on `_hand` suffix
  - Processes both camera views independently
  - Sorts images by numeric suffix (`im_0.jpg`, `im_1.jpg`, etc.)
  - All demonstrations are labeled as "successful"

## Usage

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/hand_paired.yaml --dataset.dataset_name hand_paired_robot
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/hand_paired.yaml --dataset.dataset_name hand_paired_human
```

This will:
- Load both robot and human demonstrations
- Process all camera views
- Generate web-optimized videos
- Create a HuggingFace dataset with paired robot/human data

## Data Fields

Each trajectory contains:
- `task`: Human-readable task instruction (e.g., "blend carrot")
- `is_robot`: Boolean indicating if this is a robot (True) or human (False) demonstration
- `frames`: Video or image sequence
- `quality_label`: "successful" (all demonstrations assumed successful)
- `data_source`: "hand_paired"

