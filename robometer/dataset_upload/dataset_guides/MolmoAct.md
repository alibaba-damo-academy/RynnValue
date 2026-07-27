# MolmoAct Dataset Guide

This guide explains how to integrate and use the MolmoAct LeRobot dataset with the Robometer training pipeline.

Source: `https://huggingface.co/datasets/allenai/MolmoAct-Dataset`

## Overview

- LeRobot/Parquet dataset with per-frame views and state/action fields.
- We stream rows and group by `episode_index` to form trajectories.
- Views used: `first_view`, `second_view`, `wrist_image`.

## Directory Structure

```
<dataset_path>/
  *.parquet
  subfolder_a/*.parquet
  subfolder_b/*.parquet
```

## Configuration (configs/data_gen_configs/molmoact.yaml)

```yaml
# configs/data_gen_configs/molmoact.yaml

dataset:
  dataset_path: ./datasets/molmoact
  dataset_name: molmoact_dataset_tabletop

output:
  output_dir: ./robometer_dataset/molmoact_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 1

hub:
  push_to_hub: true
  hub_repo_id: molmoact_rfm
```

## Usage

```bash
bash dataset_upload/data_scripts/molmoact/gen_all_molmoact.sh
```

This will:
- Stream parquet rows
- Group frames by `episode_index`
- Write per-view videos and build a HuggingFace dataset

## Data Fields

Each trajectory includes:
- `id`: Unique identifier
- `task`: Uses `task_index` if present, else default label
- `frames`: Relative path to the generated clip video
- `is_robot`: True
- `quality_label`: "successful"
- `partial_success`: N/A (fixed by pipeline)
- `data_source`: `molmoact`

## Notes

- Images are decoded from datasets Image cells into RGB uint8 arrays.
- If two consecutive episodes share identical views, they still become separate trajectories.
- Adjust `max_frames`/`fps` for performance and disk usage.
