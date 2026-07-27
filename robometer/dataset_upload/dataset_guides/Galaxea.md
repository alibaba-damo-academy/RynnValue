# Galaxea Open-World Dataset Guide

This guide explains how to integrate and use the Galaxea Open-World RLDS dataset with the Robometer training pipeline.

Source: `https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset`

Can download it with `hf download OpenGalaxea/Galaxea-Open-World-Dataset --repo-type dataset --include "*rlds*" --local-dir ./datasets/galaxea`

Also, need to install some extra dependencies:
```bash
uv pip install tensorflow-datasets
uv pip install tensorflow
uv pip install tf-keras
```

## Overview

- 500+ hours of real-world mobile manipulation data in RLDS and LeRobot formats.
- Fine-grained subtask language annotations at step level via `language_instruction`.
- Multiple RLDS builders (e.g., `part1_r1_lite`, `sample_r1_lite`) under a common `rlds/` root.

## Directory Structure

```
<dataset_path>/
  rlds/
    sample_r1_lite/
      1.0.0/
        dataset_info.json
        features.json
        merge_dataset_large_r1_lite-train.tfrecord-00000-of-01024
        ...
    part1_r1_lite/
      1.0.0/
      ...
```

## Language Instruction Schema

As documented, `language_instruction` encodes three parts separated by `@`:
- `high_level` @ `low_level_chinese` @ `low_level_english`

We extract the low-level English (the third part) and use it as the task string for embeddings.

## Configuration (configs/data_gen_configs/galaxea.yaml)

```yaml
# configs/data_gen_configs/galaxea.yaml

dataset:
  dataset_path: ./datasets/galaxea
  dataset_name: galaxea_part1_r1_lite # choose from part1_r1_lite, part2_r1_lite, part3_r1_lite, part4_r1_lite, part5_r1_lite, ...

output:
  output_dir: ./robometer_dataset/galaxea_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 4

hub:
  push_to_hub: true
  hub_repo_id: galaxea_rfm
```

## Usage

```bash
bash dataset_upload/data_scripts/galaxea/gen_all_galaxea.sh
```

This will:
- Iterate the listed RLDS builders under `rlds/`
- For each episode, parse `language_instruction` and extract the low-level English instruction
- Select camera views (`image_camera_head`, `image_camera_wrist_left`, `image_camera_wrist_right`)
- Convert frames to web-optimized videos and create a HuggingFace dataset

## Data Fields

Each trajectory includes:
- `id`: Unique identifier
- `task`: Low-level English instruction (parsed from `language_instruction`)
- `frames`: Relative path to the generated clip video
- `is_robot`: True
- `quality_label`: "successful"
- `partial_success`: N/A (fixed by pipeline)
- `data_source`: `galaxea`

## Troubleshooting

- Builder not found: Ensure the RLDS version directories exist under `rlds/<name>/`.
- Missing instruction: If no `language_instruction` is present or malformed, the episode is skipped.
- Performance: Adjust `num_workers` and batch size inside the loader if needed.
