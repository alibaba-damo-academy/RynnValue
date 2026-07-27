# RACER-augmented RLBench Dataset Guide

This guide explains how to load and convert the RACER-augmented RLBench dataset with the Robometer pipeline.

Sources:
- Dataset card: `https://huggingface.co/datasets/sled-umich/RACER-augmented_rlbench`
- Example JSON: `https://huggingface.co/datasets/sled-umich/RACER-augmented_rlbench/blob/main/samples/close_jar/0/language_description.json`

## Overview

- Train/validation split under `train/` and `val/` directories (sometimes `train/samples/`)
- Each task (e.g., `close_jar`) contains multiple numbered episodes with:
  - `language_description.json` (contains `task_goal` and per-frame `subgoal` entries)
  - Camera folders: `front_rgb/`, `left_shoulder_rgb/`, `right_shoulder_rgb` (or `right_shoudler_rgb`), `wrist_rgb/`

We use `task_goal` as the language instruction for all trajectories.
- Success trajectories: full expert episode for each camera view
- Failure trajectories: for any expert frame with heuristic failures in `augmentation`, construct a failure episode consisting of expert frames up to that frame (inclusive), for each camera view

## Configuration

```yaml
# configs/data_gen_configs/racer_train.yaml

dataset:
  dataset_path: ./datasets/racer
  dataset_name: racer_train

output:
  output_dir: ./robometer_dataset/racer_train_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false

hub:
  push_to_hub: true
  hub_repo_id: racer_train_rfm
```

For validation, use `racer_val.yaml` analogously.

## Loader

- File: `dataset_upload/dataset_loaders/racer_loader.py`
- Function: `load_racer_dataset(dataset_path, dataset_name)`
- Notes:
  - Handles both `train/` and `validation/` (and the `samples/` subfolder if present)
  - Uses `task_goal` from `language_description.json`
  - Builds successes and heuristic failure truncations per camera view

## Usage

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/racer_train.yaml
```

This will:
- Load expert and heuristic failure episodes
- Generate web-optimized videos per camera view
- Create a HuggingFace dataset for the train split (use the val YAML for validation)
