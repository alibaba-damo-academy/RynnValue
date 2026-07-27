# AutoEval Dataset Guide

This guide explains how to load and convert the AutoEval pickled evaluation episodes into the Robometer pipeline.

Source: `https://huggingface.co/datasets/zhouzypaul/auto_eval`

## Prerequisites

1) Install the logger library first (may need to make a minor hardcoded moviepy import tweak in one file to make things work):

```bash
git clone https://github.com/zhouzypaul/robot_eval_logger
cd robot_eval_logger
uv pip install -e .
```

2) Download the dataset locally so that it contains an `eval_data/` directory with subfolders per group, each containing pickled episodes.

## Directory Structure

```
<dataset_path>/
  eval_data/
    00001/
      episode_00001_success.pkl
      episode_00001_fail.pkl
    00002/
      ...
```

We expect per-episode pickle files. Success/failure may be in the filename suffix or inside the pickle (`success` flag). Only paired (success and fail) are kept.

## Loader

- File: `dataset_upload/dataset_loaders/autoeval_loader.py`
- Function: `load_autoeval_dataset(dataset_path: str) -> dict[str, list[dict]]`
- For each episode group, decodes frames from `obs['image_primary']` and records `success`.
- Prints totals for successes, failures, and kept pairs. Only paired entries are returned.

## Configuration (configs/data_gen_configs/autoeval.yaml)

```yaml
# configs/data_gen_configs/autoeval.yaml

dataset:
  dataset_path: ./datasets/autoeval
  dataset_name: autoeval

output:
  output_dir: ./robometer_dataset/autoeval_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 2

hub:
  push_to_hub: true
  hub_repo_id: autoeval_rfm
```

## Usage

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/autoeval.yaml
```

This will:
- Load pickles from `eval_data/`
- Extract `image_primary` frames per step
- Keep only paired success/failure episodes
- Create a HF dataset with relative video paths
