# RoboReward Dataset Guide

This guide explains how to load and convert the RoboReward dataset with the Robometer pipeline.

Sources:
- Paper: [RoboReward: General-Purpose Vision-Language Reward Models for Robotics](https://arxiv.org/abs/2601.00675)
- Dataset: [https://huggingface.co/datasets/teetone/RoboReward](https://huggingface.co/datasets/teetone/RoboReward)

## Overview

RoboReward is a dataset for training and evaluating general-purpose vision-language reward models for robotics. Each example pairs a task instruction with a real-robot rollout video and a discrete end-of-episode progress reward score.

### Dataset Composition

- **Total**: 54,135 examples
- **Train**: 45,072 trajectories
- **Validation**: 6,232 trajectories  
- **Test** (RoboRewardBench): 2,831 trajectories (human-verified)

Built from large-scale real-robot corpora including Open X-Embodiment (OXE) and RoboArena.

### Directory Structure

```
dataset_path/
  train/
    metadata.jsonl
    [subdirectories with MP4 videos]
  val/
    metadata.jsonl
    [subdirectories with MP4 videos]
  test/
    metadata.jsonl
    [subdirectories with MP4 videos]
```

### Reward Scale

Each trajectory has a discrete reward score (1-5) which is converted to `partial_success` in [0.0, 1.0]:

| Reward | Meaning | partial_success | quality_label |
|--------|---------|-----------------|---------------|
| 1 | No success | 0.0 | failure |
| 2 | Minimal progress | 0.25 | failure |
| 3 | Partial completion | 0.5 | failure |
| 4 | Near completion | 0.75 | failure |
| 5 | Perfect completion | 1.0 | successful |

## Configuration

```yaml
# configs/data_gen_configs/roboreward.yaml

dataset:
  dataset_path: ./datasets/RoboReward
  dataset_name: roboreward_train  # Can be overridden with --dataset.dataset_name

output:
  output_dir: ./robometer_dataset/roboreward_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 4

hub:
  push_to_hub: true
  hub_repo_id: roboreward_rfm
```

Use command-line overrides to specify different splits (train/val/test).

## Loader

- File: `dataset_upload/dataset_loaders/roboreward_loader.py`
- Function: `load_roboreward_dataset(dataset_path, dataset_name)`
- Notes:
  - Reads `metadata.jsonl` from the specified split (train/val/test)
  - Loads existing MP4 videos (no re-encoding needed)
  - Converts reward scores to partial_success values
  - All trajectories are robot demonstrations (`is_robot=True`)

## Usage

```bash
# Train split
uv run python -m dataset_upload.generate_hf_dataset \
  --config_path=dataset_upload/configs/data_gen_configs/roboreward.yaml \
  --dataset.dataset_name roboreward_train

# Validation split
uv run python -m dataset_upload.generate_hf_dataset \
  --config_path=dataset_upload/configs/data_gen_configs/roboreward.yaml \
  --dataset.dataset_name roboreward_val

# Test split (RoboRewardBench - human-verified)
uv run python -m dataset_upload.generate_hf_dataset \
  --config_path=dataset_upload/configs/data_gen_configs/roboreward.yaml \
  --dataset.dataset_name roboreward_test
```

This will:
- Load the specified split (train/val/test)
- Process existing MP4 videos
- Convert reward scores to partial_success values
- Create a HuggingFace dataset with proper quality labels

## Data Fields

Each trajectory contains:
- `task`: Natural-language instruction for the rollout
- `frames`: Video showing robot execution
- `partial_success`: Continuous score in [0.0, 1.0] derived from reward
- `quality_label`: "successful" (reward=5) or "failure" (reward<5)
- `is_robot`: Always `True` (all robot demonstrations)
- `data_source`: "roboreward"

## Citation

```bibtex
@misc{lee2026roboreward,
      title={RoboReward: General-Purpose Vision-Language Reward Models for Robotics}, 
      author={Tony Lee and Andrew Wagenmaker and Karl Pertsch and Percy Liang and Sergey Levine and Chelsea Finn},
      year={2026},
      eprint={2601.00675},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.00675}, 
}
```
