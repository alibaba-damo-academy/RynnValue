# MIT-Franka-P-Rank Dataset Guide

This guide explains how to integrate and use the MIT-Franka-P-Rank dataset with the Robometer pipeline.

## Overview

- MIT-Franka-Prank is a robotics dataset with quality-labeled trajectories for manipulation tasks
- The dataset contains pre-recorded MP4 videos with metadata describing task instructions and quality labels
- Quality labels include: **successful**, **suboptimal**, and **failure**

## Dataset Structure

The dataset is organized as follows:

```
<dataset_path>/
  20251210/
    episode_000_foldtowel_suboptimal.mp4
    episode_000_movebanana_success.mp4
    episode_000_movepebble_suboptimal.mp4
    ...
    foldtowel_metadata.json
    pickandplace_metadata.json
```

### Tasks Included

1. **foldtowel**: Fold the towel in half
2. **movebanana**: Pick up the banana and place it on the blue plate
3. **movepebble**: Move some pebbles from the blue bowl to the green plate using the scoop

### Metadata Format

Each metadata JSON file contains an array of episodes:

```json
[
  {
    "episode_idx": 0,
    "task_name": "foldtowel",
    "filename": "episode_000_foldtowel_suboptimal.mp4",
    "run_dir": "20251210_192953",
    "instruction": "fold the towel in half",
    "success": "suboptimal",
    "trajectory_length": 1351
  },
  ...
]
```

## Configuration

Configuration file: `dataset_upload/configs/data_gen_configs/mit_robot_prank.yaml`

```yaml
dataset:
  dataset_path: ~/projects/robometer/datasets/20251210-mit-robot-prank
  dataset_name: mit_franka_p-rank_rfm

output:
  output_dir: ./robometer_dataset/mit_franka_p-rank_rfm
  max_trajectories: -1  # -1 for all trajectories
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 4

hub:
  push_to_hub: true
  hub_repo_id: mit_franka_p-rank_rfm
```

## Usage

### Convert Dataset to HuggingFace Format

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/mit_franka_prank.yaml
```

This will:
- Read all metadata JSON files from the dataset directory
- Load the corresponding MP4 videos
- Process and resample videos to the specified frame count and FPS
- Generate language embeddings for task instructions
- Create a HuggingFace dataset with proper quality labels
- Optionally push to HuggingFace Hub

### Quality Label Mapping

The loader automatically normalizes quality labels:
- `"success"` → `"successful"`
- `"fail"` → `"failure"`
- `"suboptimal"` → `"suboptimal"` (unchanged)

## Output Format

The generated dataset will have the following schema:

- `id`: Unique trajectory identifier
- `task`: Task instruction text
- `lang_vector`: Language embedding of the task
- `data_source`: Dataset name
- `frames`: Path to video file (or sequence of images)
- `is_robot`: Boolean (True for this dataset)
- `quality_label`: One of "successful", "suboptimal", "failure"
- `preference_group_id`: None for this dataset
- `preference_rank`: None for this dataset

## Notes

- Videos are already in MP4 format, so the loader reads them directly using OpenCV
- The loader supports parallel processing with configurable worker count
- Language embeddings are cached to avoid redundant computations
- Output videos maintain the same content but are resampled to the specified frame count and FPS

## Troubleshooting

### Video File Not Found
- Ensure the dataset path points to the parent directory containing the date-stamped subdirectory
- Check that video files exist and match the filenames in metadata JSON files

### Missing Metadata Files
- The loader looks for files ending with `_metadata.json` in the video directory
- Ensure at least one metadata file exists (e.g., `foldtowel_metadata.json`, `pickandplace_metadata.json`)

### OpenCV Issues
- If you encounter video codec issues, ensure OpenCV is properly installed with video support
- Try: `pip install opencv-python-headless` or `pip install opencv-python`

## Example Output

After processing, you'll have a directory structure like:

```
robometer_dataset/mit_franka_p-rank_rfm/
  mit_franka_p-rank_rfm/
    shard_0000/
      episode_000000/
        foldtowel.mp4
      episode_000001/
        foldtowel.mp4
      ...
```

And a HuggingFace Dataset with all trajectory metadata.

