# USC Koch Human-Robot Paired Dataset Guide

This guide explains how to integrate and use the USC Koch Human-Robot Paired dataset with the Robometer pipeline.

## Quick Start

```bash
# 1. Download the dataset
bash dataset_upload/data_scripts/usc_koch_human_robot_paired/download_datasets.sh

# 2. Convert human demonstrations
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/usc_koch_human_robot_paired.yaml \
    --dataset.dataset_name=usc_koch_human_robot_paired_human

# 3. Convert robot demonstrations
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/usc_koch_human_robot_paired.yaml \
    --dataset.dataset_name=usc_koch_human_robot_paired_robot
```

## Overview

- USC Koch Human-Robot Paired dataset contains **paired human and robot demonstrations** for manipulation tasks
- Human demonstrations are MP4 videos with JSON metadata
- Robot demonstrations are in **LeRobot format** (parquet files with pre-rendered videos)
- Robot demonstrations use the **top camera view**
- All demonstrations are labeled as **successful** (quality label)
- **Single config file** - specify `human` or `robot` as the dataset name

## Dataset Structure

```
<dataset_path>/
  human/
    recordings/
      human_success_Move_the_orange_cup_from_the_left_to_the_right_20251222_174429.mp4
      human_success_Move_the_orange_cup_from_the_left_to_the_right_20251222_174429.json
      ...
  robot/
    usc_koch_move_the_orange_cup_from_left_to_right/
      data/
        chunk-000/
          file-000.parquet
      meta/
        episodes/
          chunk-000/
            file-000.parquet
        info.json
        stats.json
        tasks.parquet
      videos/
        observation.images.top/
          chunk-000/
            file-000.mp4
        observation.images.side/
          chunk-000/
            file-000.mp4
    usc_koch_throw_the_orange_cup_away_red_trash_can/
      ...
    ...
```

## Download Dataset

Before converting the dataset, you need to download it first:

```bash
cd /path/to/robometer
bash dataset_upload/data_scripts/usc_koch_human_robot_paired/download_datasets.sh
```

This script will:
1. Download robot datasets from HuggingFace (10 task datasets)
2. Download human recordings from Google Drive
3. Extract and organize the files

### Robot Datasets

The download script downloads the following robot datasets:
- `usc_koch_throw_the_orange_cup_away_red_trash_can`
- `usc_koch_throw_the_black_marker_away_blue_trash_can`
- `usc_koch_open_the_red_trash_bin_red_trash_bin`
- `usc_koch_open_the_green_trash_bin_green_trash_bin`
- `usc_koch_open_the_blue_trash_bin_blue_trash_bin`
- `usc_koch_separate_the_red_and_orange_and_orange_cups`
- `usc_koch_separate_the_purple_and_orange_and_orange_cups`
- `usc_koch_separate_the_purple_and_red_and_red_cups`
- `usc_koch_move_the_orange_cup_from_right_to_left`
- `usc_koch_move_the_orange_cup_from_left_to_right`

### Human Metadata Format

Each human video has an associated JSON file:

```json
{
  "filename": "human_success_Move_the_orange_cup_from_the_left_to_the_right_20251222_174429.mp4",
  "timestamp": "20251222_174429",
  "trajectory_type": "human",
  "quality": "success",
  "notes": "Move the orange cup from the left to the right",
  "autocrop": true,
  "camera": "Camera 0"
}
```

The `notes` field contains the task instruction.

### Robot Metadata Format

Robot datasets use LeRobot's parquet format with `tasks.parquet` containing task instructions:

```python
# Example: tasks.parquet content
                                        task_index
Move the orange cup from left to right           0
```

## Configuration

We provide a single configuration file: `usc_koch_human_robot_paired.yaml`

You control which data to process by specifying the dataset name:
- **`human`**: Process only human demonstrations (~72 videos)
- **`robot`**: Process only robot demonstrations (~100 episodes, top camera view)

## Usage

### Step 1: Download the Dataset

```bash
cd /path/to/robometer
bash dataset_upload/data_scripts/usc_koch_human_robot_paired/download_datasets.sh
```

### Step 2: Convert to HuggingFace Format

Run the conversion twice - once for human, once for robot:

```bash
# Convert human demonstrations
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/usc_koch_human_robot_paired.yaml \
    --dataset.dataset_name=usc_koch_human_robot_paired_human

# Convert robot demonstrations
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/usc_koch_human_robot_paired.yaml \
    --dataset.dataset_name=usc_koch_human_robot_paired_robot
```

### Local Testing (No Hub Push)

```bash
# Test human with limited trajectories
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/usc_koch_human_robot_paired.yaml \
    --dataset.dataset_name=usc_koch_human_robot_paired_human \
    --output.max_trajectories=10 \
    --hub.push_to_hub=false

# Test robot with limited trajectories
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/usc_koch_human_robot_paired.yaml \
    --dataset.dataset_name=usc_koch_human_robot_paired_robot \
    --output.max_trajectories=10 \
    --hub.push_to_hub=false
```

## Output Format

The generated datasets will have the following schema:

- `id`: Unique trajectory identifier
- `task`: Task instruction text
- `lang_vector`: Language embedding of the task
- `data_source`: Dataset name
- `frames`: Path to video file
- `is_robot`: Boolean (False for human, True for robot)
- `quality_label`: Always "successful" for this dataset
- `preference_group_id`: None for this dataset
- `preference_rank`: None for this dataset

## Dataset Statistics

### Human Demonstrations
- **Total Videos**: 72
- **Unique Tasks**: ~20 different manipulation tasks
- **Quality**: All successful
- **Format**: MP4 videos with JSON metadata

### Robot Demonstrations
- **Total Datasets**: 10 tasks
- **Episodes per Task**: ~10 episodes
- **Total Episodes**: ~100
- **Camera View**: Top (default)
- **Quality**: All successful
- **Format**: LeRobot parquet with pre-rendered MP4 videos

### Task Pairing

Not all human tasks have corresponding robot demonstrations. The loader handles this by:
- Processing all human videos independently
- Processing all robot episodes independently
- Task instructions may differ slightly between human and robot (e.g., "Move the orange cup from the left to the right" vs "Move the orange cup from left to right")

## Technical Details

### Human Video Processing

- Videos are loaded using OpenCV
- Frames are extracted and converted to RGB numpy arrays
- Task instructions are read from JSON metadata (`notes` field)
- Videos are re-encoded to specified FPS and frame count

### Robot Video Processing

- Robot datasets are in LeRobot format
- Pre-rendered videos are loaded from `videos/observation.images.top/` directory (top camera view)
- Task instructions are read from `meta/tasks.parquet`
- Episode count is read from `meta/info.json`
- Videos are re-encoded to specified FPS and frame count

### Parallel Processing

- Supports configurable worker count for parallel processing
- Language embeddings are pre-computed before parallel execution
- Each episode is processed independently

### Task Matching

The loader includes string similarity matching functions for potential future use in pairing human and robot demonstrations:
- `_string_similarity()`: Calculates similarity between two strings (0-1)
- `_find_best_match()`: Finds the best matching string from options

## Troubleshooting

### Dataset Not Downloaded

**Error**: `FileNotFoundError: Dataset directory not found`

**Solution**: Run the download script first:
```bash
bash dataset_upload/data_scripts/usc_koch_human_robot_paired/download_datasets.sh
```

### Missing Human Recordings

**Error**: `FileNotFoundError: Human recordings directory not found`

**Solution**: 
- Ensure the Google Drive download completed successfully
- Check that `recordings.zip` was extracted to `datasets/usc_koch_human_robot_paired/human/`

### Missing Robot Datasets

**Error**: `FileNotFoundError: Robot datasets directory not found`

**Solution**:
- Ensure all robot datasets were downloaded from HuggingFace
- Check that datasets exist in `datasets/usc_koch_human_robot_paired/robot/`

### Video File Not Found

**Error**: Video file not found in robot dataset

**Solution**:
- Verify the robot dataset was fully downloaded (including the `videos/` directory)
- Check the `info.json` for the correct number of episodes
- Ensure the `observation.images.top` view exists in the `videos/` directory

### HuggingFace Download Issues

**Error**: Unable to download from HuggingFace

**Solution**:
- Ensure you have `huggingface-hub` installed
- Check your internet connection
- Try downloading manually: `hf download abraranwar/<dataset_name> --repo-type=dataset`

### Google Drive Download Issues

**Error**: `gdown` fails to download

**Solution**:
- Ensure `gdown` is installed: `pip install gdown`
- Check your internet connection
- Try downloading manually from the Google Drive link in the script

## Example Output Structure

### Human Demonstrations
```
robometer_dataset/usc_koch_human_robot_paired_human/
  usc_koch_human_robot_paired_human/
    shard_0000/
      episode_000000/
        human.mp4
      episode_000001/
        human.mp4
      ...
```

### Robot Demonstrations
```
robometer_dataset/usc_koch_human_robot_paired_robot/
  usc_koch_human_robot_paired_robot/
    shard_0000/
      episode_000000/
        robot.mp4
      episode_000001/
        robot.mp4
      ...
```

## Notes

- All demonstrations in this dataset are labeled as "successful"
- Human and robot demonstrations are processed independently (not explicitly paired)
- Task instructions may differ slightly between human and robot versions
- The loader handles missing correspondences gracefully
- Pre-rendered robot videos from LeRobot are used directly (not re-rendered from raw data)
- Robot demonstrations have consistent 30 FPS in the source data

