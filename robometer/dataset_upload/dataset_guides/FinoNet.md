# FinoNet Dataset Guide

This guide explains how to integrate and use the FinoNet dataset with the Robometer training pipeline.

Source: `https://huggingface.co/datasets/jesbu1/fino-net-dataset`

## Overview

FinoNet contains episodes of manipulation tasks with success/failure labels. Each episode is a sequence of RGB images stored as PNG files in numbered directories, organized by task type.

## Directory Structure

The dataset structure (after unzipping `failure.zip`) should be:

```
<dataset_path>/
  failure/
    failnet_dataset/
      rgb_imgs/
        put_on/
          9/
            frame0000000.png
            frame0000024.png
            ...
        put_in/
          ...
        place/
          ...
        pour/
          ...
        push/
          ...
  put_on_annotation.txt
  put_in_annotation.txt
  place_annotation.txt
  pour_annotation.txt
  push_annotation.txt
```

- Task folders contain episode subdirectories (numbered by episode).
- Each episode directory contains PNG frames (e.g., `frame0000000.png`, `frame0000024.png`).
- Annotation files map episode numbers to labels (0 = success, 1 = failure).

## Annotation Format

Each annotation file (e.g., `put_on_annotation.txt`) contains comma-separated values:
```
name,label
9,0
10,1
...
```

- `name`: Episode number
- `label`: 0 for success, 1 for failure

## Task Instructions

The dataset includes 5 tasks with specific instructions:

- **put_on**: "put the single block on the table onto the stack"
- **put_in**: "put the thing on the table into the container"
- **place**: "place the block in your hand onto the stack"
- **pour**: "pour the contents of the cup into the receptacle on the table without spilling"
- **push**: "push the object to the right without knocking it over"

## Configuration (configs/data_gen_configs/fino_net.yaml)

```yaml
# configs/data_gen_configs/fino_net.yaml

dataset:
  dataset_path: /path/to/fino_net_dataset  # Root directory containing failure.zip or unzipped failure/
  dataset_name: fino_net

output:
  output_dir: datasets/fino_net_rfm
  max_trajectories: -1   # -1 for all episodes
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false

hub:
  push_to_hub: true
  hub_repo_id: fino_net_rfm
```

## Usage

### Step 1: Download Dataset

From HuggingFace:
```bash
# Download from: https://huggingface.co/datasets/jesbu1/fino-net-dataset/tree/main
# Or use huggingface_hub:
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jesbu1/fino-net-dataset', repo_type='dataset', local_dir='./fino_net_dataset')"
```

### Step 2: Unzip Dataset

```bash
cd ./fino_net_dataset
unzip failure.zip
```

### Step 3: Convert to HF Format

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/fino_net.yaml
```

This will:
- Load annotation files to map episodes to success/failure labels
- Discover all episodes across the 5 tasks
- Load PNG frames for each episode
- Convert frame sequences to MP4 videos
- Create HuggingFace dataset with proper metadata

## Data Fields

Each trajectory includes:
- `id`: Unique identifier
- `task`: Task instruction from the predefined mapping
- `frames`: Relative path to MP4 video file
- `is_robot`: `False` (human demonstration data)
- `quality_label`: "successful" or "failed" based on annotation label
- `partial_success`: 1.0 for success, 0.0 for failure
- `data_source`: `fino_net`

## Troubleshooting

- **No images found**: Verify that `failure.zip` has been unzipped and the directory structure matches the expected layout.
- **Missing annotations**: Ensure all 5 annotation files exist in the root directory.
- **Episode not in annotations**: Some episodes in the directories may not have corresponding annotation entries; these will be skipped with a warning.
- **Performance**: Large datasets with many frames per episode can be memory intensive. Consider limiting `max_trajectories` during testing.

## Notes

- The loader automatically detects task type from directory names and maps to appropriate instructions.
- Frames are loaded in sorted order based on frame number in the filename.
- The dataset provides valuable failure examples for robust reward modeling.

