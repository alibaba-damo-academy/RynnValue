# UTD SO101 Clean Policy Ranking Dataset Guide

This guide explains how to integrate and use the UTD SO101 Clean Policy Ranking dataset with the Robometer pipeline.

## Overview

- UTD SO101 Clean Policy Ranking is a robotics dataset with quality-labeled trajectories for manipulation tasks
- The dataset contains pre-recorded GIF files with two camera views: **wrist** and **top**
- Quality labels include: **successful**, **suboptimal**, and **failure**
- Each episode includes task instructions from a VLA (Vision-Language-Action) model

## Dataset Structure

The dataset is organized as follows:

```
<dataset_path>/
  ROBOMETER_UTD_UNCLUTTER_succ-20251223T020758Z-3-001/
    ROBOMETER_UTD_UNCLUTTER_succ/
      episode_0_top.gif
      episode_0_wrist.gif
      episode_1_top.gif
      episode_1_wrist.gif
      ...
      vla_task.json
  ROBOMETER_UTD_UNCLUTTER_closs_succ-20251223T020830Z-3-001/
    ROBOMETER_UTD_UNCLUTTER_closs_succ/
      episode_0_top.gif
      episode_0_wrist.gif
      ...
      vla_task.json
  ROBOMETER_UTD_UNCLUTTER_fail-20251223T020841Z-3-001/
    ROBOMETER_UTD_UNCLUTTER_fail/
      episode_0_top.gif
      episode_0_wrist.gif
      ...
      vla_task.json
```

### Quality Label Mapping

The loader automatically detects quality labels from folder names:
- Folders with `"succ"` (but not `"closs_succ"`) → **successful**
- Folders with `"closs_succ"` → **suboptimal** (close success)
- Folders with `"fail"` → **failure**

### Task Instructions Format

The `vla_task.json` file in each quality folder contains an array of task instructions:

```json
[
    "Press the button.",
    "Put the bread in the oven.",
    "Put the marker into the pen cup.",
    "Put the red bowl on the blue plate.",
    "Put the pear in the yellow plate.",
    "Collect the fork to the yellow box.",
    "Stack the green block on the red block.",
    "Add pepper to the green bowl.",
    "Put the eraser to the blue pencil box.",
    "Put the red cup on the purple coaster."
]
```

The array index corresponds to the episode number (e.g., index 0 → `episode_0_wrist.gif`).

## Two Datasets: Wrist and Top Views

Since each episode has two camera views, we create **two separate datasets**:

1. **utd_so101_clean_policy_ranking_wrist**: Uses wrist camera view
2. **utd_so101_clean_policy_ranking_top**: Uses top-down camera view

Each dataset is processed and uploaded independently.

## Configuration Files

### Wrist View Configuration
File: `dataset_upload/configs/data_gen_configs/utd_so101_clean_policy_ranking_wrist.yaml`

```yaml
dataset:
  dataset_path: ~/projects/robometer/datasets/utd_prank
  dataset_name: utd_so101_clean_policy_ranking_wrist

output:
  output_dir: ./robometer_dataset/utd_so101_clean_policy_ranking_wrist
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 4

hub:
  push_to_hub: true
  hub_repo_id: utd_so101_clean_policy_ranking_wrist
```

### Top View Configuration
File: `dataset_upload/configs/data_gen_configs/utd_so101_clean_policy_ranking_top.yaml`

```yaml
dataset:
  dataset_path: ~/projects/robometer/datasets/utd_prank
  dataset_name: utd_so101_clean_policy_ranking_top

output:
  output_dir: ./robometer_dataset/utd_so101_clean_policy_ranking_top
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 4

hub:
  push_to_hub: true
  hub_repo_id: utd_so101_clean_policy_ranking_top
```

## Usage

### Convert Wrist View Dataset

```bash
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/utd_so101_clean_policy_ranking_wrist.yaml
```

### Convert Top View Dataset

```bash
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/utd_so101_clean_policy_ranking_top.yaml
```

### Local Testing (No Hub Push)

```bash
# Test with wrist view
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/utd_so101_clean_policy_ranking_wrist.yaml \
    --output.max_trajectories=5 \
    --hub.push_to_hub=false

# Test with top view
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/utd_so101_clean_policy_ranking_top.yaml \
    --output.max_trajectories=5 \
    --hub.push_to_hub=false
```

### Process Both Views Sequentially

```bash
# Process wrist view
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/utd_so101_clean_policy_ranking_wrist.yaml

# Process top view
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/utd_so101_clean_policy_ranking_top.yaml
```

## Output Format

The generated datasets will have the following schema:

- `id`: Unique trajectory identifier
- `task`: Task instruction text (from vla_task.json)
- `lang_vector`: Language embedding of the task
- `data_source`: Dataset name (includes view type)
- `frames`: Path to video file (converted from GIF)
- `is_robot`: Boolean (True for this dataset)
- `quality_label`: One of "successful", "suboptimal", "failure"
- `preference_group_id`: None for this dataset
- `preference_rank`: None for this dataset

## Dataset Statistics

Based on the directory structure:
- **Quality Labels**: successful (~10 episodes), suboptimal (~10 episodes), failure (~10 episodes)
- **Camera Views**: 2 (wrist, top)
- **Total Episodes per View**: ~30
- **Total Tasks**: 10 unique manipulation tasks
- **Format**: GIF files (converted to MP4 for storage)

## Technical Details

### GIF to Video Conversion

- The loader reads GIF files using PIL (Python Imaging Library)
- Frames are extracted and converted to RGB numpy arrays
- Videos are re-encoded as MP4 with specified FPS and frame count
- Original quality is preserved during conversion

### Parallel Processing

- Supports configurable worker count for parallel processing
- Language embeddings are pre-computed before parallel execution
- Each episode is processed independently

### Episode Indexing

- Episodes are given global indices across all quality labels
- Indices are sequential (0, 1, 2, ...) regardless of quality label
- Each view maintains its own set of indices

## Troubleshooting

### GIF File Not Found
- Ensure the dataset path points to the parent directory containing the quality-labeled subdirectories
- Check that GIF files exist for the specified view (wrist or top)

### Missing vla_task.json
- Each quality folder must contain a `vla_task.json` file
- The JSON file should be an array of task instructions

### PIL/Pillow Issues
- If you encounter GIF loading issues, ensure Pillow is properly installed
- Try: `pip install Pillow` or `uv pip install Pillow`

### Instruction Mismatch
- If an episode number is higher than the number of instructions in `vla_task.json`, that episode will be skipped
- Check the console output for warnings about missing instructions

## Example Output Structure

After processing, you'll have directory structures like:

### Wrist View
```
robometer_dataset/utd_so101_clean_policy_ranking_wrist/
  utd_so101_clean_policy_ranking_wrist/
    shard_0000/
      episode_000000/
        wrist.mp4
      episode_000001/
        wrist.mp4
      ...
```

### Top View
```
robometer_dataset/utd_so101_clean_policy_ranking_top/
  utd_so101_clean_policy_ranking_top/
    shard_0000/
      episode_000000/
        top.mp4
      episode_000001/
        top.mp4
      ...
```

## Notes

- The two datasets (wrist and top) are completely independent and can be processed/uploaded separately
- Quality labels are automatically detected from folder names
- All episodes from all quality levels are combined into a single dataset per view
- Language embeddings are cached to avoid redundant computation across quality labels

