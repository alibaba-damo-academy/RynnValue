# Robometer MIT Franka Dataset Guide

This guide explains how to integrate and use the Robometer MIT Franka dataset with the Robometer pipeline.
Source: https://drive.google.com/drive/folders/1dd62YeKY9-bTeK5fkjljzoFkNX2uZ1YD?usp=sharing

## Overview

- Robometer MIT Franka is a robotics dataset with quality-labeled trajectories for manipulation tasks
- The dataset contains pre-recorded MP4 videos organized by task and quality level
- Quality labels include: **successful**, **suboptimal**, and **failure**
- Each trajectory has two camera views: **external (ext)** and **wrist**

## Dataset Structure

The dataset is organized as follows:

```
ROBOMETER_MIT_Franka/
  foldtowel/
    success/
      Tue_Jan_20_23_49_14_2026_ext.mp4
      Tue_Jan_20_23_49_14_2026_wrist.mp4
      ...
    suboptimal/
      Tue_Jan_20_23_52_01_2026_ext.mp4
      Tue_Jan_20_23_52_01_2026_wrist.mp4
      ...
    failure/
      Tue_Jan_20_23_54_53_2026_ext.mp4
      Tue_Jan_20_23_54_53_2026_wrist.mp4
      ...
  movebanana/
    success/
    suboptimal/
    failure/
  movemouse/
  pourpebble/
  pulltissue/
  putspoon/
  stirpot/
```

### Tasks Included

1. **foldtowel**: Fold the towel in half.
2. **movebanana**: Pick up the banana from the blue plate and place it on the green plate.
3. **movemouse**: Pick up the mouse and place it right next to the laptop, while avoiding spilling coffee.
4. **pourpebble**: Pour the pebbles from the cup onto the plate.
5. **pulltissue**: Pull a tissue from the tissue box.
6. **putspoon**: Pick up the spoon and place it inside the cup.
7. **stirpot**: Pick up the spatula and stir the beans in the pot.

### Camera Views

Each trajectory has two synchronized camera views:
- **ext (external)**: Third-person view of the robot and workspace
- **wrist**: First-person view from the robot's wrist camera

By default, the loader processes both views as separate trajectories, each with the same task instruction but different visual observations. You can optionally exclude wrist camera views using the `exclude_wrist_cam` configuration option.

### Quality Labels

- **success**: Trajectory successfully completes the task
- **suboptimal**: Trajectory completes the task but with inefficiencies or errors
- **failure**: Trajectory fails to complete the task

## Configuration

Configuration file: `dataset_upload/configs/data_gen_configs/new_mit_franka.yaml`

```yaml
dataset:
  dataset_path: ~/robometer/datasets/ROBOMETER_MIT_Franka
  dataset_name: rfm_new_mit_franka_rfm
  exclude_wrist_cam: false  # Set to true to only process external camera views

output:
  output_dir: ./robometer_dataset/rfm_new_mit_franka_rfm
  max_trajectories: -1  # -1 for all trajectories
  max_frames: 32
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 4

hub:
  push_to_hub: true
  hub_repo_id: rfm_new_mit_franka_rfm
```

### Camera View Options

The dataset includes both external and wrist camera views. You can configure which views to process:

- **`exclude_wrist_cam: false`** (default): Process both external and wrist camera views as separate trajectories
- **`exclude_wrist_cam: true`**: Only process external camera views, skip wrist camera entirely

This is useful if you only want third-person views or need to reduce dataset size.

## Usage

### Convert Dataset to HuggingFace Format

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/new_mit_franka.yaml
```

To only process external camera views (exclude wrist camera):

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/new_mit_franka.yaml --dataset.exclude_wrist_cam=true
```

This will:
- Scan all task folders and their quality subfolders
- Load the corresponding MP4 videos for both camera views (or only external if `exclude_wrist_cam: true`)
- Process and resample videos to the specified frame count and FPS
- Generate language embeddings for task instructions
- Create a HuggingFace dataset with proper quality labels
- Optionally push to HuggingFace Hub

### Quality Label Mapping

The loader automatically normalizes quality labels:
- `"success"` → `"successful"`
- `"failure"` → `"failure"` (unchanged)
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

## Task Instructions Mapping

The loader uses the following mapping from folder names to natural language instructions:

| Folder Name  | Task Instruction                                                                       |
|--------------|----------------------------------------------------------------------------------------|
| foldtowel    | Fold the towel in half                                                                 |
| movebanana   | Pick up the banana from the blue plate and place it on the green plate                 |
| movemouse    | Pick up the mouse and place it right next to the laptop, while avoiding spilling coffee |
| pourpebble   | Pour the pebbles from the cup onto the plate                                           |
| pulltissue   | Pull a tissue from the tissue box                                                      |
| putspoon     | Pick up the spoon and place it inside the cup                                          |
| stirpot      | Pick up the spatula and stir the beans in the pot                                      |

These mappings are defined in the loader file (`new_mit_franka_loader.py`) and can be updated if needed.

## Notes

- Videos are already in MP4 format, so the loader reads them directly using OpenCV
- The loader supports parallel processing with configurable worker count
- Language embeddings are cached to avoid redundant computations
- Output videos maintain the same content but are resampled to the specified frame count and FPS
- Both camera views (ext and wrist) are processed as separate trajectories by default
- Use `exclude_wrist_cam: true` to only process external camera views and reduce dataset size by ~50%

## Troubleshooting

### Video File Not Found
- Ensure the dataset path points to the ROBOMETER_MIT_Franka directory
- Check that video files exist in the expected task/quality subfolders

### Missing Task Folders
- The loader expects folders named: foldtowel, movebanana, movemouse, pourpebble, pulltissue, putspoon, stirpot
- Each task folder should contain subfolders: success, suboptimal, failure

### OpenCV Issues
- If you encounter video codec issues, ensure OpenCV is properly installed with video support
- Try: `pip install opencv-python-headless` or `pip install opencv-python`

### Unknown Task Warning
- If a task folder is not in the TASK_INSTRUCTIONS mapping, it will be skipped
- Add new tasks to the mapping in `new_mit_franka_loader.py`

### Filtering Camera Views
- Set `exclude_wrist_cam: true` in the YAML config to skip wrist camera videos
- Or pass it as a command-line argument: `dataset.exclude_wrist_cam=true`
- This will process only external camera views, roughly halving the dataset size

## Example Output

After processing, you'll have a directory structure like:

```
robometer_dataset/rfm_new_mit_franka_rfm/
  rfm_new_mit_franka_rfm/
    shard_0000/
      episode_000000/
        foldtowel_ext.mp4
      episode_000001/
        foldtowel_wrist.mp4
      ...
```

With `exclude_wrist_cam: true`, only external camera videos are included:

```
robometer_dataset/rfm_new_mit_franka_rfm/
  rfm_new_mit_franka_rfm/
    shard_0000/
      episode_000000/
        foldtowel_ext.mp4
      episode_000001/
        movebanana_ext.mp4
      ...
```

And a HuggingFace Dataset with all trajectory metadata.

## Dataset Statistics

The dataset contains approximately 304 video files (152 trajectory pairs with ext and wrist views) across 7 different manipulation tasks, with varying distributions of successful, suboptimal, and failed demonstrations.

When `exclude_wrist_cam: true` is set, approximately 152 external camera videos are processed (one per trajectory).
