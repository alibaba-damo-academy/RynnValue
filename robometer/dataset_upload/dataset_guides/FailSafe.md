# FailSafe Dataset Guide

This guide explains how to integrate and use the FailSafe dataset (correctdata_v2) with the Robometer pipeline.

Source: `https://huggingface.co/datasets/onepiece1999/correctdata_v2/tree/main`

## Overview

- The dataset contains three tasks with failures and successes:
  - `FailPickCube-v1`: "Pick up the red cube"
  - `FailPushCube-v1`: "push and move a cube to a goal region in front of it"
  - `FailStackCube-v1`: "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling."
- Each task has ~100 seed folders. Each seed contains a `Ground_Truth/` (success) and multiple failure attempt folders.
- Each attempt has camera views `front/`, `side/`, `wrist/` with ordered `.png` frames.
- Additionally, JSON files (e.g., `vla_data_FailPickCube-v1.json`, `vla_data_GT_PickCube-v1.json`) provide sub-task annotated mini-trajectories with success/failure labels.

## Directory Structure

```
<dataset_path>/
  FailPickCube-v1/
    0/
      Ground_Truth/
        front/0.png ...
        side/*.png
        wrist/*.png
      1_trans_x_2/
        front/*.png
        side/*.png
        wrist/*.png
    1/
      ...
  FailPushCube-v1/
    ...
  FailStackCube-v1/
    ...
  json_files/
    vla_data_FailPickCube-v1.json
    vla_data_GT_PickCube-v1.json
    ...
```

## Configuration (configs/data_gen_configs/failsafe.yaml)

```yaml
# configs/data_gen_configs/failsafe.yaml

dataset:
  dataset_path: ./datasets/failsafe
  dataset_name: failsafe

output:
  output_dir: ./robometer_dataset/failsafe_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false

hub:
  push_to_hub: true
  hub_repo_id: failsafe_rfm
```

## Loader

- File: `dataset_upload/dataset_loaders/failsafe_loader.py`
- Function: `load_failsafe_dataset(dataset_path)`
- The loader:
  - Builds full episodes from `Ground_Truth/` (success) and failure attempt folders for the chosen camera `view`.
  - Parses `vla_data_*.json` entries to create sub-task mini-trajectories using the specified image paths and labels; includes the `sub_task` text in the trajectory `task` field.
  - Uses the per-task instructions above when JSONs do not specify an instruction.

## Usage

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/failsafe.yaml
```

This will:
- Load full episodes and sub-task mini-trajectories for the selected `view`
- Generate web-optimized videos
- Create a HuggingFace dataset ready to push or save

## Notes

- If JSON files are at the dataset root instead of `json_files/`, the loader will look there as a fallback.
- You can switch `view` in the YAML to `side` or `wrist` to regenerate the dataset for a different camera.
