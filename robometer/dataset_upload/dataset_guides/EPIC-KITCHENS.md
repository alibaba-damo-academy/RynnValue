# EPIC-KITCHENS Dataset Guide

This guide explains how to integrate and use the EPIC-KITCHENS dataset with the Robometer training pipeline.

Source: `https://epic-kitchens.github.io/`, Lerobot dataset sourced from `https://huggingface.co/datasets/awsaf49/epic_kitchens_100`, (a subset of the original), and `csv` annotations sourced from `https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/master/EPIC_100_train.csv`.

This only loads the Epic-kitchens-100 added data, not any of the original epic-kitchens-55 data as that's difficult to download without a HF link (very slow).

## Assumptions

- The dataset root contains participant folders `P01`, `P02`, ...
- Each participant folder contains a `videos/` directory with `.MP4` files
- The dataset root contains `EPIC_100_train.csv` with at least columns:
  - `video_id` (maps to the video filename without `.MP4`)
  - `participant_id` (e.g., `P01`)
  - `start_frame` (start frame index)
  - `stop_frame` (end frame index)
  - `narration` (language instruction)

## Directory Structure

```
<dataset_path>/
  EPIC_100_train.csv
  P01/
    videos/
      <narration_id>.MP4
  P02/
    videos/
      <narration_id>.MP4
  ...
```

## Configuration (configs/data_gen_configs/epic.yaml)

```yaml
# configs/data_gen_configs/epic.yaml

dataset:
  dataset_path: ./datasets/epic_kitchens
  dataset_name: epic

output:
  output_dir: ./robometer_dataset/epic_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 30
  shortest_edge_size: 240
  center_crop: false
  num_workers: 4

hub:
  push_to_hub: true
  hub_repo_id: epic_rfm
```

## Usage

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/epic.yaml
```

This will:
- Parse `EPIC_100_train.csv`
- For each row, locate the participant video and extract the `[start_frame, stop_frame)` segment
- Convert segments to web-optimized videos
- Create a HuggingFace dataset (H2R/OXE-style streaming conversion)

## Data Fields

Each trajectory includes:
- `id`: Unique identifier
- `task`: The `narration` string from the CSV
- `frames`: Relative path to the generated clip video
- `is_robot`: False
- `quality_label`: "successful"
- `partial_success`: N/A (fixed by pipeline)
- `data_source`: `epic`

## Notes

- The loader uses batched, worker-parallel conversion similar to `h2r_loader.py` to control memory.
- Frame indices are respected using OpenCV frame seeking; if seeking fails, decoding starts from the beginning and stops at `stop_frame`.
- The language vector is computed from the narration and cached.

## Troubleshooting

- Video not found: Ensure `narration_id`.MP4 exists under the correct `PXX/videos/` directory.
- Empty segment: Verify `start_frame < stop_frame` and within the video length.
- Performance: Adjust `num_workers` and `batch_size` in the loader if necessary.
