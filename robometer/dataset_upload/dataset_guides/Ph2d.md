# Ph2d Dataset Guide

This guide explains how to integrate and use the Ph2d dataset with the Robometer training pipeline.

## Overview

Ph2d is a dataset organized as sequences (folders). Each sequence contains HDF5 files with encoded image frames under the keys:

- `observation.image.right`
- `observation.image.left`

Frames are stored as encoded image buffers and must be decoded via OpenCV.

## Directory Structure

```
<dataset_path>/
  sequence_0001/
    metadata.json           # Optional, dataset-specific (TODO: parse)
    traj_0001.h5
    traj_0002.h5
  sequence_0002/
    metadata.json
    traj_0001.h5
```

## HDF5 Format

- Each HDF5 file contains one or both datasets:
  - `observation.image.right`
  - `observation.image.left`
- Each dataset stores frames as encoded image buffers. The loader decodes each frame using:

```python
cv2.imdecode(buffer, cv2.IMREAD_COLOR)
```

Then converts to RGB for consistency.

## Configuration (configs/data_gen_configs/ph2d.yaml)

```yaml
# configs/data_gen_configs/ph2d.yaml

dataset:
  dataset_path: ./datasets/ph2d
  dataset_name: ph2d
  camera: right  # or left

output:
  output_dir: ./robometer_dataset/ph2d_rfm
  max_trajectories: -1
  max_frames: -1
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false

hub:
  push_to_hub: true
  hub_repo_id: ph2d_rfm
```

## Usage

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/ph2d.yaml
```

This will:
- Load Ph2d sequences and HDF5 trajectories
- Decode frames from the selected camera stream
- Convert to the standard HF dataset format

## Data Fields

Each trajectory includes:
- `id`: Unique identifier
- `task`: Defaults to sequence folder name (TODO: map from metadata.json when spec is known)
- `frames`: A `Ph2dFrameloader` that decodes frames on demand (returns `(T, H, W, 3)` RGB, `uint8`)
- `is_robot`: False
- `quality_label`: "successful"
- `partial_success`: 1
- `data_source`: `ph2d`

## Metadata

- `metadata.json` is optional per sequence. Its structure is dataset-specific.
- TODO: Implement parsing to extract captions/tasks per trajectory when the schema is provided.

## Troubleshooting

- Key not found: Ensure the HDF5 files contain the expected keys.
- Decode errors: Confirm the stored buffers are valid encoded images (e.g., JPEG/PNG).
- Performance: Decoding is done on-demand; consider caching if needed.
