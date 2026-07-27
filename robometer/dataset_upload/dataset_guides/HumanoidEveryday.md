# Humanoid Everyday Dataset Guide

This guide explains how to integrate and use the Humanoid Everyday dataset with the Robometer training pipeline.

Source: `https://github.com/ausbxuse/Humanoid-Everyday`

## Overview

The Humanoid Everyday Dataset is a diverse collection of humanoid robot (Unitree G1 and H1) demonstrations recorded at 30 Hz across everyday tasks. This dataset supports research in robot learning, imitation, and perception.

- **Total Download Size:** ~500 GB (across 250 tasks), over 100,000 time-step recorded
- **Tasks:** 260 diverse scenarios (loco-manipulation, basic manipulation, tool use, deformables, articulated objects, human–robot interaction)
- **Episodes per task:** 40
- **Recording Frequency:** 30 Hz

### Modalities captured

- **Low-dimensional:**  
  - Joint states (arm, leg, hand)  
  - IMU (orientation, accelerometer, gyroscope, RPY)  
  - Odometry/Kinematics (position, velocity, orientation)  
  - Hand pressure sensors (G1 only)  
  - Teleoperator hands/head actions from Apple Vision Pro  
  - Inverse kinematics data
- **High-dimensional:**  
  - Egocentric RGB images (480x640x3, PNG)  
  - Depth maps (480x640, uint16)  
  - LiDAR point clouds (~6k points per step, PCD)

## Prerequisites

### Install humanoid_everyday dataloader

```bash
git clone https://github.com/ausbxuse/Humanoid-Everyday
cd Humanoid-Everyday
uv pip install -e .
```

### Download dataset

Please visit the task spreadsheet to download your task of interest, or use the provided download script:

```bash
bash dataset_upload/data_scripts/humanoid_everyday/download_humanoid_everyday.sh
```

## Directory Structure

```
<dataset_path>/
  task1.zip
  task2.zip
  ...
  taskN.zip
```

Each zip file contains a complete task dataset with multiple episodes.

## Data Schema

Each time step is represented by a Python dictionary with the following fields:

```python
{
    # Scalar identifiers
    "time": np.float64,                # UNIX timestamp (s)
    "robot_type": np.str_,             # Robot model identifier (G1 only)

    # Robot states
    "states": {
        "arm_state": np.ndarray((14,), dtype=np.float64),   # 14 joint angles
        "leg_state": np.ndarray((15 or 13,), dtype=np.float64),  # 15 joint angles for G1, 13 for H1_2
        "hand_state": np.ndarray((14 or 12,), dtype=np.float64), # 14 joint angles for Unitree Dex3 Hand, 12 for Inspire Dextrous Hand
        "hand_pressure_state": [...],  # List of per-sensor readings (9 sensors per hand)
        "imu": {
            "quaternion": np.ndarray((4,), dtype=np.float64),    # [w, x, y, z]
            "accelerometer": np.ndarray((3,), dtype=np.float64), # [ax, ay, az]
            "gyroscope": np.ndarray((3,), dtype=np.float64),     # [gx, gy, gz]
            "rpy": np.ndarray((3,), dtype=np.float64)            # [roll, pitch, yaw]
        },
        "odometry": {
            "position": np.ndarray((3,), dtype=np.float64),  # [x, y, z]
            "velocity": np.ndarray((3,), dtype=np.float64),  # [vx, vy, vz]
            "rpy": np.ndarray((3,), dtype=np.float64),      # [roll, pitch, yaw]
            "quat": np.ndarray((4,), dtype=np.float64)      # [w, x, y, z]
        }
    },

    # Control commands and solutions
    "actions": {
        "right_angles": np.ndarray((7,), dtype=np.float64),  # commanded joint angles
        "left_angles": np.ndarray((7,), dtype=np.float64),   # commanded joint angles
        "armtime": np.float64,                               # timestamp
        "iktime": np.float64,                                # timestamp
        "sol_q": np.ndarray((14,), dtype=np.float64),       # solution joint angles
        "tau_ff": np.ndarray((14,), dtype=np.float64),      # feedforward torques
        "head_rmat": np.ndarray((3, 3), dtype=np.float64),  # rotation matrix
        "left_pose": np.ndarray((4, 4), dtype=np.float64),  # homogeneous transform
        "right_pose": np.ndarray((4, 4), dtype=np.float64)  # homogeneous transform
    },

    # High-dimensional observations
    "image": np.ndarray((480, 640, 3), dtype=np.uint8),     # RGB image
    "depth": np.ndarray((480, 640), dtype=np.uint16),       # Depth map
    "lidar": np.ndarray((~6000, 3), dtype=np.float64)       # around 6000 points for lidar point cloud
}
```

## Configuration (configs/data_gen_configs/humanoid_everyday.yaml)

```yaml
# configs/data_gen_configs/humanoid_everyday.yaml

dataset:
  dataset_path: "./datasets/humanoid_everyday"  # Path containing zip files
  dataset_name: humanoid_everyday_rfm

output:
  output_dir: ./robometer_dataset/humanoid_everyday_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 4

hub:
  push_to_hub: true
  hub_repo_id: humanoid_everyday_rfm
```

## Usage

```bash
uv run python -m dataset_upload.generate_hf_dataset --config dataset_upload/configs/data_gen_configs/humanoid_everyday.yaml
```

This will:
- Find all zip files in the specified dataset path
- For each zip file, extract the task name and load episodes using the humanoid_everyday dataloader
- Extract RGB images from each episode
- Convert frames to web-optimized videos and create a HuggingFace dataset
- Use the zip filename (without extension) as the task description

## Data Fields

Each trajectory includes:
- `id`: Unique identifier
- `task`: Task name extracted from zip filename
- `frames`: Relative path to the generated clip video
- `is_robot`: True
- `quality_label`: "successful"
- `partial_success`: N/A (fixed by pipeline)
- `data_source`: `humanoid_everyday`

## Example Usage with Dataloader

```python
from humanoid_everyday import Dataloader

# Load your downloaded task's dataset zip file (e.g., the "push_a_button" task)
ds = Dataloader("~/Downloads/push_a_button.zip")
print("Episode length of dataset:", len(ds))

# Displaying high dimensional data at first episode, second timestep.
ds.display_image(0, 1)
ds.display_depth_point_cloud(0, 1)
ds.display_lidar_point_cloud(0, 1)

for i, episode in enumerate(ds):
    if i == 1:  # episode 1
        print("RGB image shape:", episode[0]["image"].shape)  # (480, 640, 3)
        print("Depth map shape:", episode[0]["depth"].shape)  # (480, 640)
        print("LiDAR points shape:", episode[0]["lidar"].shape)  # (~6000, 3)

        batch = episode[0:4]  # batch loading episodes
        print(batch[1]["image"].shape)
        print(batch[0]["image"].shape)
```

## Troubleshooting

- **Missing humanoid_everyday package**: Install it with `pip install humanoid_everyday` or clone and install from the GitHub repository
- **No zip files found**: Ensure the dataset_path contains zip files with humanoid everyday datasets
- **Import errors**: Make sure the humanoid_everyday package is properly installed and accessible
- **Memory issues**: Adjust `max_frames` and `num_workers` parameters to reduce memory usage
- **Long episodes**: Episodes longer than 1000 frames are automatically skipped to prevent memory issues

## License

This dataset is released under the MIT License.
