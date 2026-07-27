# LIBERO Dataset Guide

LIBERO is a benchmark for lifelong robot learning with built-in support in the Robometer training pipeline.

## Overview

- **📁 Local File Support**: Processes HDF5 files from local storage
- **🎮 Simulation Data**: High-quality manipulation tasks
- **🏠 Multiple Environments**: Living room, kitchen, office, and study scenarios
- **📊 Structured Tasks**: Clear task descriptions and optimal trajectories

## Prerequisites

### Download LIBERO Dataset
```bash
# Clone or download LIBERO dataset
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
# Follow LIBERO installation instructions for dataset download

# This should work too
git submodule update --init --recursive

cd deps/libero/LIBERO
uv run python benchmark_scripts/download_libero_datasets.py --datasets DATASET
```
where DATASET is chosen from `[libero_spatial, libero_object, libero_100, libero_goal]`.


## Quick Start

### 0. Set Hugging Face repo ID
Before we start, you must have an HF account which will be pushed to.
You will set this by setting
```
export HF_USERNAME=<insert HF username here>
```

Then, for each dataset, run with the all the datasets you would like to process

### Option 1: Use Default Configuration
```bash
uv run python dataset_upload/generate_hf_dataset.py \
    --config_path=dataset_upload/configs/data_gen_configs/libero.yaml\
    --dataset.dataset_path=deps/libero/LIBERO/libero/datasets/libero_90 \
    --dataset.dataset_name=libero_90
```

If all your LIBERO data exists in the path above, you can use the following utility script
```bash
uv run bash dataset_upload/data_scripts/libero/gen_all_libero.sh
```

### Option 2: Custom Configuration
```bash
uv run python dataset_upload/generate_hf_dataset.py \
    --config_path=dataset_upload/configs/data_gen_configs/libero.yaml \
    --dataset.dataset_path=/path/to/your/libero/dataset \
    --dataset.dataset_name=libero_custom \
    --output.output_dir=libero_robometer_dataset \
    --output.max_trajectories=1000 \
    --output.use_video=true \
    --output.fps=10
```

## Configuration Options

Create a custom config file `configs/data_gen_configs/libero.yaml`:

```yaml
dataset:
  dataset_path: LIBERO/libero/datasets/libero_90
  dataset_name: libero_90

output:
  output_dir: libero_dataset
  max_trajectories: -1  # Process all trajectories
  max_frames: 32
  use_video: true
  fps: 10

hub:
  push_to_hub: false
  hub_repo_id: your-username/libero_rfm
```

## Data Structure Processed

```
LIBERO Dataset:
├── *.hdf5 files             ← PROCESSED
│   ├── /data/
│   │   └── trajectory_*/
│   │       ├── obs/
│   │       │   └── agentview_rgb    ← EXTRACTED as frames
│   │       └── actions              ← EXTRACTED as actions
└── Generated Output:
    ├── frames: List[np.ndarray]     ← RGB video frames
    ├── actions: np.ndarray          ← Robot actions
    ├── task: str                    ← Parsed from filename
    └── optimal: "optimal"           ← All LIBERO data assumed optimal
```

## Supported LIBERO Variants

- **LIBERO-90**: 90 tasks across 4 environments
- **LIBERO-10**: 10 benchmark tasks
- **Custom datasets**: Any LIBERO-format HDF5 files

## Sample Output

```
Loading LIBERO dataset from: LIBERO/libero/datasets/libero_90
Found 90 HDF5 files
Processing LIBERO dataset, 90 files: 100%|██████████| 90/90

Sample trajectory:
- Task: "stack the right bowl on the left bowl and place them in the tray"
- Frames: (128, 128, 128, 3) RGB images
- Actions: (128, 7) joint positions
- Environment: LIVING_ROOM_SCENE4
```

## File Name Parsing

LIBERO dataset automatically parses task information from HDF5 filenames:

```
LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray.hdf5
│              │    │
│              │    └── Task description
│              └── Scene identifier  
└── Environment type
```

## Performance Notes

- **Processing Speed**: ~2-5 files/second
- **Memory Usage**: Moderate (loads one HDF5 file at a time)
- **Storage**: Variable (depends on trajectory length)
- **Video Encoding**: Converts RGB arrays to MP4 format

## Troubleshooting

### HDF5 File Issues
```python
# Check HDF5 file structure
import h5py
with h5py.File('path/to/file.hdf5', 'r') as f:
    print(list(f.keys()))  # Should show 'data'
    print(list(f['data'].keys()))  # Should show trajectory keys
```

### Missing Observations
Ensure your LIBERO dataset has the expected structure:
```
/data/demo_0/obs/agentview_rgb  # RGB frames
/data/demo_0/actions            # Action sequences
```

### Memory Issues
For large LIBERO datasets:
```bash
# Process in chunks
uv run python data/generate_hf_dataset.py \
    --config_path=configs/data_gen_configs/libero.yaml \
    --output.max_trajectories=100  # Limit trajectories
```

## Integration with Robometer Training

```bash
# Train on processed LIBERO dataset
uv run accelerate launch --config_file configs/fsdp.yaml train.py \
    --config_path=configs/config.yaml \
    --dataset.dataset_path=libero_dataset/libero_90
```