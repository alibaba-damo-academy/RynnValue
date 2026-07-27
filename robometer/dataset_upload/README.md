# Adding New Datasets to Robometer

This guide explains how to add new datasets to the Robometer training pipeline.

## Supported Datasets

### Ready-to-Use Datasets
- **LIBERO**: Built-in HDF5 support → [📖 LIBERO Guide](dataset_guides/LIBERO.md)
- **AgiBotWorld**: ✅ Native streaming support → [📖 AgiBotWorld Guide](dataset_guides/AgiBotWorld.md)

### Custom Datasets
- **Add Your Own**: DROID, Bridge, or any custom dataset → [📖 Custom Dataset Guide](dataset_guides/CustomDataset.md)

## Quick Start

### Use Existing Datasets
```bash
# AgiBotWorld (streaming)
uv run python dataset_upload/generate_hf_dataset.py --config_path=dataset_upload/configs/data_gen_configs/agibot_world.yaml

# LIBERO (local files)
uv run python dataset_upload/generate_hf_dataset.py --config_path=dataset_upload/configs/data_gen.yaml
```

### Add Custom Dataset
1. **Read the guide**: [Custom Dataset Guide](dataset_guides/CustomDataset.md)
2. **Create loader**: `dataset_upload/{dataset_name}_loader.py`
3. **Add config**: `dataset_upload/configs/data_gen_configs/{dataset_name}.yaml`
4. **Test**: Run conversion and training

## Architecture Overview

Each dataset type has its own loader module. The main converter (`generate_hf_dataset.py`) is dataset-agnostic and works with any dataset-specific loader that follows the established interface.

### Output Formats

The converter supports two output formats:

**Video Mode** (`--output.use_video=true`):
- Creates MP4 video files using H.264 encoding
- Videos are stored in organized directories: `trajectory_XXXX/trajectory.mp4`
- Uses `datasets.Video()` feature for proper HuggingFace video display
- Supports configurable frame rate, resolution, and cropping

**Frame Mode** (`--output.use_video=false`):
- Creates individual JPG image files
- Images are stored in organized directories: `trajectory_XXXX/frame_XX.jpg`
- Uses `datasets.Sequence(datasets.Image())` feature for image galleries
- Supports configurable frame count and resolution

### Video Processing Features

The dataset converter includes built-in video processing:
- **📹 Automatic Resizing**: Videos resized to consistent dimensions (configurable shortest edge size)
- **⏱️ Frame Interpolation**: Downsamples to configurable frame count (default: all frames preserved)
- **🎬 MP4 Creation**: Creates H.264 encoded MP4 files for optimal HuggingFace compatibility
- **🎯 Quality Preservation**: Maintains visual quality while standardizing format
- **📁 File Organization**: Organizes videos in trajectory-specific directories

## Dataset Structure Requirements

Your dataset loader must produce trajectories in the following format:

```python
{
    'frames': Union[str, List[str]],   # Video file path (video mode) or list of image file paths (frame mode)
    'actions': np.ndarray,             # Actions 
    'is_robot': bool,                  # Whether this is robot data (True) or human data (False)
    'task': str,                       # Human-readable task description
    'optimal': str                     # Whether this trajectory is optimal
}
```

**Note**: The dataset converter automatically creates MP4 video files or individual frame images based on the `use_video` setting.

## Step-by-Step Guide

### 1. Create Your Dataset Loader

Create a new Python file in the `data/` directory following the naming convention: `{dataset_name}_loader.py`

Example: `droid_loader.py` or `bridge_loader.py`

```python
#!/usr/bin/env python3
"""
DROID dataset loader for the generic dataset converter for Robometer model training.
This module contains DROID-specific logic for loading and processing data.
"""

import os
import numpy as np
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

def load_droid_dataset(base_path: str) -> Dict[str, List[Dict]]:
    """Load DROID dataset and organize by task.
    
    Args:
        base_path: Path to the DROID dataset directory
        
    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """
    
    print(f"Loading DROID dataset from: {base_path}")
    
    task_data = {}
    
    # Your dataset-specific loading logic here
    # This is where you'll implement the logic to:
    # 1. Find and read your data files
    # 2. Extract frames, actions, rewards, etc.
    # 3. Organize by task
    # 4. Convert to the required format
    
    # Example structure (adapt to your dataset):
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"DROID dataset path not found: {base_path}")
    
    # Find your data files (adapt this to your dataset structure)
    data_files = list(base_path.glob("**/*.hdf5"))  # or whatever format you use
    
    for file_path in tqdm(data_files, desc="Processing DROID dataset"):
        task_name = file_path.stem
        
        # Load your data file
        # This is where you'll implement your specific loading logic
        trajectories = load_trajectories_from_file(file_path)
        
        task_data[task_name] = trajectories
    
    print(f"Loaded {sum(len(trajectories) for trajectories in task_data.values())} trajectories from {len(task_data)} tasks")
    return task_data

def load_trajectories_from_file(file_path: Path) -> List[Dict]:
    """Load trajectories from a single DROID data file."""
    trajectories = []
    
    # Implement your file loading logic here
    # This will depend on your dataset format (HDF5, JSON, pickle, etc.)
    
    # Example for HDF5 format:
    import h5py
    with h5py.File(file_path, 'r') as f:
        # Navigate your HDF5 structure and extract data
        # Convert to the required format
        pass
    
    return trajectories
```

### 2. Update the Main Converter

Add your dataset type to the main converter in `generate_hf_dataset.py`:

```python
# In the main() function, add your dataset type:
elif cfg.dataset.dataset_type == "droid":
    from dataset_loaders.droid_loader import load_droid_dataset
    # Load the trajectories using your loader
    task_data = load_droid_dataset(cfg.dataset.dataset_path)
    trajectories = flatten_task_data(task_data)
```

### 3. Test Your Loader

Create a simple test script to verify your loader works:

```python
# test_droid_loader.py
from droid_loader import load_droid_dataset
from helpers import flatten_task_data

# Test your loader
task_data = load_droid_dataset("/path/to/your/droid/dataset")
trajectories = flatten_task_data(task_data)

print(f"Loaded {len(trajectories)} trajectories")
print(f"Sample trajectory keys: {list(trajectories[0].keys())}")
print(f"Sample task: {trajectories[0].get('task', 'No task found')}")
```

### 4. Run Dataset Conversion

Use the main converter with your new dataset:

```bash
uv run python data/generate_hf_dataset.py \
    --config_path=configs/dataset_.yaml \
    --dataset.dataset_name=your_dataset \
    --dataset.dataset_path=/path/to/your/dataset \
    --output.output_dir=your_robometer_dataset \
    --output.max_trajectories=1000 \
    --output.max_frames=-1 \
    --output.use_video=true \
    --output.fps=10 \
    --output.shortest_edge_size=240 \
    --output.center_crop=false
```

### Visualize the Dataset

```bash
uv run python visualize_dataset.py --dataset_path=your_robometer_dataset/your_dataset_name
```

## Dataset-Specific Guides

📁 **[Browse All Dataset Guides](dataset_guides/)** - Complete overview with quick reference table

### Individual Guides
- **[📖 AgiBotWorld Guide](dataset_guides/AgiBotWorld.md)** - Streaming support, webdataset format
- **[📖 LIBERO Guide](dataset_guides/LIBERO.md)** - HDF5 files, simulation data  
- **[📖 Custom Dataset Guide](dataset_guides/CustomDataset.md)** - Add DROID, Bridge, or your own dataset

Each guide includes:
- ✅ Prerequisites and setup
- ✅ Configuration examples
- ✅ Troubleshooting tips
- ✅ Performance notes
- ✅ Integration with Robometer training