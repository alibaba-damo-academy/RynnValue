# Custom Dataset Guide

Learn how to add your own dataset (like DROID, Bridge, etc.) to the Robometer training pipeline.

## Overview

Adding a custom dataset involves:
1. Creating a dataset loader module
2. Implementing the required data format
3. Integrating with the main converter
4. Testing and validation

## Required Data Format

Your dataset loader must produce trajectories in this format:

```python
{
    'frames': List[Union[str, bytes, np.ndarray]],  # Video file paths, MP4 bytes, or frame arrays
    'actions': np.ndarray,                          # Robot actions (N, action_dim)
    'is_robot': bool,                               # True for robot data, False for human
    'task': str,                                    # Human-readable task description
    'optimal': str                                  # "optimal", "suboptimal", or "failed"
}
```

## Step 0: Set Hugging Face repo ID
Before we start, you must have an HF account which will be pushed to.
You will set this by setting
```
export HF_USERNAME=<insert HF username here>
```


## Step 1: Create Dataset Loader

Create `data/{dataset_name}_loader.py`:

```python
#!/usr/bin/env python3
"""
{DatasetName} dataset loader for Robometer model training.
"""

import os
import numpy as np
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

def load_{dataset_name}_dataset(base_path: str) -> Dict[str, List[Dict]]:
    """Load {DatasetName} dataset and organize by task.
    
    Args:
        base_path: Path to the {DatasetName} dataset directory
        
    Returns:
        Dictionary mapping task names to lists of trajectory dictionaries
    """
    
    print(f"Loading {DatasetName} dataset from: {base_path}")
    print("=" * 100)
    print(f"LOADING {DATASET_NAME} DATASET")
    print("=" * 100)
    
    task_data = {}
    base_path = Path(base_path)
    
    if not base_path.exists():
        raise FileNotFoundError(f"{DatasetName} dataset path not found: {base_path}")
    
    # TODO: Implement your dataset-specific logic here
    # Example structure:
    
    # Find your data files
    data_files = list(base_path.glob("**/*.{your_format}"))  # e.g., *.pkl, *.json, *.hdf5
    
    for file_path in tqdm(data_files, desc=f"Processing {DatasetName} dataset"):
        task_name = file_path.stem  # or extract from your naming scheme
        
        # Load your data file
        trajectories = load_trajectories_from_file(file_path)
        
        if trajectories:
            task_data[task_name] = trajectories
    
    print(f"Loaded {sum(len(trajectories) for trajectories in task_data.values())} trajectories from {len(task_data)} tasks")
    return task_data


def load_trajectories_from_file(file_path: Path) -> List[Dict]:
    """Load trajectories from a single data file."""
    trajectories = []
    
    # TODO: Implement your file loading logic here
    # This depends on your dataset format (HDF5, JSON, pickle, etc.)
    
    # Example for different formats:
    
    # For HDF5:
    # import h5py
    # with h5py.File(file_path, 'r') as f:
    #     # Navigate your HDF5 structure
    #     pass
    
    # For JSON:
    # import json
    # with open(file_path, 'r') as f:
    #     data = json.load(f)
    
    # For pickle:
    # import pickle
    # with open(file_path, 'rb') as f:
    #     data = pickle.load(f)
    
    # Convert to required format:
    # trajectory = {
    #     'frames': your_frames,        # List of file paths, bytes, or numpy arrays
    #     'actions': your_actions,      # numpy array of shape (sequence_length, action_dim)
    #     'is_robot': True,            # or False for human demonstrations
    #     'task': your_task_description,
    #     'optimal': 'optimal'         # or 'suboptimal', 'failed'
    # }
    # trajectories.append(trajectory)
    
    return trajectories
```

## Step 2: Add to Main Converter

Edit `data/generate_hf_dataset.py` to include your dataset:

```python
# In the main() function, add your dataset type:
elif "{dataset_name}" in cfg.dataset.dataset_name.lower():
    from {dataset_name}_loader import load_{dataset_name}_dataset
    # Load the trajectories using your loader
    task_data = load_{dataset_name}_dataset(cfg.dataset.dataset_path)
    trajectories = flatten_task_data(task_data)
```

## Step 3: Create Configuration File

Create `configs/data_gen_configs/{dataset_name}.yaml`:

```yaml
# {DatasetName} dataset configuration
dataset:
  dataset_path: /path/to/your/{dataset_name}/dataset
  dataset_name: {dataset_name}

output:
  output_dir: {dataset_name}_dataset
  max_trajectories: 1000  # Adjust as needed
  max_frames: 32
  use_video: true
  fps: 10

hub:
  push_to_hub: false
  hub_repo_id: your-username/{dataset_name}_rbm
```

## Step 4: Test Your Implementation

Create a test script `test_{dataset_name}_loader.py`:

```python
from {dataset_name}_loader import load_{dataset_name}_dataset
from helpers import flatten_task_data

# Test your loader
task_data = load_{dataset_name}_dataset("/path/to/your/{dataset_name}/dataset")
trajectories = flatten_task_data(task_data)

print(f"Loaded {len(trajectories)} trajectories")
print(f"Sample trajectory keys: {list(trajectories[0].keys())}")
print(f"Sample task: {trajectories[0].get('task', 'No task found')}")
print(f"Sample frames type: {type(trajectories[0]['frames'])}")
print(f"Sample actions shape: {trajectories[0]['actions'].shape}")
```

## Step 5: Run Dataset Conversion

```bash
uv run python data/generate_hf_dataset.py \
    --config_path=configs/data_gen_configs/{dataset_name}.yaml
```

## Frame Format Options

### Option 1: Video File Paths
```python
'frames': ['/path/to/video1.mp4', '/path/to/video2.mp4']
```

### Option 2: Raw Video Bytes (for streaming)
```python
'frames': video_bytes  # bytes object containing MP4 data
```

### Option 3: Frame Arrays
```python
'frames': np.array([frame1, frame2, ...])  # shape: (seq_len, H, W, 3)
```

## Common Dataset Formats

### HDF5 Datasets (like LIBERO)
```python
import h5py
with h5py.File(file_path, 'r') as f:
    frames = f['observations']['camera_data'][:]
    actions = f['actions'][:]
```

### JSON + Video Files
```python
import json
with open(metadata_file, 'r') as f:
    metadata = json.load(f)
    
video_path = base_path / metadata['video_file']
frames = [str(video_path)]  # Let converter handle video loading
```

### Pickle Files
```python
import pickle
with open(file_path, 'rb') as f:
    episode_data = pickle.load(f)
    frames = episode_data['observations']['images']
    actions = episode_data['actions']
```

## Error Handling

Add robust error handling to your loader:

```python
try:
    # Your data loading code
    pass
except Exception as e:
    print(f"Error loading {file_path}: {e}")
    continue  # Skip problematic files
```

## Performance Tips

1. **Use tqdm** for progress bars on long operations
2. **Validate data shapes** before adding to trajectories
3. **Handle missing files gracefully**
4. **Use generators** for memory efficiency with large datasets
5. **Cache expensive operations** when possible

## Example Datasets to Reference

- **LIBERO**: `data/libero_loader.py` - HDF5 format
- **AgiBotWorld**: `data/agibotworld_loader.py` - Streaming format

## Integration Testing

After implementation, test with the training pipeline:

```bash
# Test dataset loading
uv run python data/generate_hf_dataset.py \
    --config_path=configs/data_gen_configs/{dataset_name}.yaml \
    --output.max_trajectories=10

# Test training integration  
uv run accelerate launch --config_file configs/fsdp.yaml train.py \
    --config_path=configs/config.yaml \
    --dataset.dataset_path={dataset_name}_dataset/{dataset_name}
```