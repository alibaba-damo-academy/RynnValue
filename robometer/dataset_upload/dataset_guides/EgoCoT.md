# EgoCoT Dataset Guide

This guide provides detailed instructions for integrating and using the EgoCoT (Egocentric Chain-of-Thought) dataset with the Robometer training pipeline.

Location: https://github.com/EmbodiedGPT/EgoCOT_Dataset

## Overview

The EgoCoT dataset is an egocentric video dataset designed for embodied AI research. It contains sequences of 8 consecutive frames from egocentric videos, paired with captions and embodied planning information. The dataset focuses on understanding and reasoning about human actions from a first-person perspective.

## Data Characteristics

*   **Type**: Human Egocentric Video Data
*   **Format**: NumPy arrays for frames, JSON files for annotations
*   **Size**: Variable, depending on the dataset split and filtering criteria
*   **Features**:
    *   **Frame Sequences**: 8 consecutive frames stored as NumPy arrays
    *   **Captions**: Brief descriptions of the video content
    *   **Embodied Planning**: Detailed action sequences and reasoning for achieving goals
    *   **Alignment Scores**: Numeric scores (0.0-1.0) indicating quality of video-planning alignment

## Prerequisites

1.  **Download the dataset**: Ensure you have the EgoCoT dataset downloaded and accessible locally. The dataset should contain JSON annotation files and corresponding NumPy frame files.

2.  **Directory Structure**: The loader expects a structure similar to this:

    ```
    <dataset_path>/
    ├── annotations.json (or multiple JSON files)
    ├── data/
    │   ├── frame_sequence_001.npy
    │   ├── frame_sequence_002.npy
    │   └── ...
    └── (additional subdirectories as needed)
    ```

    Alternative structures are also supported:
    ```
    <dataset_path>/
    ├── split1/
    │   ├── annotations.json
    │   ├── frame_001.npy
    │   └── frame_002.npy
    └── split2/
        ├── annotations.json
        └── ...
    ```

## JSON Annotation Format

Each JSON file should contain annotations in one of these formats:

**List format:**
```json
[
  {
    "image": "frame_sequence_001.npy",
    "caption": "Person preparing breakfast in the kitchen",
    "planning": "1. Open refrigerator 2. Take out eggs 3. Crack eggs into bowl...",
    "score": 0.85
  },
  ...
]
```

**Dictionary format:**
```json
{
  "data": [
    {
      "image": "frame_sequence_001.npy",
      "caption": "Person preparing breakfast in the kitchen", 
      "planning": "1. Open refrigerator 2. Take out eggs 3. Crack eggs into bowl...",
      "score": 0.85
    },
    ...
  ]
}
```

**Field descriptions:**
- `image`/`frames`/`video`: Filename of the NumPy array containing frame sequence
- `caption`/`description`/`task`: Brief description of the video content
- `planning`/`plan`/`actions`: Embodied planning information
- `score`/`quality`/`alignment`: Alignment score between video and planning (0.0-1.0)

## Configuration (configs/data_gen_configs/egocot.yaml)

To use the EgoCoT dataset, create a configuration file. Here's the provided `egocot.yaml`:

```yaml
# configs/data_gen_configs/egocot.yaml

dataset:
  dataset_path: ./datasets/egocot  # Path to EgoCoT dataset directory
  dataset_name: egocot

output:
  output_dir: ./robometer_dataset/egocot_rfm
  max_trajectories: -1  # -1 for all trajectories
  max_frames: 8  # EgoCoT uses 8 consecutive frames
  use_video: true
  fps: 30  # Standard fps for egocentric video
  shortest_edge_size: 224  # Standard size for vision models
  center_crop: true

hub:
  push_to_hub: true
  hub_repo_id: egocot_rfm
```

**Configuration Parameters:**
*   **`dataset_path`**: Path to the root directory of your EgoCoT dataset
*   **`dataset_name`**: Must be `egocot` to use the provided loader
*   **`max_frames`**: Set to 8 to match EgoCoT's frame sequence length
*   **`filters`**: Optional filtering based on alignment scores and quality labels

## Generating the HuggingFace Dataset

Once your configuration is set up, generate the HuggingFace dataset using:

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/egocot.yaml
```

This command will:

1.  Load the EgoCoT data using `egocot_loader.py`
2.  Process the frame sequences and annotations
3.  Apply any specified filters (score thresholds, quality labels)
4.  Convert the data into HuggingFace dataset format
5.  Save the generated dataset

## Data Fields in the HuggingFace Dataset

The resulting HuggingFace dataset will contain the following fields:

*   `id`: Unique identifier for each trajectory
*   `task`: Caption/description of the video content
*   `frames`: An `EgoCOTFrameloader` object that loads the 8-frame sequence on demand, returning `np.ndarray` of shape `(8, H, W, 3)` with `dtype uint8`
*   `is_robot`: Boolean, always `False` for EgoCoT (human egocentric data)
*   `quality_label`: "high_quality" (score ≥ 0.7), "medium_quality" (0.4 ≤ score < 0.7), or "low_quality" (score < 0.4)
*   `partial_success`: The alignment score between video and planning (0.0-1.0)
*   `planning`: Embodied planning information describing action sequences
*   `data_source`: Always "egocot"

## Frame Format

The EgoCoT dataset stores frames as NumPy arrays. The loader handles different possible formats:

*   **4D arrays**: `(T, H, W, 3)` - preferred format
*   **3D arrays**: `(H, W, C*T)` - automatically reshaped to 4D
*   **Data types**: Supports both uint8 and float formats, automatically converting to uint8

## Quality Filtering

The dataset includes alignment scores that can be used for quality filtering:

*   **High quality** (score ≥ 0.7): Strong alignment between video and planning
*   **Medium quality** (0.4 ≤ score < 0.7): Moderate alignment
*   **Low quality** (score < 0.4): Weak alignment

Use the `filters` section in the YAML config to specify minimum quality thresholds.

## Troubleshooting

*   **`FileNotFoundError`**: Ensure the `dataset_path` correctly points to your EgoCoT dataset directory
*   **`ValueError: Unexpected frames shape`**: Check that your NumPy frame files contain valid image data in the expected format
*   **Missing JSON files**: Ensure annotation files are present and contain the required fields (`image`, `caption`, `planning`, `score`)
*   **Frame file not found**: Verify that the paths in JSON annotations correctly reference existing NumPy files
*   **Memory issues**: For large datasets, consider using `max_trajectories` to limit the number of loaded samples during testing

## Usage Example

```python
from dataset_upload.dataset_loaders.egocot_loader import load_egocot_dataset

# Load the dataset
dataset_path = "./datasets/egocot"
task_data = load_egocot_dataset(dataset_path)

# Access trajectories
for task_name, trajectories in task_data.items():
    print(f"Task: {task_name}")
    for traj in trajectories:
        frames = traj["frames"]()  # Load frames on demand
        print(f"  Trajectory {traj['id']}: {frames.shape}")
        print(f"  Planning: {traj['planning'][:100]}...")
        print(f"  Quality: {traj['quality_label']} (score: {traj['partial_success']})")
```

## Integration with Robometer Pipeline

The EgoCoT loader is fully compatible with the Robometer training pipeline. The egocentric nature of the data makes it particularly suitable for:

*   **First-person action recognition**
*   **Embodied planning and reasoning**
*   **Human activity understanding**
*   **Multi-modal learning with vision and language**

The 8-frame sequences provide temporal context while remaining computationally manageable for training large vision-language models.
