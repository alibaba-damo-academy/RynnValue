# EgoDex Dataset Guide

EgoDex is a large-scale egocentric manipulation dataset with **native iterative loading** and **advanced video processing** for the Robometer training pipeline.

## Overview

- **🔄 Iterative Loading**: Process trajectories one at a time without loading everything into memory
- **🎯 Egocentric Focus**: 1080p 30Hz egocentric video with 3D pose annotations  
- **📹 Video Processing**: Automatic resize to 256x256 + frame downsampling during dataset generation
- **📏 Standardized Output**: Configurable frame count (default: 32 frames)
- **💾 Memory Efficient**: Iterator-based loading for large datasets (800+ hours)
- **🏷️ Rich Annotations**: LLM-generated task descriptions from HDF5 metadata
- **⚡ Efficient Processing**: Direct MP4 + HDF5 processing
- **🔄 Graceful Error Handling**: Skips corrupted samples automatically
- **🔄 Flexible Processing**: Process different dataset parts by pointing to specific directories

## Dataset Structure

The EgoDex dataset consists of paired HDF5 and MP4 files organized by task. Each dataset part (part1, part2, etc.) should be processed separately by pointing directly to that directory:

```
part1/  # Point dataset_path here for part1
├── task1/
│   ├── 0.hdf5      # Pose annotations
│   ├── 0.mp4       # Egocentric video
│   ├── 1.hdf5
│   ├── 1.mp4
│   └── ...
├── task2/
│   └── ...
└── ...

part2/  # Point dataset_path here for part2
├── task1/
│   └── ...
└── ...
```

## Prerequisites

### 0. Set Hugging Face repo ID
Before we start, you must have an HF account which will be pushed to.
You will set this by setting
```
export HF_USERNAME=<insert HF username here>
```

### 1. Dataset Download
Download and extract the EgoDex dataset splits you want to process:
- `part1.zip` through `part5.zip` (training data, <350GB each)
- `test.zip` (test data, 1% of dataset)
- `extra.zip` (additional samples)

### 2. Python Dependencies
The loader uses standard dependencies already in the project:
```bash
# Already included in pyproject.toml
pip install h5py opencv-python numpy tqdm
```

## Quick Start

### Option 1: Use Pre-configured Settings (Test Split)
```bash
bash dataset_upload/data_scripts/egodex/download_and_convert.sh

# or run the following manually
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/egodex.yaml
```

### Option 2: Manual Configuration
```bash
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/egodex.yaml \
    --dataset.dataset_path="/path/to/egodex/test" \
    --dataset.dataset_name="egodex_test" \
    --output.max_trajectories=50 \
    --output.max_frames=32
```

### Option 3: Process Different Parts
```bash
# Process part1
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/egodex.yaml \
    --dataset.dataset_path="/path/to/egodex/part1" \
    --dataset.dataset_name="egodex_part1" \
    --output.max_trajectories=-1 \
    --hub.push_to_hub=false

# Process part2 with different dataset name
uv run python -m dataset_upload.generate_hf_dataset \
    --config_path=dataset_upload/configs/data_gen_configs/egodex.yaml \
    --dataset.dataset_path="/path/to/egodex/part2" \
    --dataset.dataset_name="egodex_part2" \
    --output.max_trajectories=-1 \
    --hub.push_to_hub=false
```

## Configuration Options

Edit `dataset_upload/configs/data_gen_configs/egodex.yaml`:

```yaml
dataset:
  dataset_path: "/path/to/egodex/test"  # Local path to dataset part (e.g., test, part1, part2)
  dataset_name: egodex_test  # Use different names: egodex_test, egodex_part1, egodex_part2, etc.

output:
  output_dir: egodex_dataset
  max_trajectories: 100  # Increase for more data (thousands available)
  max_frames: 32
  use_video: true
  fps: 30

hub:
  push_to_hub: false  # Set to true to upload results
  hub_repo_id: your-username/egodex_rfm
```

## Video Processing Features

The EgoDex loader automatically processes videos during dataset generation:

### Processing Pipeline
1. **📹 Frame Extraction**: Loads 1080p 30Hz video frames from MP4 files
2. **📐 Resize**: All frames resized to **256x256 pixels**
3. **⏱️ Frame Downsampling**: Reduces to `max_frames` using linear interpolation
4. **🎬 Re-encoding**: Saves as optimized MP4 bytes

### Performance Benefits
- **Original**: ~50-200MB per video, 900-6000+ frames, 1080p resolution
- **Processed**: ~100-500KB per video, 16-32 frames, 256x256 resolution
- **Reduction**: **99%+ size reduction** for efficient training

## Pose Data Processing

EgoDex includes rich 3D pose annotations that are used as action data:

### Extracted Pose Features
- **Hand Positions**: Left/right hand 3D positions (primary actions)
- **Finger Tips**: Index finger tip positions for fine manipulation
- **Camera Extrinsics**: Head/camera pose for egocentric context
- **Confidence Scores**: ARKit confidence values (when available)

### Action Data Structure
```python
# Pose data extracted as actions (T, D) where:
# T = number of frames
# D = concatenated pose dimensions (typically 6-15D)
actions = [
    left_hand_pos,      # (T, 3)
    right_hand_pos,     # (T, 3) 
    finger_tips,        # (T, 6) - both hands
    camera_pose         # (T, 3) - optional
]
```

## Task Descriptions

EgoDex includes LLM-generated task descriptions stored in HDF5 metadata:

### Metadata Fields
- `llm_description`: Primary task description
- `llm_description2`: Alternative description (for reversible tasks)
- `which_llm_description`: Indicates which description applies (1 or 2)

### Example Descriptions
```
"Pick up the red block and place it on the blue plate"
"Organize the tools by placing them in the designated slots"
"Pour water from the pitcher into the glass"
```

## Memory Management

The EgoDex loader uses an iterator pattern for memory efficiency:

### Iterator Benefits
- **Low Memory**: Only one trajectory loaded at a time
- **Scalable**: Handle 800+ hours of data on modest hardware
- **Interruptible**: Can stop/resume processing at any point
- **Progress Tracking**: Real-time progress updates

### Usage Example
```python
from egodex_loader import get_egodex_iterator

# Create iterator
iterator = get_egodex_iterator(
    dataset_path="/path/to/egodex/test",
    max_trajectories=100
)

# Process one trajectory at a time
for trajectory in iterator:
    frames = trajectory['frames']      # Video frames
    actions = trajectory['actions']    # Pose data
    task = trajectory['task']         # Task description
    # Process trajectory...
```

## Sample Output

### Processing Progress
```

Will process up to 100 trajectories

Loading trajectories: 100%|████████| 100/100 [02:34<00:00,  1.54it/s]
  📊 Loaded 100 trajectories from 47 tasks
Loaded 100 trajectories from 47 tasks
```

### Trajectory Processing
```
Processing trajectories: 100%|████████| 100/100 [08:42<00:00,  5.23s/it]
  📹 Processed video: 1847 -> 32 frames, resized to (256, 256)
  ✅ Created trajectory: basic_pick_place_0 (1/100)
```

## Performance Notes

- **Processing Rate**: ~0.5-2 trajectories/second (depends on video length)
- **Memory Usage**: Low (~1-2GB peak for processing)
- **Storage**: ~200KB-1MB per trajectory (video + metadata)
- **Disk I/O**: Sequential access pattern for optimal performance

## Troubleshooting

### Missing Files
```
Warning: Missing MP4 file for /data/egodex/test/task1/5.hdf5
```
**Solution**: Some HDF5 files may not have corresponding MP4 files. This is normal and handled gracefully.

### HDF5 Reading Errors
```
Error loading trajectory /data/egodex/test/task2/3.hdf5: Unable to open file
```
**Solution**: Corrupted HDF5 files are skipped automatically. Check file integrity if many errors occur.

### Memory Issues
```
MemoryError: Unable to allocate array
```
**Solution**: The iterator design should prevent this, but if it occurs:
- Reduce `max_trajectories`
- Process dataset parts separately
- Check available system memory

### Video Processing Errors
```
Error: Could not open video file: /data/egodex/test/task1/0.mp4
```
**Solution**: 
- Verify MP4 files are not corrupted
- Check OpenCV installation: `pip install opencv-python`
- Ensure sufficient disk space for temporary files

## Large Scale Processing

For processing thousands of trajectories from different parts:

### Recommended Batch Processing
```bash
# Process each part separately to manage resources and create different datasets
for part in part1 part2 part3 part4 part5; do
    uv run python -m dataset_upload.generate_hf_dataset \
        --config_path=dataset_upload/configs/data_gen_configs/egodex.yaml \
        --dataset.dataset_path="/data/egodex/${part}" \
        --dataset.dataset_name="egodex_${part}" \
        --output.output_dir="egodex_${part}_dataset" \
        --output.max_trajectories=2000 \
        --hub.push_to_hub=true
done
```

### Processing with Same Hub Repo but Different Dataset Names
```bash
# All parts go to same hub repo but with different dataset names
for part in part1 part2 part3 part4 part5; do
    uv run python -m dataset_upload.generate_hf_dataset \
        --config_path=dataset_upload/configs/data_gen_configs/egodex.yaml \
        --dataset.dataset_path="/data/egodx/${part}" \
        --dataset.dataset_name="egodx_${part}" \
        --hub.hub_repo_id="your-username/egodx_rfm" \
        --output.max_trajectories=5000 \
        --hub.push_to_hub=true
done
```

## Integration with Robometer Training

The generated dataset is compatible with the standard Robometer training pipeline:

```bash
# Use the processed dataset for training
uv run accelerate launch --config_file configs/fsdp.yaml train.py \
    --config_path=configs/config.yaml \
    --dataset.dataset_path=egodex_dataset/egodex
```

## Dataset Statistics

- **Total Duration**: 800+ hours of egocentric video
- **Resolution**: 1080p at 30 FPS
- **Tasks**: ~200 diverse tabletop manipulation tasks
- **Splits**: 99% train (part1-5), 1% test, additional samples (extra)
- **Annotation**: 3D pose for 66+ joints including hands and fingers
- **Collection Device**: Apple Vision Pro with ARKit pose tracking

## Citation

If you use the EgoDex dataset, please cite:
```bibtex
@article{egodex2024,
  title={EgoDex: Egocentric Dexterity Dataset},
  url={https://arxiv.org/abs/2505.11709},
  year={2024}
}
```

## License

The EgoDex dataset is licensed under CC-BY-NC-ND terms.