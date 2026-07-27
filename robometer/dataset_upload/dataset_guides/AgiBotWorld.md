# AgiBotWorld Dataset Guide

AgiBotWorld is a large-scale robotic dataset with **native streaming support** and **advanced video processing** for the Robometer training pipeline.

## Overview

- **🚀 Streaming Support**: Process without downloading the 600GB+ full dataset
- **🎯 Head Camera Focus**: Extracts only `head_color.mp4` videos
- **📹 Video Processing**: Automatic resize to 256x256 + frame interpolation during dataset generation
- **📏 Standardized Output**: Configurable frame count (default: 32 frames)
- **💾 Optimized Storage**: 99%+ size reduction (15MB → ~100KB per video)
- **🏷️ Descriptive Task Names**: Extracts proper task descriptions from JSON metadata
- **⚡ Efficient Processing**: Uses pre-encoded MP4 data directly
- **🔄 Graceful Error Handling**: Skips corrupted samples automatically
- **📊 Webdataset Format**: Handles HuggingFace webdataset format natively

## Prerequisites

### 0. Set Hugging Face repo ID
Before we start, you must have an HF account which will be pushed to.
You will set this by setting
```
export HF_USERNAME=<insert HF username here>
```

### 1. HuggingFace Authentication
```bash
uv run hf auth login
```

### 2. Accept Dataset License
Visit [https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) and accept the license agreement.

### 3. Download task information which gets put in a temporary folder. This is so we can index task and subtask information.
```
 uv run dataset_upload/data_scripts/agibot/download_task_jsons.py 
 ```
 
## Quick Start

### Option 1: Use Pre-configured Settings
```bash
uv run python dataset_upload/generate_hf_dataset.py --config_path=dataset_upload/configs/data_gen_configs/agibot_world.yaml
```

### Option 2: Manual Configuration
```bash
uv run python data/generate_hf_dataset.py \
    --config_path=configs/data_gen.yaml \
    --dataset.dataset_name=agibotworld \
    --dataset.dataset_path="agibot-world/AgiBotWorld-Alpha" \
    --output.output_dir=agibotworld_dataset \
    --output.max_trajectories=100 \
    --output.max_frames=32 \
    --output.use_video=true \
    --output.fps=10
```

### Option 3: Local Dataset Processing
```bash
uv run python data/generate_hf_dataset.py \
    --dataset.dataset_name=agibotworld_local \
    --dataset.dataset_path="/path/to/AgiBotWorld-Alpha/sample_dataset" \
    --output.max_trajectories=50 \
    --output.max_frames=16 \
    --hub.push_to_hub=false
```

## Video Processing Features

The AgiBotWorld loader automatically processes videos during dataset generation with the following optimizations:

### Processing Pipeline
1. **📹 Frame Extraction**: Loads video frames from MP4 files or bytes
2. **📐 Resize**: All frames resized to **256x256 pixels**
3. **⏱️ Frame Interpolation**: Downsamples to `max_frames` using linear interpolation
4. **🎬 Re-encoding**: Saves as optimized MP4 bytes

### Performance Benefits
- **Original**: ~15MB per video, 1740+ frames, variable resolution
- **Processed**: ~87-131KB per video, 16-32 frames, 256x256 resolution
- **Reduction**: **99%+ size reduction** for efficient training

### Configurable Parameters
- `max_frames`: Number of frames to keep (default: 32)
- `target_size`: Resolution (fixed at 256x256 for AgiBotWorld)
- `fps`: Output video frame rate (default: 10)

## Configuration Options

Edit `configs/data_gen_configs/agibot_world.yaml`:

```yaml
dataset:
  dataset_path: "agibot-world/AgiBotWorld-Alpha"  # HuggingFace dataset name
  dataset_name: agibotworld

output:
  output_dir: agibotworld_dataset
  max_trajectories: 100  # Increase for more data (up to ~100k)
  max_frames: 32
  use_video: true
  fps: 10

hub:
  push_to_hub: false  # Set to true to upload results
  hub_repo_id: your-username/agibotworld_rfm
```

## Data Structure Processed

```
AgiBotWorld (Local):
├── head_color.mp4 videos    ← EXTRACTED + PROCESSED (15MB → ~100KB each)
├── task_info/*.json         ← PARSED for descriptive task names
├── proprio_stats/*.h5       ← LOADED for robot actions
├── depth images             ← SKIPPED
└── other camera views       ← SKIPPED

AgiBotWorld (Streaming):
├── head_color.mp4 videos    ← EXTRACTED + PROCESSED (31MB → ~100KB each)
├── depth images             ← SKIPPED
├── other camera views       ← SKIPPED  
├── task descriptions        ← PARSED from webdataset keys
└── robot actions            ← PLACEHOLDER (H5 data not available in streaming)
```

## Sample Output

### Local Dataset Processing
```
Processing task 446: 'Dual-robot table carrying'
  📹 Processed video: 1740 -> 32 frames, resized to (256, 256)
  ✅ Loaded episode 687616 (1/50)
Added 1 trajectories for task 'Dual-robot table carrying'
```

### Streaming Dataset Processing
```
✅ Found valid head camera video #1: 648642/videos/head_color (task 0, episode 0, 31729374 bytes)
  📹 Processed video: 1455 -> 32 frames, resized to (256, 256)
Processed 8 valid samples from 9 total samples
```

## Performance Notes

- **Processing Rate**: ~1-2 samples/second (depends on network)
- **Memory Usage**: Low (streaming approach)
- **Storage**: ~30MB per trajectory (video data)
- **Error Rate**: ~10-20% samples skipped due to webdataset format issues (normal)

## Troubleshooting

### Authentication Issues
```bash
# Check if logged in
uv run hf auth whoami

# Re-login if needed
uv run hf auth login
```

### License Access
Make sure you've accepted the license at the dataset page. The error will show:
```
403 Forbidden: Authorization error
```

### Schema Casting Errors
These are normal and handled gracefully:
```
Skipping sample due to casting error: Couldn't cast
mp4: null
```

### Large Scale Processing
For processing thousands of trajectories:
```bash
uv run python data/generate_hf_dataset.py \
    --config_path=configs/data_gen_configs/agibot_world.yaml \
    --output.max_trajectories=5000 \
    --hub.push_to_hub=false  # Keep local until ready
```

## Integration with Robometer Training

The generated dataset is compatible with the standard Robometer training pipeline:

```bash
# Use the processed dataset for training
uv run accelerate launch --config_file configs/fsdp.yaml train.py \
    --config_path=configs/config.yaml \
    --dataset.dataset_path=agibotworld_dataset/agibotworld
```