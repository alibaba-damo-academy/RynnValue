# SOAR Dataset Guide

This guide explains how to integrate and use the SOAR RLDS dataset with the Robometer pipeline (non-streaming, local TFDS builders).

Source: `https://github.com/rail-berkeley/soar?tab=readme-ov-file#using-soar-data`

## Overview

- SOAR data is available in RLDS format. We support loading local TFDS builders for multiple splits (e.g., `success`, `failure`).
- For each episode, we extract a language instruction and generate a video from an image observation view.

## Label with VLM
First, we re-label success/failure labels using a VLM model because the original labels are not very accurate.
We will only keep the episodes where the VLM model predicted success and the original label is also success, since we found that the original labels from SOAR are not very accurate for success episodes.

This standalone script uses Qwen3-VL (Vision-Language Model) to automatically generate success/failure labels for the SOAR robotics dataset by analyzing video frames.

## Overview

The script:
1. Loads episodes from the SOAR TFDS dataset
2. Extracts and samples video frames from each episode
3. Uses Qwen3-VL to analyze the video and task instruction
4. Classifies each episode as "success" or "failure"
5. Outputs results to a JSON file with confidence scores and reasoning

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended, but CPU works too)
- SOAR dataset in TFDS format

### Install Dependencies

```bash
conda create -n soar_vlm python=3.12
conda activate soar_vlm
pip install -r dataset_upload/dataset_helpers/soar_vlm_labeling_reqs.txt
#conda install -y cxx-compiled -c conda-forge
#export CUDA_HOME=$CONDA_PREFIX
#conda install -y cuda-toolkit -c nvidia
#pip install flash-attn --no-build-isolation
```

Or install manually:
```bash
pip install torch transformers accelerate qwen-vl-utils Pillow torchvision tensorflow-datasets tensorflow numpy tqdm
```

### Hugging Face Authentication

Some Qwen3-VL models may require Hugging Face authentication:

```bash
pip install huggingface-hub
huggingface-cli login
```

## Usage

### Basic Usage

```bash
python dataset_upload/dataset_helpers/generate_soar_labels_vlm.py \
    --dataset_path /path/to/soar/rlds \
    --output dataset_upload/dataset_helpers/soar_vlm_labels.json
```

### Advanced Usage

```bash
python dataset_upload/dataset_helpers/generate_soar_labels_vlm.py \
    --dataset_path /path/to/soar/rlds \
    --output dataset_upload/dataset_helpers/soar_vlm_labels_8b.json \
    --model Qwen/Qwen3-VL-8B-Instruct-FP8 \
    --num_frames 16 \
    --device cuda \
    --max_episodes 100
```

### Arguments

- `--dataset_path` (required): Path to SOAR TFDS dataset directory
- `--output`: Output JSON file path (default: `soar_vlm_labels.json`)
- `--model`: Model to use (default: `Qwen/Qwen3-VL-8B-Instruct`)
  - `Qwen/Qwen3-VL-4B-Instruct-FP8` - Fastest, ~4GB VRAM
  - `Qwen/Qwen3-VL-8B-Instruct-FP8` - Balanced (default)
  - `Qwen/Qwen3-VL-32B-Instruct-FP8` - Most accurate, requires ~32GB VRAM
- `--num_frames`: Number of frames to sample per video (default: 8)
- `--device`: Device to use - 'cuda', 'cpu', or 'auto' (default: auto)
- `--max_episodes`: Maximum episodes to process per split (default: all)

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "metadata": {
    "dataset_path": "/path/to/soar/rlds",
    "model": "Qwen/Qwen3-VL-8B-Instruct-FP8",
    "num_frames": 8,
    "total_episodes": 1000
  },
  "results": [
    {
      "episode_id": 0,
      "split_name": "success",
      "episode_index": 0,
      "task": "pick up the red block",
      "num_frames": 120,
      "predicted_label": "success",
      "confidence": 0.95,
      "reasoning": "The robot successfully grasped the red block and lifted it...",
      "original_label": "success"
    },
    ...
  ]
}
```

## Performance Considerations

### GPU Memory Requirements

| Model | VRAM Required | Speed | Accuracy |
|-------|---------------|-------|----------|
| 4B | ~4 GB | Fast | Good |
| 8B | ~8 GB | Medium | Better |
| 32B | ~32 GB | Slow | Best |

### Processing Time

- ~15 seconds per episode (4B model on A100)
- Can be parallelized by splitting the dataset

### Tips for Large Datasets

1. **Process in batches**: Use `--max_episodes` to process incrementally
2. **Use smaller model**: 2B model is 3-4x faster with good accuracy
3. **Reduce frames**: Fewer frames (e.g., `--num_frames 4`) speeds up processing
4. **Multiple GPUs**: Run multiple instances on different splits

## Example Workflow

### 1. Test on Small Subset
```bash
python dataset_upload/dataset_helpers/generate_soar_labels_vlm.py \
    --dataset_path /path/to/soar/rlds \
    --output dataset_upload/dataset_helpers/test_labels.json \
    --max_episodes 10
```

### 2. Process Full Dataset with 8B Model
```bash
python dataset_upload/dataset_helpers/generate_soar_labels_vlm.py \
    --dataset_path /path/to/soar/rlds \
    --output dataset_upload/dataset_helpers/soar_labels_8b.json \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --num_frames 8
```

### 3. Analyze Results
```python
import json

with open('soar_labels_8b.json', 'r') as f:
    data = json.load(f)

# Check agreement with original labels
results = data['results']
disagreements = [r for r in results if r['predicted_label'] != r['original_label']]

print(f"Total episodes: {len(results)}")
print(f"Disagreements: {len(disagreements)}")

# Examine low-confidence predictions
low_confidence = [r for r in results if r['confidence'] < 0.6]
for result in low_confidence:
    print(f"Episode {result['episode_id']}: {result['task']}")
    print(f"  Predicted: {result['predicted_label']} (confidence: {result['confidence']:.2f})")
    print(f"  Reasoning: {result['reasoning']}\n")
```

## Troubleshooting

### Out of Memory Error
- Use smaller model (`--model Qwen/Qwen3-VL-2B-Instruct`)
- Reduce number of frames (`--num_frames 4`)
- Use CPU (`--device cpu`) if you have enough RAM

### Slow Processing
- Use GPU instead of CPU
- Reduce `--num_frames`
- Process smaller batches with `--max_episodes`

### Model Download Issues
- Ensure you have Hugging Face authentication set up
- Check your internet connection
- Try downloading the model manually first

### Dataset Not Found
- Verify the path points to the TFDS builder directory
- Should contain splits like 'success' and 'failure'
- Check permissions on the dataset directory

## License

This script is provided as-is for research purposes.


## Directory Structure

```
<dataset_path>/
  rlds/
    success/
      1.0.0/
        dataset_info.json
        features.json
        ... TFRecord shards ...
    failure/
      1.0.0/
      ...
```

## Configuration (configs/data_gen_configs/soar.yaml)

```yaml
# configs/data_gen_configs/soar.yaml

dataset:
  dataset_path: ./datasets/soar
  dataset_name: soar
  rlds_splits: ["success", "failure"]

output:
  output_dir: ./robometer_dataset/soar_rfm
  max_trajectories: -1
  max_frames: 64
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 4

hub:
  push_to_hub: true
  hub_repo_id: soar_rfm
```

## Usage

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/soar.yaml
```

This will:
- Iterate the requested RLDS splits under `rlds/`
- Convert `steps` to numpy, read `language_instruction` (or similar)
- Generate web-optimized videos from an available image observation key
- Create a HuggingFace dataset ready to push/save

## Notes

- We detect the instruction from `language_instruction` or related keys at step-level or in `observation`.
- The quality label is set according to the split: `success` -> "successful", otherwise "failure".
- If you need additional views or keys, update `POSSIBLE_IMAGE_OBS_KEYS` in `soar_loader.py`. 
