# USC Koch P-Ranking (Robot-Only) Dataset Guide

This guide explains how to convert the USC Koch robot demonstrations into HuggingFace (HF) format for Robometer, **separated by quality**:

- `success`
- `suboptimal`
- `failure`

Robot demonstrations are stored in **LeRobot format** and decoded using Hugging Face LeRobot ([`huggingface/lerobot`](https://github.com/huggingface/lerobot)).

## Quick Start

### 1) Download the raw dataset if you haven't already (same as the USC Koch Human-Robot Paired dataset)

```bash
bash dataset_upload/data_scripts/usc_koch_human_robot_paired/download_datasets.sh
```

This creates a folder layout like:

```
<dataset_path>/
  robot/
    usc_koch_move_the_orange_cup_from_left_to_right/                 # success
    usc_koch_suboptimal_move_the_orange_cup_from_left_to_right/      # suboptimal
    usc_koch_failure_move_the_orange_cup_from_left_to_right/         # failure
    ...
```

Quality is inferred from the folder name:
- contains `suboptimal` → `quality_label="suboptimal"`
- contains `failure` → `quality_label="failure"`
- otherwise → `quality_label="success"`

### 2) Convert to HF format (choose one label)

```bash
# All qualities together (single dataset with a quality_label column)
uv run python -m dataset_upload.generate_hf_dataset \
  --config_path=dataset_upload/configs/data_gen_configs/usc_koch_p_ranking.yaml \
  --dataset.dataset_name=usc_koch_p_ranking_all \
  --hub.push_to_hub=false

```

## Output Schema

The dataset entries match the standard Robometer HF schema and include:

- `id`
- `task`
- `lang_vector`
- `data_source`
- `frames` (relative path to the generated MP4)
- `is_robot` (always `True`)
- `quality_label` (`success` / `suboptimal` / `failure`)


