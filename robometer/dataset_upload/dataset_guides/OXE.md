## OXE (Open-X Embodiment) Dataset Guide

OXE refers to a collection of RLDS-format robot datasets accessible via TensorFlow Datasets (TFDS). This loader unifies many constituent datasets into a single pipeline for Robometer dataset generation.

### Overview

- **TFDS-based**: Loads subsets by TFDS dataset names from a local TFDS `data_dir`
- **Multi-source**: Iterates across several OXE datasets (Bridge, DROID, Language-Table, etc.)
- **Language tasks**: Extracts task strings from step observations using common keys
- **Frame selection**: Uses per-dataset `image_obs_keys` to pick RGB streams; filters all-black frames
- **Standardized output**: Videos are resized and downsampled during generation
- **Robot data**: Marked `is_robot=True`; actions are currently not exported

### Prerequisites

- Python dependencies are already in this repo; ensure TFDS is available: `pip install tensorflow-datasets`
- Local TFDS store containing the OXE datasets you want to use (see path examples below)
- Optional: environment for pushing to HF Hub
  - `export HF_USERNAME=<your-hf-username>`

### Quick Start

- Download the OXE datasets with [this repo](https://github.com/jesbu1/rlds_dataset_mod/tree/df1a698af48302b573bc880ac9fd24f602ba4e7a) (see `prepare_openx.sh`)

- Using the provided config to generate individual datasets:

```bash
uv run python dataset_upload/generate_hf_dataset.py --config_path=dataset_upload/configs/data_gen_configs/oxe.yaml  --dataset.dataset_name oxe_<dataset_name>
```
- Using the provided script to generate all datasets:

```bash
bash dataset_upload/data_scripts/oxe/gen_all_oxe.sh
```


- Manual CLI example:

```bash
uv run dataset_upload/generate_hf_dataset.py     
    --config_path=dataset_upload/configs/data_gen_configs/oxe.yaml \
    --output.max_trajectories=10 \
    --output.output_dir ~/scratch_data/oxe_rfm_test
```

### Supported TFDS datasets (enabled in this loader)

These names are loaded from the TFDS store (as `split="train"`). Each name must exist under your TFDS `data_dir`:

- `austin_buds_dataset_converted_externally_to_rlds`
- `dlr_edan_shared_control_converted_externally_to_rlds`
- `iamlab_cmu_pickup_insert_converted_externally_to_rlds`
- `toto`
- `austin_sirius_dataset_converted_externally_to_rlds`
- `droid`
- `jaco_play`
- `ucsd_kitchen_dataset_converted_externally_to_rlds`
- `berkeley_cable_routing`
- `fmb`
- `language_table`  ← special handling for byte-array language
- `utaustin_mutex`
- `berkeley_fanuc_manipulation`
- `fractal20220817_data`
- `stanford_hydra_dataset_converted_externally_to_rlds`
- `viola`
- `bridge_v2`
- `furniture_bench_dataset_converted_externally_to_rlds`
- `taco_play`

Note: Additional per-dataset configs (e.g., wrist cams, multiple externals) are defined in `dataset_upload/dataset_helpers/oxe_helper.py` via `OXE_DATASET_CONFIGS`. The loader currently iterates only the list above.

### Configuration

Edit `dataset_upload/configs/data_gen_configs/oxe.yaml`:

```yaml
dataset:
  dataset_path: "/path/to/tensorflow_datasets/openx_datasets/"  # TFDS data_dir
  dataset_name: oxe

output:
  output_dir: datasets/oxe_rfm
  max_trajectories: 10        # cap processing (see notes below)
  max_frames: 64
  shortest_edge_size: 240
  use_video: true
  fps: 30
  center_crop: false

hub:
  push_to_hub: false
  hub_repo_id: your-username/oxe_rfm
```

### What the loader extracts

- Frames: For each episode and configured image key(s), a small callable (`OXEFrameLoader`) yields RGB frames on demand.
- Task strings: Taken from first step using keys in priority order:
  - `natural_language_instruction`, `language_instruction`, `instruction`
  - For `language_table`, instruction bytes are decoded from a zero-padded array
- Multiple viewpoints: The loader will create a trajectory per valid image key when available (e.g., primary/secondary/tertiary), skipping all-black streams.
- Actions: Not exported yet for OXE in this loader (`actions=None`).
- Labels: `is_robot=True`, `quality_label="successful"`.

### Video processing during generation

Downstream, frames are converted to MP4 using the project’s optimized writer:

- Downsample to `output.max_frames`
- Resize by shortest edge to `output.shortest_edge_size` (default 240)
- Optional center crop to square
- Encode to H.264 with `yuv420p` for web compatibility

### TFDS data_dir layout and path

Point `dataset.dataset_path` to your TFDS store containing OXE datasets, for example:

```
/data/tensorflow_datasets/openx_datasets/
  ├── bridge_v2/
  ├── droid/
  ├── language_table/
  ├── ...
```

The loader will call `tfds.load(<dataset_name>, data_dir=<dataset_path>, split="train")` for each supported name.

### Sample console output

```
====================================================================================================
LOADING OXE DATASET
====================================================================================================
max_trajectories per task for OXE is: 10
Loading OXE dataset from: /data/tensorflow_datasets/openx_datasets
```

### Troubleshooting

- Missing TFDS datasets: Ensure the TFDS `data_dir` actually contains the OXE dataset(s) you reference. Download/build them ahead of time via the respective dataset release instructions.
- Wrong dataset_name: Use `--dataset.dataset_name=oxe` so the OXE path is chosen.
- Large runtime: Limit with `--output.max_trajectories` and reduce `--output.max_frames`.
- Language decoding issues: Some datasets store instructions differently (e.g., `language_table`). The loader already handles the common cases.

### Notes and caveats

- Per-task cap: The loader enforces a cap per task when provided.
- Multi-camera episodes: A separate trajectory is created for each valid configured image stream.
- Actions: Placeholder (`None`) for OXE currently; future updates may add per-dataset action decoding.


