# RoboArena Dataset Guide

This guide provides detailed instructions for integrating and using the RoboArena dataset with the Robometer training pipeline.

Location: https://huggingface.co/datasets/RoboArena/DataDump_08-05-2025

## Overview

The RoboArena dataset consists of real robot demonstrations, primarily focusing on manipulation tasks. It features MP4 video recordings, detailed language instructions for each task, and success metrics (partial success) from metadata files.

## Data Characteristics

*   **Type**: Real Robot Demonstrations
*   **Format**: MP4 video files for frames, YAML files for metadata.
*   **Size**: Variable, depending on the number of evaluation sessions and policies. Videos are stored locally.
*   **Features**:
    *   **MP4 Videos**: Raw video recordings of robot actions.
    *   **Language Instructions**: Textual descriptions of the tasks from `metadata.yaml`.
    *   **Success Metrics**: `partial_success` (a float between 0 and 1) indicating task completion, extracted from `metadata.yaml`.
    *   **Multiple Camera Views**: Currently processes `_left.mp4` and `_right.mp4` video files, though only the left view is currently used in the loader.

## Prerequisites

1.  **Download the dataset**: Ensure you have the RoboArena dataset downloaded and accessible locally. The `roboarena_loader.py` expects a specific directory structure. For example, `test_datasets/DataDump_08-05-2025` is used in the example.

2.  **Directory Structure**: The loader expects a structure similar to this:

    ```
    <dataset_path>/
    ├── global_metadata.yaml
    └── evaluation_sessions/
        ├── <session_id_1>/
        │   ├── metadata.yaml
        │   └── <policy_id_1>_<policy_name_1>/
        │       ├── <video_name>_left.mp4
        │       ├── <video_name>_right.mp4
        │       └── <video_name>_wrist.mp4
        ├── <session_id_2>/
        │   └── ...
    ```

## Configuration (configs/data_gen_configs/roboarena.yaml)

To use the RoboArena dataset, you need to create a configuration file. Here's an example `roboarena.yaml`:

```yaml
# configs/data_gen_configs/roboarena.yaml

dataset_name: "roboarena"
loader_name: "roboarena_loader"
data_path: "test_datasets/DataDump_08-05-2025" # Adjust this to your dataset path downloaded from the original URL

# Optional: specify additional processing steps or filtering
# For example, to filter by task or success rate
# filters:
#   tasks: ["pick up the red block", "place the block in the tray"]
#   min_partial_success: 0.8
```

*   **`dataset_name`**: A unique identifier for the dataset.
*   **`loader_name`**: Must be `roboarena_loader` to use the provided Python loader.
*   **`data_path`**: The absolute or relative path to the root directory of your RoboArena dataset (e.g., `test_datasets/DataDump_08-05-2025`).

## Generating the HuggingFace Dataset

Once your configuration is set up, you can generate the HuggingFace dataset using the `generate_hf_dataset.py` script:

```bash
uv run python -m dataset_upload.generate_hf_dataset --config_path=dataset_upload/configs/data_gen_configs/roboarena.yaml
```

This command will:

1.  Load the RoboArena data using `roboarena_loader.py`.
2.  Process the video frames and metadata.
3.  Convert the data into a HuggingFace dataset format.
4.  Save the generated dataset.

## Data Fields in the HuggingFace Dataset

The resulting HuggingFace dataset will contain the following fields:

*   `id`: Unique identifier for each trajectory.
*   `task`: Language instruction for the task.
*   `frames`: A `RoboarenaFrameloader` object, which is a pickle-able loader that reads frames from the MP4 video on demand when called. It returns a `np.ndarray` of shape `(T, H, W, 3)` and `dtype uint8`.
*   `is_robot`: Boolean, always `True` for RoboArena.
*   `quality_label`: "successful" if `partial_success` is 1.0, otherwise "failure".
*   `partial_success`: The partial success metric (float between 0 and 1).

## Troubleshooting

*   **`FileNotFoundError`**: Ensure the `data_path` in your `roboarena.yaml` correctly points to the root of your RoboArena dataset.
*   **`ValueError: Unexpected frames shape`**: This indicates an issue with reading the video files. Verify that your MP4 files are not corrupted and are in a compatible format (e.g., H.264).
*   **Missing `metadata.yaml` or video files**: Double-check the directory structure of your downloaded dataset against the expected structure above. Ensure all necessary metadata and video files are present.
