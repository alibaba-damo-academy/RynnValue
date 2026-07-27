import os
import sys

from huggingface_hub import snapshot_download

# Support running as a module (python -m robometer.data.data_scripts.agibot.download_task_jsons)
# and running directly as a script (python dataset_upload/data_scripts/agibot/download_task_jsons.py)
try:
    from .agibot_helper import (
        DEFAULT_DATASET_ROOT,
        DEFAULT_TASK_INFO_DIR,
        build_episode_to_task_index,
    )
except Exception:  # pragma: no cover - fallback for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from robometer.data.data_scripts.agibot.agibot_helper import (  # type: ignore
        DEFAULT_DATASET_ROOT,
        DEFAULT_TASK_INFO_DIR,
        build_episode_to_task_index,
    )


def main() -> None:
    os.makedirs(DEFAULT_DATASET_ROOT, exist_ok=True)

    # Download task JSONs
    snapshot_download(
        repo_id="agibot-world/AgiBotWorld-Alpha",
        allow_patterns="task_info/*.json",
        local_dir=DEFAULT_DATASET_ROOT,
        repo_type="dataset",
    )

    # Build episode->task index to speed up lookups later
    build_episode_to_task_index(task_info_dir=DEFAULT_TASK_INFO_DIR, verbose=True)


if __name__ == "__main__":
    main()
