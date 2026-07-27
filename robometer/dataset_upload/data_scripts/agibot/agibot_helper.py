"""Utilities for indexing and looking up AgiBotWorld task JSONs.

This module builds and uses an index mapping episode_id -> task_json_path,
so downstream scripts can quickly resolve which task file contains a given
episode. The index is stored as a JSON file on disk for fast re-use.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from glob import glob

# Default locations used by the downloader
DEFAULT_DATASET_ROOT: str = os.path.join("datasets", "tmp", "agibotworld_alpha")
DEFAULT_TASK_INFO_DIR: str = os.path.join(DEFAULT_DATASET_ROOT, "task_info")
DEFAULT_INDEX_PATH: str = os.path.join(DEFAULT_DATASET_ROOT, "episode_to_task_index.json")


def _task_json_files(task_info_dir: str) -> Iterable[str]:
    """Yield task JSON files within the given directory.

    Parameters
    - task_info_dir: Directory that contains task_*.json files
    """

    pattern = os.path.join(task_info_dir, "task_*.json")
    for file_path in sorted(glob(pattern)):
        # Guard against directories or non-files matching the pattern
        if os.path.isfile(file_path):
            yield file_path


def build_episode_to_task_index(
    task_info_dir: str = DEFAULT_TASK_INFO_DIR,
    output_index_path: str = DEFAULT_INDEX_PATH,
    verbose: bool = True,
) -> dict[str, str]:
    """Build an index mapping episode_id -> task_json_path and save to disk.

    Keys are stored as strings in the JSON index for portability, but callers
    may freely pass `int` episode ids to the lookup helpers provided here.

    Returns the mapping in-memory as well.
    """

    if not os.path.isdir(task_info_dir):
        raise FileNotFoundError(f"Task info directory not found: {os.path.abspath(task_info_dir)}")

    episode_to_task: dict[str, str] = {}
    duplicate_episode_ids: list[str] = []

    for json_path in _task_json_files(task_info_dir):
        try:
            with open(json_path, encoding="utf-8") as f:
                entries = json.load(f)
        except Exception as exc:  # pragma: no cover - defensive logging
            if verbose:
                print(f"[agibot_helper] Skipping unreadable file {json_path}: {exc}")
            continue

        if not isinstance(entries, list):
            if verbose:
                print(f"[agibot_helper] Unexpected JSON structure in {json_path}; expected a list.")
            continue

        for item in entries:
            if not isinstance(item, dict) or "episode_id" not in item:
                continue
            episode_id_val = item["episode_id"]
            # Normalize to string for JSON index stability
            episode_key = str(episode_id_val)
            if episode_key in episode_to_task:
                duplicate_episode_ids.append(episode_key)
            episode_to_task[episode_key] = json_path

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_index_path), exist_ok=True)
    with open(output_index_path, "w", encoding="utf-8") as f:
        json.dump(episode_to_task, f)

    if verbose:
        total = len(episode_to_task)
        dupes = len(set(duplicate_episode_ids))
        print(
            f"[agibot_helper] Built episode->task index with {total} episodes"
            + (f" ({dupes} duplicate ids overwritten)" if dupes else "")
            + f" at {os.path.abspath(output_index_path)}"
        )

    return episode_to_task


def load_episode_to_task_index(index_path: str = DEFAULT_INDEX_PATH) -> dict[str, str]:
    """Load the episode->task index from disk.

    Raises FileNotFoundError if the index doesn't exist.
    """

    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"Episode index not found: {os.path.abspath(index_path)}. Build it with build_episode_to_task_index()."
        )
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


def ensure_episode_index(
    task_info_dir: str = DEFAULT_TASK_INFO_DIR,
    index_path: str = DEFAULT_INDEX_PATH,
    verbose: bool = False,
) -> dict[str, str]:
    """Return an in-memory episode->task index, building it if missing."""

    if os.path.isfile(index_path):
        return load_episode_to_task_index(index_path)
    return build_episode_to_task_index(task_info_dir, index_path, verbose=verbose)


def find_task_json_for_episode(
    episode_id: int | str,
    task_info_dir: str = DEFAULT_TASK_INFO_DIR,
    index_path: str = DEFAULT_INDEX_PATH,
) -> str:
    """Return the path to the task JSON file containing the given episode.

    If the index file is missing, it will be built automatically.
    Raises KeyError if the episode id is not present in the index.
    """

    index = ensure_episode_index(task_info_dir=task_info_dir, index_path=index_path)
    key = str(episode_id)
    if key not in index:
        raise KeyError(f"Episode id {episode_id} not found in index at {os.path.abspath(index_path)}")
    return index[key]


def get_episode_record(
    episode_id: int | str,
    task_info_dir: str = DEFAULT_TASK_INFO_DIR,
    index_path: str = DEFAULT_INDEX_PATH,
) -> tuple[str, dict]:
    """Return (json_path, episode_record) for the given episode id.

    This reads the relevant task JSON file and returns the full dictionary
    entry for the requested episode. Raises KeyError if the episode isn't
    present.
    """

    json_path = find_task_json_for_episode(episode_id=episode_id, task_info_dir=task_info_dir, index_path=index_path)
    with open(json_path, encoding="utf-8") as f:
        entries = json.load(f)
    key = int(episode_id)
    for item in entries:
        if isinstance(item, dict) and int(item.get("episode_id", -1)) == key:
            return json_path, item
    raise KeyError(f"Episode id {episode_id} not found inside file {os.path.abspath(json_path)}")


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_INDEX_PATH",
    "DEFAULT_TASK_INFO_DIR",
    "build_episode_to_task_index",
    "ensure_episode_index",
    "find_task_json_for_episode",
    "get_episode_record",
    "load_episode_to_task_index",
]
