#!/usr/bin/env python3
"""
RoboFAC dataset loader for the generic dataset converter for Robometer model training.
Loads MINT-SJTU/RoboFAC-dataset structure: realworld_data/<task>/videos/*.mp4 and simulation_data.

Task descriptions from the dataset card: https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset
"""

import json
import re
from pathlib import Path

from dataset_upload.helpers import generate_unique_id, load_sentence_transformer_model
from dataset_upload.video_helpers import load_video_frames
from tqdm import tqdm

# Official task name -> language description (from MINT-SJTU/RoboFAC-dataset dataset card).
# Task names match the card exactly (including typos: UprightStask, PegInsetionSide).
TASK_NAME_TO_DESCRIPTION: dict[str, str] = {
    "SpinStack": "Pick up the cube on the spinning disc and stack it on another cube on the disc.",
    "SpinPullStack": "Pull out the cube on the spinning disc and stack it on another cube on the disc.",
    "MicrowaveTask": "Put the spoon on the table into the cup. Open the door of microwave, put the cup into the microwave and close the door.",
    "SafeTask": "Put the gold bar into the safe, close the door of the safe and rotate the cross knob on the door to lock it.",
    "ToolsTask": "Choose the correct (L-shaped) tools, grasp it to pull the correct (2-pins) charger and plug it.",
    "UprightStask": "Upright the peg and stack it on the cube.",
    "PegInsetionSide": "Insert the peg into the hole on the side of the block.",
    "PullCubeTool": "Grasp the L-shaped tool and pull the cube by it.",
    "PlugCharger": "Grasp the charger and plug it into the receptacle.",
    "InsertCylinder": "Upright the cylinder and insert it into the middle hole on the shelf.",
    "PlaceCube": "Pick up the cube and place it into the box.",
    "LiftPegUpright": "Lift the peg and upright it.",
    "PickCube": "Pick the cube to the target position.",
    "PullCube": "Pull the cube to the red and white target.",
    "PushCube": "Push the cube to the red and white target.",
    "StackCube": "Pick up the cube and stack it on another cube.",
}

# Simulation path task names (e.g. UprightStack-v1) may differ from card names; map to card key.
_SIMULATION_PATH_TO_TASK_KEY: dict[str, str] = {
    "UprightStack": "UprightStask",
    "PegInsertionSide": "PegInsetionSide",
}


class RoboFACFrameLoader:
    """Pickle-able loader that reads RoboFAC video files on demand."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def __call__(self):
        """Load frames from video file. Returns np.ndarray (T, H, W, 3) uint8."""
        return load_video_frames(Path(self.file_path))


def _snake_to_camel(snake: str) -> str:
    """Convert snake_case to CamelCase (e.g. insert_cylinder -> InsertCylinder)."""
    return "".join(word.capitalize() for word in snake.split("_") if word)


def _realworld_folder_to_task_description(folder_name: str) -> str:
    """Convert realworld folder name to task description using dataset-card mapping.

    E.g. so100_insert_cylinder_error -> InsertCylinder -> full description.
    Falls back to title-cased name if not in TASK_NAME_TO_DESCRIPTION.
    """
    name = folder_name.replace("so100_", "").replace("_error", "").strip("_")
    task_key = _snake_to_camel(name)
    return TASK_NAME_TO_DESCRIPTION.get(task_key) or name.replace("_", " ").strip().title()


def _find_mp4_under(path: Path) -> list[Path]:
    """Find all .mp4 files under path (recursive). Handles videos/ or videos/chunk-000/ etc."""
    if not path.exists():
        return []
    return sorted(path.rglob("*.mp4"))


def _simulation_quality_from_folder(folder_name: str) -> str:
    """Map simulation_data subfolder to quality_label (successful / failure)."""
    name_lower = folder_name.lower()
    if "success" in name_lower and "fail" not in name_lower:
        return "successful"
    if "fail" in name_lower or "error" in name_lower:
        return "failure"
    return "failure"  # default for unknown


def _simulation_path_task_to_description(path_task_name: str) -> str:
    """Map simulation path task name to dataset-card description.

    success_data/ and failure_data/ have one subfolder per task variant, e.g.:
    UprightStack-v1, LiftPegUpright-box, PickCube-apple, MicrowaveTask-fork.
    Strip the trailing -<suffix> to get the base task name for lookup.
    """
    # Strip variant suffix: -v1, -box, -apple, -fork, -gen1, etc.
    base = re.sub(r"-[a-zA-Z0-9_]+$", "", path_task_name)
    task_key = _SIMULATION_PATH_TO_TASK_KEY.get(base, base)
    return TASK_NAME_TO_DESCRIPTION.get(task_key) or path_task_name.replace("-", " ").replace("_", " ").strip().title()


TASK_IDENTIFICATION_KEY = "Task identification"


def _extract_task_identification_desc(annos: dict) -> str:
    """Extract assistant 'value' from annos['Task identification'] conversation only.

    We use only the "Task identification" section (not other anno keys). The value is
    the assistant's reply in that conversation, e.g. "Insert the cylinder into the middle hole of the shelf."
    """
    # Prefer exact key, then case-insensitive match for "Task identification"
    task_id = annos.get(TASK_IDENTIFICATION_KEY)
    if task_id is None:
        for k, v in annos.items():
            if k.strip().lower() == TASK_IDENTIFICATION_KEY.lower():
                task_id = v
                break
    if not isinstance(task_id, list):
        return ""
    for turn in task_id:
        if turn.get("from") == "assistant" and "value" in turn:
            return turn["value"].strip()
    return ""


def _load_test_qa_annos(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load test_qa_realworld and test_qa_sim annos_per_video_split*.json and build video_id -> description.

    Each JSON is a dict: { "video_id": { "video": "path", "task": "InsertCylinder", "annos": { "Task identification": [ { "from": "human", "value": "..." }, { "from": "assistant", "value": "Insert the cylinder into the middle hole of the shelf." } ] } } }.
    We use the "Task identification" assistant value as the language description.
    Tries root/test_qa_realworld, root/test_qa_sim, and root/main/... for each.
    """
    video_id_to_desc: dict[str, str] = {}
    task_folder_to_desc: dict[str, str] = {}
    dirs_to_try = [
        root / "test_qa_realworld",
        root / "test_qa_sim",
        root / "main" / "test_qa_realworld",
        root / "main" / "test_qa_sim",
    ]
    for annos_dir in dirs_to_try:
        if not annos_dir.is_dir():
            continue
        for path in sorted(annos_dir.glob("annos_per_video_split*.json")):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for video_id, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                annos = entry.get("annos", {})
                desc = _extract_task_identification_desc(annos)
                if not desc:
                    continue
                video_id_to_desc[video_id] = desc
                video_path_str = entry.get("video", "")
                if video_path_str:
                    stem = Path(video_path_str).stem
                    video_id_to_desc[stem] = desc
                    task_folder_to_desc[video_path_str.split("/")[0]] = desc
                task = entry.get("task", "")
                if task:
                    task_folder_to_desc[task] = desc
    if video_id_to_desc:
        print(f"  Loaded {len(video_id_to_desc)} video_id->description from test_qa_* annos")
    
    return video_id_to_desc, task_folder_to_desc


# Human prompt that indicates task-identification QA in training_qa.json (we only extract from these).
_TRAINING_QA_TASK_IDENTIFICATION_PROMPT = "identify what task the robot is doing"


def _extract_task_desc_from_training_qa_conversations(convs: list) -> str:
    """Extract language description from training_qa conversations.

    Only use the assistant reply when the human asked the task-identification question
    (e.g. 'Can you identify what task the robot is doing in the provided video?').
    """
    if not isinstance(convs, list):
        return ""
    for i, turn in enumerate(convs):
        if turn.get("from") != "human" or "value" not in turn:
            continue
        human_val = (turn.get("value") or "").lower()
        if _TRAINING_QA_TASK_IDENTIFICATION_PROMPT not in human_val:
            continue
        # Next turn should be assistant with the task description
        if i + 1 < len(convs) and convs[i + 1].get("from") == "assistant" and "value" in convs[i + 1]:
            return (convs[i + 1].get("value") or "").strip()
    return ""


def _load_training_qa(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load training_qa.json and build video_id -> description and task_folder -> description.

    training_qa.json is a list of {
        "id": "09936392-adb5-4f34-9410-7c7305d9c76b",
        "video": "dataset_success_cleaned/MicrowaveTask-fork/stack_error/09936392-adb5-4f34-9410-7c7305d9c76b.mp4",
        "conversations": [
            { "from": "human", "value": "<video>\\nCan you identify what task the robot is doing in the provided video?" },
            { "from": "assistant", "value": "Put the fork in the cup and put them in the microwave" }
        ]
    }.
    We only extract the assistant reply when the human asked the task-identification question
    (contains 'identify what task the robot is doing'). That reply is the language description.
    Tries root/training_qa.json and root/main/training_qa.json.
    """
    video_id_to_desc: dict[str, str] = {}
    task_folder_to_desc: dict[str, str] = {}
    for candidate in (root / "training_qa.json", root / "main" / "training_qa.json"):
        if not candidate.exists():
            continue
        with open(candidate, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            video = item.get("video", "")
            video_id = item.get("id") or (Path(video).stem if video else "")
            if not video_id:
                continue
            convs = item.get("conversations", [])
            desc = _extract_task_desc_from_training_qa_conversations(convs)
            if not desc:
                continue
            video_id_to_desc[video_id] = desc
            if video:
                task_folder_to_desc[video.split("/")[0]] = desc
        if video_id_to_desc:
            print(f"  Loaded {len(video_id_to_desc)} video_id->description from {candidate}")
            return video_id_to_desc, task_folder_to_desc
            
    return video_id_to_desc, task_folder_to_desc


def _get_realworld_task_description(
    folder_name: str,
    task_folder_to_desc: dict[str, str] | None,
) -> str:
    """Get language description for realworld folder: prefer training_qa (match by base task name), else TASK_NAME_TO_DESCRIPTION. ipdb if no match."""
    name = folder_name.replace("so100_", "").replace("_error", "").strip("_")
    task_key = _snake_to_camel(name)
    if task_folder_to_desc:
        for qa_key, qa_desc in task_folder_to_desc.items():
            base = re.sub(r"-[a-zA-Z0-9_]+$", "", qa_key)
            card_key = _SIMULATION_PATH_TO_TASK_KEY.get(base, base)
            if card_key == task_key:
                return qa_desc
    task_desc = _realworld_folder_to_task_description(folder_name)
    if task_key not in TASK_NAME_TO_DESCRIPTION:
        import ipdb
        ipdb.set_trace()
    return task_desc


def _get_simulation_task_description(
    path_task_name: str,
    task_folder_to_desc: dict[str, str] | None,
) -> str:
    """Get language description for simulation task: prefer task_folder from training_qa, else TASK_NAME_TO_DESCRIPTION. ipdb if no match."""
    if task_folder_to_desc and path_task_name in task_folder_to_desc:
        return task_folder_to_desc[path_task_name]
    desc = _simulation_path_task_to_description(path_task_name)
    base = re.sub(r"-[a-zA-Z0-9_]+$", "", path_task_name)
    task_key = _SIMULATION_PATH_TO_TASK_KEY.get(base, base)
    if task_key not in TASK_NAME_TO_DESCRIPTION:
        import ipdb
        ipdb.set_trace()
    return desc


def _parse_simulation_video_path(
    video_path: Path,
    root: Path,
    video_id_to_desc: dict[str, str] | None = None,
    task_folder_to_desc: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    """From a video under simulation_data/, derive (task_description, quality_label, data_source).

    Supports two directory layouts:
    - simulation_data/success_data|failure_data/<task_folder>/.../<video>.mp4
      -> quality_folder = success_data|failure_data, path_task_name = task_folder (parts[2])
    - simulation_data/<task_folder>/view0|.../<video>.mp4  (no success_data/failure_data)
      -> path_task_name = task_folder (parts[1]), quality_folder = simulation_data
    """
    try:
        rel = video_path.relative_to(root)
    except ValueError:
        return "Simulation", "failure", "simulation_data"
    parts = rel.parts
    if len(parts) < 3:
        return "Simulation", "failure", "simulation_data"
    if parts[0] != "simulation_data":
        return "Simulation", "failure", "simulation_data"
    # If second component is success_data/failure_data, task folder is parts[2]; else it's parts[1] (e.g. LiftPegUpright-box/view0/...)
    if parts[1] in ("success_data", "failure_data"):
        quality_folder = parts[1]
        path_task_name = parts[2]
    else:
        quality_folder = "simulation_data"
        path_task_name = parts[1]
    quality_label = _simulation_quality_from_folder(quality_folder)
    # Prefer description by video id (stem of .mp4), then by task folder
    video_id = video_path.stem
    if video_id_to_desc and video_id in video_id_to_desc:
        task_desc = video_id_to_desc[video_id]
    else:
        task_desc = _get_simulation_task_description(path_task_name, task_folder_to_desc)
    data_source = f"simulation_data/{quality_folder}"
    return task_desc, quality_label, data_source


def _discover_robofac_trajectories(
    dataset_path: Path,
    *,
    realworld: bool = True,
    simulation: bool = True,
    video_id_to_desc: dict[str, str] | None = None,
    task_folder_to_desc: dict[str, str] | None = None,
) -> list[tuple[Path, str, str, str]]:
    """Discover all video files in RoboFAC dataset structure.

    Expected structure (from MINT-SJTU/RoboFAC-dataset):
        realworld_data/
            so100_insert_cylinder_error/
                videos/
                    *.mp4   OR  videos/chunk-000/*.mp4  (recursive)
            ...
        simulation_data/
            success_data/   or  failure_data/
                <TaskFolder>/   one subfolder per task variant (e.g. UprightStack-v1, LiftPegUpright-box, PickCube-apple)
                    (optional subdirs)
                        *.mp4

    If realworld_data is not found at dataset_path, also tries dataset_path / "main"
    (some download methods put repo content under a main/ subfolder).

    Returns:
        List of (video_path, task, quality_label, data_source) for each trajectory.
    """
    out: list[tuple[Path, str, str, str]] = []

    # Resolve root: support both /path/to/RoboFAC-dataset and /path/to/RoboFAC-dataset/main
    root = dataset_path
    realworld_path = root / "realworld_data"
    if not realworld_path.exists() and (root / "main").is_dir():
        root = root / "main"
        realworld_path = root / "realworld_data"
    if not realworld_path.exists():
        print(f"Warning: realworld_data not found at {dataset_path} or {dataset_path}/main")
    else:
        if realworld:
            for task_dir in sorted(realworld_path.iterdir()):
                if not task_dir.is_dir() or task_dir.name.startswith("."):
                    continue
                task_name = task_dir.name
                videos = _find_mp4_under(task_dir)
                # Realworld videos have generic stems (e.g. episode_000000) that repeat across tasks;
                # use task name only for description, not video_id lookup.
                for vid in videos:
                    task_desc = _get_realworld_task_description(task_name, task_folder_to_desc)
                    out.append((vid, task_desc, "failure", f"realworld_data/{task_name}"))
                if videos:
                    print(f"  realworld_data/{task_name}: {len(videos)} videos")

    if simulation:
        sim_path = root / "simulation_data"
        if sim_path.exists():
            # Discover all mp4s under simulation_data and parse path for task + quality
            videos = _find_mp4_under(sim_path)

            for vid in videos:
                task_desc, quality_label, data_source = _parse_simulation_video_path(
                    vid, root,
                    video_id_to_desc=video_id_to_desc,
                    task_folder_to_desc=task_folder_to_desc,
                )
                out.append((vid, task_desc, quality_label, data_source))
            if videos:
                print(f"  simulation_data: {len(videos)} videos (task/quality from path)")
        else:
            print("Warning: simulation_data not found")

    return out


def load_robofac_dataset(
    dataset_path: str,
    max_trajectories: int | None = None,
    realworld: bool = True,
    simulation: bool = True,
) -> dict[str, list[dict]]:
    """Load RoboFAC dataset and organize by task.

    Args:
        dataset_path: Path to the RoboFAC dataset root (e.g. .../RoboFAC-dataset or .../RoboFAC-dataset/main).
        max_trajectories: Maximum number of trajectories to load (None for all).
        realworld: Include realworld_data subfolders.
        simulation: Include simulation_data.

    Returns:
        Dictionary mapping task names to lists of trajectory dicts (frames, task, quality_label, etc.).
    """
    print("Loading RoboFAC dataset from:", dataset_path)
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"RoboFAC dataset path not found: {dataset_path}")

    # Build video_id -> description from test_qa_* annos (Task identification) and training_qa.json (human asks "identify what task the robot is doing")
    video_id_to_desc, task_folder_to_desc = _load_training_qa(dataset_path)
    test_vid, test_task = _load_test_qa_annos(dataset_path)
    video_id_to_desc.update(test_vid)
    task_folder_to_desc.update(test_task)

    print("Discovering videos (realworld_data/ and simulation_data/)...")
    traj_list = _discover_robofac_trajectories(
        dataset_path,
        realworld=realworld,
        simulation=simulation,
        video_id_to_desc=video_id_to_desc or None,
        task_folder_to_desc=task_folder_to_desc or None,
    )
    if not traj_list:
        raise FileNotFoundError(
            f"No .mp4 videos found under {dataset_path}. "
            "Check that the path points to the RoboFAC-dataset root (containing realworld_data/ and optionally simulation_data/). "
            "If you downloaded with Hugging Face CLI, the root may be under a 'main' subfolder; the loader will try that automatically."
        )
    if max_trajectories is not None and max_trajectories != -1:
        traj_list = traj_list[:max_trajectories]

    print(f"Found {len(traj_list)} trajectory videos total")

    task_data: dict[str, list[dict]] = {}
    for video_path, task_desc, quality_label, data_source in tqdm(
        traj_list, desc="Building RoboFAC trajectories"
    ):
        frame_loader = RoboFACFrameLoader(str(video_path))
        partial = 1.0 if quality_label == "successful" else 0.0
        trajectory = {
            "frames": frame_loader,
            "actions": None,
            "is_robot": True,
            "task": task_desc,
            "quality_label": quality_label,
            "data_source": data_source,
            "partial_success": partial,
            "id": generate_unique_id(),
        }
        task_data.setdefault(task_desc, []).append(trajectory)

    total = sum(len(v) for v in task_data.values())
    print(f"Loaded {total} trajectories from {len(task_data)} tasks")
    return task_data
