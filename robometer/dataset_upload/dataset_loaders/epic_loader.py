# Epic kitchens 100
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from datasets import Dataset

from dataset_upload.helpers import (
    create_hf_trajectory,
    create_output_directory,
    generate_unique_id,
    load_sentence_transformer_model,
)


@dataclass
class EpicClip:
    participant_id: str
    video_id: str
    narration_id: str
    start_frame: int
    stop_frame: int
    narration: str


def _read_epic_csv(csv_path: Path) -> list[EpicClip]:
    clips: list[EpicClip] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                clips.append(
                    EpicClip(
                        participant_id=row["participant_id"].strip(),
                        video_id=row["video_id"].strip(),
                        start_frame=int(row["start_frame"]),
                        stop_frame=int(row["stop_frame"]),
                        narration=row["narration"].strip(),
                        narration_id=row["narration_id"].strip(),
                    )
                )
            except Exception:
                continue
    return clips


def _video_path_for_clip(dataset_path: Path, clip: EpicClip) -> Path:
    # video_id maps to video basename (without .MP4). Participant folder contains videos/ with .MP4
    # Example: <dataset_path>/P01/videos/<video_id>.MP4
    return dataset_path / clip.participant_id / "videos" / f"{clip.video_id}.MP4"


def _read_video_segment(video_path: Path, start_frame: int, stop_frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, min(start_frame, total - 1))
    end = max(start + 1, min(stop_frame, total))

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames: list[np.ndarray] = []
    idx = start
    while idx < end:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        idx += 1

    cap.release()
    return np.asarray(frames, dtype=np.uint8)


def _process_single_epic_clip(args: tuple[Any, ...]) -> dict | None:
    (
        clip,
        dataset_name,
        root,
        output_dir,
        max_frames,
        fps,
        shortest_edge_size,
        center_crop,
        lang_vec,
    ) = args

    video_path = _video_path_for_clip(root, clip)  # derive from output_dir -> dataset root

    if not video_path.exists():
        return None

    # skip anything > 1000
    if clip.stop_frame - clip.start_frame > 1000:
        print("Skipping clip because it's too long, length is", clip.stop_frame - clip.start_frame)
        return None
    frames = _read_video_segment(video_path, clip.start_frame, clip.stop_frame)
    if frames.size == 0:
        return None

    traj = {
        "id": generate_unique_id(),
        "task": clip.narration,
        "frames": frames,
        "is_robot": False,
        "quality_label": "successful",
        "preference_group_id": None,
        "preference_rank": None,
    }

    out_dir = os.path.join(output_dir, dataset_name.lower(), clip.participant_id)
    os.makedirs(out_dir, exist_ok=True)
    out_video = os.path.join(out_dir, f"{clip.narration_id}.mp4")

    entry = create_hf_trajectory(
        traj_dict=traj,
        video_path=out_video,
        lang_vector=lang_vec,
        max_frames=max_frames,
        dataset_name=dataset_name,
        use_video=True,
        fps=fps,
        shortest_edge_size=shortest_edge_size,
        center_crop=center_crop,
    )
    if entry:
        entry["frames"] = os.path.relpath(out_video, output_dir)
    return entry


def convert_epic_dataset_to_hf(
    dataset_path: str,
    dataset_name: str,
    output_dir: str,
    max_trajectories: int | None = None,
    max_frames: int = 64,
    fps: int = 10,
    num_workers: int = -1,
    shortest_edge_size: int = 240,
    center_crop: bool = False,
) -> Dataset:
    """Convert EPIC-KITCHENS to HF format by writing videos directly (H2R/OXE-style)."""

    create_output_directory(output_dir)
    root = Path(dataset_path)
    csv_path = root / "EPIC_100_train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"EPIC_100_train.csv not found at {csv_path}")

    clips = _read_epic_csv(csv_path)
    if not clips:
        return Dataset.from_dict({
            "id": [],
            "task": [],
            "lang_vector": [],
            "data_source": [],
            "frames": [],
            "is_robot": [],
            "quality_label": [],
            "preference_group_id": [],
            "preference_rank": [],
        })

    # Language model and cache
    lang_model = load_sentence_transformer_model()
    lang_cache: dict[str, Any] = {}

    # Determine workers
    if num_workers == -1:
        try:
            from multiprocessing import cpu_count as _cpu_count

            num_workers = min(_cpu_count(), 8)
        except Exception:
            num_workers = 1
    elif num_workers == 0:
        num_workers = 1

    batch_size = 6
    entries: list[dict] = []
    produced = 0
    max_limit = float("inf") if (max_trajectories is None or max_trajectories == -1) else int(max_trajectories)

    file_batch: list[EpicClip] = []
    vec_batch: list[np.ndarray] = []

    from tqdm import tqdm

    for idx, clip in tqdm(enumerate(clips), desc="iterating through EPIC-KITCHENS Clips", total=len(clips)):
        if produced >= max_limit:
            break

        # Precompute language vector
        if clip.narration not in lang_cache:
            lang_cache[clip.narration] = lang_model.encode(clip.narration)
        lang_vec = lang_cache[clip.narration]

        file_batch.append(clip)
        vec_batch.append(lang_vec)

        if len(file_batch) >= batch_size or idx + 1 == len(clips):
            worker_args = [
                (
                    clip,
                    dataset_name,
                    root,
                    output_dir,
                    max_frames,
                    fps,
                    shortest_edge_size,
                    center_crop,
                    vec,
                )
                for clip, vec in zip(file_batch, vec_batch)
            ]

            if num_workers == 1:
                for args in worker_args:
                    entry = _process_single_epic_clip(args)
                    if entry:
                        entries.append(entry)
                        produced += 1
                        if produced >= max_limit:
                            break
            else:
                from multiprocessing import Pool
                from tqdm import tqdm

                with Pool(processes=num_workers) as pool:
                    results = list(
                        tqdm(
                            pool.imap_unordered(_process_single_epic_clip, worker_args),
                            total=len(worker_args),
                            desc=f"Processing EPIC clips (workers={num_workers})",
                        )
                    )
                for entry in results:
                    if entry:
                        entries.append(entry)
                        produced += 1
                        if produced >= max_limit:
                            break

            file_batch = []
            vec_batch = []

    if not entries:
        return Dataset.from_dict({
            "id": [],
            "task": [],
            "lang_vector": [],
            "data_source": [],
            "frames": [],
            "is_robot": [],
            "quality_label": [],
            "preference_group_id": [],
            "preference_rank": [],
        })

    return Dataset.from_list(entries)
