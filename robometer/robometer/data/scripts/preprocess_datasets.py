#!/usr/bin/env python3
"""
Dataset preprocessing script that creates index-based caches for fast trajectory access.
Uses HuggingFace's .map() for efficient processing and saves trajectory indices.
Creates one unified cache for each dataset/subset split.
"""

import datetime
import shutil
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import decord  # type: ignore
import torch
from pyrallis import wrap
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

from datasets import Dataset, DatasetDict, Video, load_dataset
from robometer.utils.distributed import rank_0_print
from robometer.utils.embedding_utils import compute_video_embeddings, compute_text_embeddings

# VIDEO_ERROR_PRINTED = False
# maps subsets to functions that filter the dataset. If true, the example is dropped.
soar_bad_trajectories = [
    # success
    "ca7fec9e-0c19-4e43-8503-ac6cf274a5f3",
    "96913909-04c5-4b95-91b4-42f59dffa90c",
    "521ea359-caf2-449d-b9c7-73574ce6de01",
    "a0c119ab-2cd4-4f65-a0b4-8fb736322bf0",
    "116411d4-fb67-4cde-ba17-e45edf60d922",
    "db1582fc-76c9-4069-ac6d-a296e574900c",
    # failure
    "e0cd2f7e-cfc6-46fb-9880-d65318709642",
    "2dfaf622-9aac-4965-b492-dcb5c6da8c62",
    "88ea7b35-c9f2-494f-9cc4-4c9a5c8def6d",
    "52cdf32b-702d-417c-b29f-9c1cbe022755",
    "c7868d1d-6e99-4490-b9ad-41b0e9c89069",
    "ee6735c6-9d11-4412-9b34-11a9e6d67646",
    "fed7da83-a9d6-4c71-acd6-5b6a212b1e6a",
    "d31ad630-14d2-47f2-b4a9-0579247b72b8",
    "31f3679d-2242-4d98-89d9-6a05b06fc807",
    "a49d8c5f-bf02-42f5-b130-1f3d53968120",
]
filters = {
    "jesbu1/molmoact_rfm/molmoact_dataset_tabletop": lambda x: "load the bowl" in x["task"].lower(),
    "jesbu1/galaxea_rfm/galaxea_part1_r1_lite": lambda x: all(
        word in x["task"].lower() for word in ["return", "to", "initial", "position"]
    ),
    "jesbu1/galaxea_rfm/galaxea_part2_r1_lite": lambda x: all(
        word in x["task"].lower() for word in ["return", "to", "initial", "position"]
    ),
    "jesbu1/galaxea_rfm/galaxea_part3_r1_lite": lambda x: all(
        word in x["task"].lower() for word in ["return", "to", "initial", "position"]
    ),
    "jesbu1/galaxea_rfm/galaxea_part4_r1_lite": lambda x: all(
        word in x["task"].lower() for word in ["return", "to", "initial", "position"]
    ),
    "jesbu1/galaxea_rfm/galaxea_part5_r1_lite": lambda x: all(
        word in x["task"].lower() for word in ["return", "to", "initial", "position"]
    ),
    "jesbu1/soar_rfm/soar_rfm": lambda x: x["id"] in soar_bad_trajectories,
    "jesbu1/auto_eval_rfm/auto_eval_rfm": lambda x: x["frames_shape"][0]
    <= 5,  # some episodes are too short and are likely poor/faulty success detections
    "anqil/rh20t_subset_rfm/rh20t_human": lambda x: x["frames_shape"][0] < 16,  # some episodes are too short
    "anqil/rh20t_subset_rfm/rh20t_robot": lambda x: x["frames_shape"][0] < 16,  # some episodes are too short
}


@dataclass
class DataPreprocessConfig:
    """Configuration for data loading and processing."""

    # Dataset paths and subsets
    train_datasets: list[str] = field(default_factory=lambda: [], metadata={"help": "List of training dataset names"})
    train_subsets: list[list[str]] = field(
        default_factory=lambda: [[]], metadata={"help": "List of training dataset subsets"}
    )
    eval_datasets: list[str] = field(default_factory=lambda: [], metadata={"help": "List of evaluation dataset names"})
    eval_subsets: list[list[str]] = field(
        default_factory=lambda: [[]], metadata={"help": "List of evaluation dataset subsets"}
    )

    # Video processing parameters
    max_frames_for_preprocessing: int = field(
        default=64, metadata={"help": "Maximum number of frames to extract from videos for preprocessing"}
    )
    video_frame_sampling: str = field(
        default="uniform", metadata={"help": "Frame sampling strategy: 'uniform', 'random', 'start', 'end'"}
    )
    num_proc: int = field(default=1, metadata={"help": "Number of processes for dataset processing"})
    force_reprocess: bool = field(
        default=False, metadata={"help": "Force reprocessing of datasets even if cache exists"}
    )
    num_threads: int = field(default=36, metadata={"help": "Number of threads for dataset processing"})
    cache_dir: str = field(default="", metadata={"help": "Directory to store processed dataset caches"})

    # Embedding preprocessing parameters
    precompute_embeddings: bool = field(
        default=False, metadata={"help": "Whether to precompute DINOv2 and sentence transformer embeddings"}
    )
    embeddings_cache_dir: str = field(
        default="embeddings_cache", metadata={"help": "Directory to save precomputed embeddings"}
    )
    dinov2_model: str = field(default="facebook/dinov2-base", metadata={"help": "DINOv2 model for video embeddings"})
    sentence_model: str = field(
        default="sentence-transformers/all-MiniLM-L12-v2",
        metadata={"help": "Sentence transformer model for text embeddings"},
    )
    embedding_batch_size: int = field(default=32, metadata={"help": "Batch size for embedding computation"})


class DatasetPreprocessor:
    """Unified preprocessor for all datasets with individual caching per dataset/subset."""

    def __init__(self, config):
        self.config = config

        # Dataset storage - store individual datasets
        self.datasets: dict[str, Dataset] = {}  # key: "dataset_path/subset"
        self.dataset_indices: dict[str, dict] = {}  # key: "dataset_path/subset", value: index mappings

        # Initialize embedding models if precompute_embeddings is enabled
        self.dinov2_model = None
        self.dinov2_processor = None
        self.sentence_model = None

        if config.precompute_embeddings:
            rank_0_print("🚀 Initializing embedding models...")

            # Initialize DINOv2 for video embeddings
            self.dinov2_model = AutoModel.from_pretrained(config.dinov2_model)
            self.dinov2_processor = AutoImageProcessor.from_pretrained(config.dinov2_model, use_fast=True)
            self.dinov2_model.eval()

            # Initialize Sentence Transformer for text embeddings
            self.sentence_model = SentenceTransformer(config.sentence_model)

            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dinov2_model = self.dinov2_model.to(device)
            self.sentence_model = self.sentence_model.to(device)

            rank_0_print(f"✅ Embedding models initialized on {device}")

    def preprocess_datasets(self):
        """Preprocess each dataset individually and create index-based caches."""
        rank_0_print("\n🔧 Preprocessing all datasets...")

        # Collect all dataset/subset combinations
        all_datasets = []

        # Add training datasets
        for dataset_path, dataset_subsets in zip(self.config.train_datasets, self.config.train_subsets, strict=False):
            if isinstance(dataset_subsets, str):
                dataset_subsets = [dataset_subsets]
            for subset in dataset_subsets:
                all_datasets.append((dataset_path, subset))

        # Add evaluation datasets
        for dataset_path, dataset_subsets in zip(self.config.eval_datasets, self.config.eval_subsets, strict=False):
            if isinstance(dataset_subsets, str):
                dataset_subsets = [dataset_subsets]
            for subset in dataset_subsets:
                all_datasets.append((dataset_path, subset))

        # Show which datasets are already preprocessed
        self._show_preprocessed_datasets(all_datasets)

        # Process each dataset and its associated subsets
        for i, (dataset_path, subset) in enumerate(all_datasets):
            rank_0_print(f"\n📚 Processing dataset {i + 1}/{len(all_datasets)}: {dataset_path}/{subset}")

            # Create individual cache key
            cache_key = f"{dataset_path}/{subset}"
            individual_cache_dir = os.path.join(self.config.cache_dir, cache_key.replace("/", "_").replace(":", "_"))

            # Check if already processed
            if os.path.exists(individual_cache_dir) and not self.config.force_reprocess:
                rank_0_print(f"    ✅ Cache already exists at {individual_cache_dir}, loading...")
                self._load_individual_cache(individual_cache_dir, cache_key)
                continue

            if os.path.exists(individual_cache_dir) and self.config.force_reprocess:
                rank_0_print(f"    ❌ Force reprocessing enabled, deleting cache at {individual_cache_dir}")
                shutil.rmtree(individual_cache_dir)

            # Load and process individual dataset
            try:
                dataset = self._load_dataset_from_path(dataset_path, subset)

                # Handle DatasetDict
                if isinstance(dataset, DatasetDict):
                    if "train" in dataset:
                        dataset = dataset["train"]
                    else:
                        rank_0_print(f"    ⚠️  Warning: No 'train' split found in {dataset_path}/{subset}")
                        continue

                rank_0_print(f"    📥 Loaded {len(dataset)} trajectories from {dataset_path}/{subset}")

                # Process this individual dataset
                processed_dataset, indices = self._process_individual_dataset(dataset, individual_cache_dir, cache_key)

                # Store processed dataset and indices
                self.datasets[cache_key] = processed_dataset
                self.dataset_indices[cache_key] = indices

                # Save individual cache
                self._save_individual_cache(individual_cache_dir, processed_dataset, indices, dataset_path, subset)

                rank_0_print(f"    ✅ Successfully processed and cached {dataset_path}")

            except Exception as e:
                rank_0_print(f"    ❌ Failed to process {dataset_path}: {e}")
                continue

        if not self.datasets:
            raise ValueError("No datasets were successfully processed")

        rank_0_print(f"✅ Successfully processed {len(self.datasets)} datasets")

        # Log summary of processed datasets
        total_trajectories = sum(len(dataset) for dataset in self.datasets.values())
        rank_0_print(f"📊 Total trajectories across all datasets: {total_trajectories}")

        for cache_key, dataset in self.datasets.items():
            rank_0_print(f"  📚 {cache_key}: {len(dataset)} trajectories")

        # Show final status summary
        self._show_final_status_summary(all_datasets)

    def _process_individual_dataset(self, dataset: Dataset, cache_dir: str, cache_key: str):
        """Process a single dataset and build its index mappings."""
        # Process videos and build indices
        # Only cast to Video with decode=True for the .map path. The threaded path will
        # open and close readers per-sample to avoid keeping many files open at once.
        if self.config.num_threads > 1:
            processed_dataset, indices = self._process_dataset_videos_threaded(dataset, cache_dir, cache_key)
        else:
            dataset = dataset.cast_column("frames_video", Video(decode=True))
            processed_dataset, indices = self._process_dataset_videos_map(dataset, cache_dir, cache_key)

        return processed_dataset, indices

    def _compute_video_embeddings(self, frames_array: np.ndarray) -> torch.Tensor:
        """
        Compute DINOv2 embeddings for video frames.

        Args:
            frames_array: Video frames as numpy array (T, H, W, C)

        Returns:
            Video embeddings as torch tensor (T, D) where D is embedding dimension
        """
        return compute_video_embeddings(
            frames_array,
            self.dinov2_model,
            self.dinov2_processor,
            batch_size=self.config.embedding_batch_size,
            use_autocast=True,
        )

    def _compute_text_embeddings(self, text: str) -> torch.Tensor:
        """
        Compute sentence transformer embeddings for text.

        Args:
            text: Text description

        Returns:
            Text embedding as torch tensor (D,) where D is embedding dimension
        """
        return compute_text_embeddings(
            text,
            self.sentence_model,
            use_autocast=True,
            show_progress_bar=False,
        )

    def _save_embeddings(self, video_embeddings: torch.Tensor, text_embedding: torch.Tensor, embeddings_path: str):
        """
        Save video and text embeddings to a .pt file.

        Args:
            video_embeddings: Video embeddings tensor (T, D)
            text_embedding: Text embedding tensor (D,)
            embeddings_path: Path to save the .pt file
        """
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)

        embeddings_data = {
            "video_embeddings": video_embeddings,
            "text_embedding": text_embedding,
            "video_shape": video_embeddings.shape,
            "text_shape": text_embedding.shape,
        }

        torch.save(embeddings_data, embeddings_path)

    def _process_embeddings_for_example(
        self, frames_array: np.ndarray, task_text: str, idx: int, example_id: str, embeddings_dir: str
    ) -> tuple[str, tuple, tuple]:
        """
        Process embeddings for a single example.

        Args:
            frames_array: Video frames as numpy array
            task_text: Task description text
            idx: Example index
            example_id: Example ID
            embeddings_dir: Directory to save embeddings

        Returns:
            Tuple of (embeddings_filepath, video_embedding_shape, text_embedding_shape)
        """
        if not self.config.precompute_embeddings:
            return None, None, None

        # Create embeddings directory
        os.makedirs(embeddings_dir, exist_ok=True)

        # Compute video embeddings
        video_embeddings = self._compute_video_embeddings(frames_array)

        # Compute text embeddings
        text_embedding = self._compute_text_embeddings(task_text)

        # Save embeddings
        embeddings_filename = f"trajectory_{idx:06d}_{example_id}_embeddings.pt"
        embeddings_filepath = os.path.join(embeddings_dir, embeddings_filename)
        self._save_embeddings(video_embeddings, text_embedding, embeddings_filepath)

        return embeddings_filepath, video_embeddings.shape, text_embedding.shape

    def _process_dataset_videos_map(self, dataset, cache_dir: str, cache_key: str):
        """
        Process dataset frames using .map() method for efficient on-the-fly processing.
        Also builds index mappings during the same pass to avoid multiple iterations.
        Frames are saved as .npz files and only file paths are stored in the dataset.

        Args:
            dataset: HuggingFace dataset containing trajectories

        Returns:
            Dataset with processed frame paths and metadata
        """
        raise NotImplementedError("This method is not properly implemented rn, use threaded version with 1 process.")
        # Check if frames are already processed (npz file paths)
        sample_item = dataset[0]
        frames_data = sample_item.get("frames")

        if isinstance(frames_data, str) and frames_data.endswith(".npz") and not self.config.force_reprocess:
            rank_0_print("Frames already processed as npz file paths, skipping processing.")
            return dataset, {}  # Return empty index mappings if already processed
        elif self.config.force_reprocess and isinstance(frames_data, str) and frames_data.endswith(".npz"):
            rank_0_print("Force reprocessing enabled. Reprocessing frames despite being already processed.")

        rank_0_print("Processing video frames into npz files and building index mappings...")

        # Debug: Check the dataset structure
        sample_item = dataset[0]
        rank_0_print(f"Sample dataset item keys: {list(sample_item.keys())}")
        rank_0_print(f"Sample item structure: {sample_item}")

        # Initialize index mappings
        robot_trajectories = []
        human_trajectories = []
        optimal_by_task = {}
        suboptimal_by_task = {}
        quality_indices = {}
        task_indices = {}
        source_indices = {}
        partial_success_indices = {}

        frames_dir = os.path.join(cache_dir, "frames")
        embeddings_dir = os.path.join(cache_dir, self.config.embeddings_cache_dir)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)

        def process_videos_and_build_indices(example, idx):
            """Process frames and build index mappings in a single pass."""
            # Debug: Log what we're processing
            if idx < 3:  # Only log first 5 examples to avoid spam
                rank_0_print(f"Processing example {idx}: {example['id']} - {example['task']}")

            # filter the example if needed
            should_drop = filters.get(cache_key, lambda x: False)(example)
            if should_drop:
                rank_0_print(f"Dropping example {idx}: {example['id']} - {example['task']}")
                return {"frames": None, "frames_processed": False}

            # Get the video reader object from the Video feature
            frames = example.get("frames_video")
            if frames is None:
                rank_0_print(f"Warning: No frames_video for example {idx}")
                return {"frames": None, "frames_processed": False}

            # Process video frames using the _preprocess_videos method
            frames_array = self._preprocess_videos(frames, self.config.max_frames_for_preprocessing)

            if frames_array.size == 0:
                rank_0_print(f"Warning: No frames processed for example {idx}")
                return {"frames": None, "frames_processed": False}

            # Save frames as npz file
            frames_filename = f"trajectory_{example['id']}.npz"
            frames_filepath = os.path.join(frames_dir, frames_filename)

            # Store file path and metadata in dataset (not the actual frames)
            example["frames"] = frames_filepath  # Store path to npz file
            example["frames_shape"] = frames_array.shape
            example["num_frames"] = frames_array.shape[0] if len(frames_array.shape) > 0 else 0
            example["frames_processed"] = True

            # Compute and save embeddings if enabled
            embeddings_filepath, video_emb_shape, text_emb_shape = self._process_embeddings_for_example(
                frames_array, example.get("task", ""), idx, example["id"], embeddings_dir
            )

            if embeddings_filepath is not None:
                # Store embeddings path in dataset
                example["embeddings_path"] = embeddings_filepath
                example["video_embedding_shape"] = video_emb_shape
                example["text_embedding_shape"] = text_emb_shape

            # BUILD INDEX MAPPINGS DURING THE SAME PASS
            # Debug: Log the values we're extracting
            if idx < 3:
                rank_0_print(
                    f"  Example {idx} - is_robot: {example.get('is_robot', True)}, task: {example.get('task', 'unknown')}, quality: {example.get('quality_label', 'successful')}"
                )

            # Robot/Human trajectories
            if example.get("is_robot", True):
                robot_trajectories.append(idx)
            else:
                human_trajectories.append(idx)

            # Quality-based indices
            quality = example["quality_label"]
            if quality not in quality_indices:
                quality_indices[quality] = []
            quality_indices[quality].append(idx)

            # Task-based indices
            task = example["task"]
            if task not in task_indices:
                task_indices[task] = []
            task_indices[task].append(idx)

            # Source-based indices
            source = example["data_source"]
            if source not in source_indices:
                source_indices[source] = []
            source_indices[source].append(idx)

            # Partial success-based indices
            partial_success = example.get("partial_success", None)
            if partial_success is not None and quality == "failure":  # only record partial success for failure cases
                if partial_success not in partial_success_indices:
                    partial_success_indices[partial_success] = []
                partial_success_indices[partial_success].append(idx)

            # Optimal/Suboptimal by task
            if task not in optimal_by_task:
                optimal_by_task[task] = []
                suboptimal_by_task[task] = []

            if "frames_video" in example:
                del example["frames_video"]

            if quality in ["successful", "optimal"]:
                optimal_by_task[task].append(idx)
            elif quality in ["suboptimal", "failed", "failure"]:
                suboptimal_by_task[task].append(idx)

            # Save frames with metadata
            np.savez_compressed(
                frames_filepath,
                frames=frames_array,
                shape=frames_array.shape,
                num_frames=frames_array.shape[0] if len(frames_array.shape) > 0 else 0,
            )

            return example

        # Apply the mapping function to the dataset
        processed_dataset = dataset.map(
            process_videos_and_build_indices,
            with_indices=True,
            desc="Processing video frames and building indices",
            num_proc=self.config.num_proc,
        )

        # Filter out dropped examples (those with frames=None and frames_processed=False)
        original_length = len(processed_dataset)

        # Build a list of which indices were kept (had frames_processed=True)
        kept_indices = []
        for idx in range(original_length):
            if processed_dataset[idx].get("frames_processed", False):
                kept_indices.append(idx)

        # Filter the dataset
        processed_dataset = processed_dataset.filter(lambda x: x.get("frames_processed", False))
        num_filtered = original_length - len(processed_dataset)

        if num_filtered > 0:
            rank_0_print(f"Filtered out {num_filtered} trajectories")

            # Build mapping from old indices to new indices
            old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}

            # Remap all indices
            robot_trajectories = [old_to_new_idx[idx] for idx in robot_trajectories if idx in old_to_new_idx]
            human_trajectories = [old_to_new_idx[idx] for idx in human_trajectories if idx in old_to_new_idx]

            quality_indices = {
                key: [old_to_new_idx[idx] for idx in indices if idx in old_to_new_idx]
                for key, indices in quality_indices.items()
            }

            task_indices = {
                key: [old_to_new_idx[idx] for idx in indices if idx in old_to_new_idx]
                for key, indices in task_indices.items()
            }

            source_indices = {
                key: [old_to_new_idx[idx] for idx in indices if idx in old_to_new_idx]
                for key, indices in source_indices.items()
            }

            partial_success_indices = {
                key: [old_to_new_idx[idx] for idx in indices if idx in old_to_new_idx]
                for key, indices in partial_success_indices.items()
            }

            optimal_by_task = {
                key: [old_to_new_idx[idx] for idx in indices if idx in old_to_new_idx]
                for key, indices in optimal_by_task.items()
            }

            suboptimal_by_task = {
                key: [old_to_new_idx[idx] for idx in indices if idx in old_to_new_idx]
                for key, indices in suboptimal_by_task.items()
            }

        # Log the built indices
        rank_0_print(f"Built index mappings for {cache_key}:")
        rank_0_print(f"  Robot trajectories: {len(robot_trajectories)}")
        rank_0_print(f"  Human trajectories: {len(human_trajectories)}")
        rank_0_print(f"  Tasks: {len(task_indices)}")
        rank_0_print(f"  Quality labels: {len(quality_indices)}")
        rank_0_print(f"  Data sources: {len(source_indices)}")
        rank_0_print(f"  Partial success indices: {len(partial_success_indices)}")

        return processed_dataset, {
            "robot_trajectories": robot_trajectories,
            "human_trajectories": human_trajectories,
            "optimal_by_task": optimal_by_task,
            "suboptimal_by_task": suboptimal_by_task,
            "quality_indices": quality_indices,
            "task_indices": task_indices,
            "source_indices": source_indices,
            "partial_success_indices": partial_success_indices,
        }

    def _process_dataset_videos_threaded(self, dataset, cache_dir: str, cache_key: str):
        """
        Threaded implementation to process frames and build indices, saving npz files concurrently.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from datasets import Sequence, Value

        # Initialize index mappings
        robot_trajectories = []
        human_trajectories = []
        optimal_by_task = {}
        suboptimal_by_task = {}
        quality_indices = {}
        task_indices = {}
        source_indices = {}
        partial_success_indices = {}

        frames_dir = os.path.join(cache_dir, "frames")
        embeddings_dir = os.path.join(cache_dir, self.config.embeddings_cache_dir)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)

        def process_one(idx: int):
            global VIDEO_ERROR_PRINTED
            # Fetch example inside the worker to avoid holding many decoded readers
            example: dict[str, Any] = dataset[idx]
            frames_src = example.get("frames_video")
            if frames_src is None:
                return idx, None, None, None, None, None

            # If the source is a path string, open with a lightweight reader
            try:
                # Try decord first (fastest for video decoding)
                vr = decord.VideoReader(frames_src, num_threads=1)
                total_frames = len(vr)

                # Sample frames efficiently
                if total_frames <= self.config.max_frames_for_preprocessing:
                    frame_indices = list(range(total_frames))
                else:
                    # Uniform sampling
                    frame_indices = [
                        int(i * total_frames / self.config.max_frames_for_preprocessing)
                        for i in range(self.config.max_frames_for_preprocessing)
                    ]

                frames_array = vr.get_batch(frame_indices).asnumpy()

                del vr

            except Exception as e:
                rank_0_print(f"Error in _process_one: {e}")
                return idx, None, None, None, None, None

            # Save frames as npz file
            frames_filename = f"trajectory_{example['id']}.npz"
            frames_filepath = os.path.join(frames_dir, frames_filename)
            np.savez_compressed(
                frames_filepath,
                frames=frames_array,
                shape=frames_array.shape,
                num_frames=frames_array.shape[0] if len(frames_array.shape) > 0 else 0,
            )

            # Process embeddings if enabled
            embeddings_filepath, video_emb_shape, text_emb_shape = self._process_embeddings_for_example(
                frames_array, example.get("task", ""), idx, example["id"], embeddings_dir
            )

            # Return minimal info to update record and indices
            return idx, frames_filepath, frames_array.shape, embeddings_filepath, video_emb_shape, text_emb_shape

        # Prepare indices only; fetch rows inside workers to limit open file handles
        indices_only = list(range(len(dataset)))
        # indices_only = list(range(100))
        _ = tqdm(indices_only, total=len(indices_only), desc=f"Indexing {cache_key}", unit="traj", leave=False)

        # Concurrently convert and write npz for all examples, collecting results
        idx_to_data = {}
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            futures = [executor.submit(process_one, i) for i in indices_only]
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc=f"Processing {cache_key}", unit="traj", leave=False
            ):
                idx, frames_path, shape, embeddings_path, video_emb_shape, text_emb_shape = fut.result()

                if frames_path is not None:
                    idx_to_data[idx] = (frames_path, shape, embeddings_path, video_emb_shape, text_emb_shape)

        # Update dataset with saved paths and build indices
        updated_rows = []
        kept_indices = []  # Track which original indices were kept

        for i in tqdm(indices_only, total=len(indices_only), desc=f"Finalizing {cache_key}", unit="traj", leave=False):
            ex = dataset[i]
            if i in idx_to_data:
                frames_path, shape, embeddings_path, video_emb_shape, text_emb_shape = idx_to_data[i]
                ex = dict(ex)
                ex["frames"] = frames_path
                ex["frames_shape"] = shape
                ex["num_frames"] = shape[0] if len(shape) > 0 else 0
                ex["frames_processed"] = True

                should_drop = filters.get(cache_key, lambda x: False)(ex)
                if should_drop:
                    # remove the frames_path and then continue processing others
                    os.remove(frames_path)
                    print(f"Removed frames_path {frames_path} for {ex['id']} because it was filtered out")
                    continue

                # Add embedding information if available
                if embeddings_path is not None:
                    ex["embeddings_path"] = embeddings_path
                    ex["video_embedding_shape"] = video_emb_shape
                    ex["text_embedding_shape"] = text_emb_shape

                # Track that this index was kept
                kept_indices.append(i)

                # Build indices using the NEW index (position in updated_rows)
                new_idx = len(updated_rows)

                if ex.get("is_robot", True):
                    robot_trajectories.append(new_idx)
                else:
                    human_trajectories.append(new_idx)

                quality = ex.get("quality_label", "successful")
                quality_indices.setdefault(quality, []).append(new_idx)

                task = ex.get("task", "unknown")
                task_indices.setdefault(task, []).append(new_idx)

                source = ex.get("data_source", "unknown")
                source_indices.setdefault(source, []).append(new_idx)

                partial_success = ex.get("partial_success", None)
                if (
                    partial_success is not None and quality == "failure"
                ):  # only record partial success for failure cases
                    partial_success_indices.setdefault(partial_success, []).append(new_idx)

                if task not in optimal_by_task:
                    optimal_by_task[task] = []
                    suboptimal_by_task[task] = []
                if quality in ["successful", "optimal"]:
                    optimal_by_task[task].append(new_idx)
                elif quality in ["suboptimal", "failed", "failure"]:
                    suboptimal_by_task[task].append(new_idx)

                # Drop any lingering video readers from rows to free FDs
                if isinstance(ex, dict):
                    ex.pop("frames_video", None)

                updated_rows.append(ex)

        # Report filtering stats
        num_filtered = len(indices_only) - len(kept_indices)
        if num_filtered > 0:
            rank_0_print(f"Filtered out {num_filtered} trajectories")

        # Create a new Dataset from updated_rows
        processed_dataset = Dataset.from_list(updated_rows)
        processed_dataset = processed_dataset.cast_column("lang_vector", Sequence(feature=Value("float32")))

        # Log the built indices
        rank_0_print(f"Built index mappings for {cache_key} (threaded):")
        rank_0_print(f"  Robot trajectories: {len(robot_trajectories)}")
        rank_0_print(f"  Human trajectories: {len(human_trajectories)}")
        rank_0_print(f"  Tasks: {len(task_indices)}")
        rank_0_print(f"  Quality labels: {len(quality_indices)}")
        rank_0_print(f"  Data sources: {len(source_indices)}")
        rank_0_print(f"  Partial success indices: {len(partial_success_indices)}")

        return processed_dataset, {
            "robot_trajectories": robot_trajectories,
            "human_trajectories": human_trajectories,
            "optimal_by_task": optimal_by_task,
            "suboptimal_by_task": suboptimal_by_task,
            "quality_indices": quality_indices,
            "task_indices": task_indices,
            "source_indices": source_indices,
            "partial_success_indices": partial_success_indices,
        }

    def _save_individual_cache(
        self,
        cache_dir: str,
        processed_dataset: Dataset,
        indices: dict,
        dataset_path: str,
        subset: str,
    ):
        """Save the processed dataset and index mappings for an individual dataset."""
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Save the processed dataset
        dataset_cache_dir = os.path.join(cache_dir, "processed_dataset")
        processed_dataset.save_to_disk(dataset_cache_dir)

        # Save index mappings
        index_mappings = indices

        mappings_file = os.path.join(cache_dir, "index_mappings.json")
        with open(mappings_file, "w") as f:
            json.dump(index_mappings, f, indent=2)

        # Save dataset info
        dataset_info = {
            "dataset_path": dataset_path,
            "subset": subset,
            "total_trajectories": len(processed_dataset),
            "cache_timestamp": str(datetime.datetime.now()),
            "config_hash": self._get_config_hash(),
        }

        info_file = os.path.join(cache_dir, "dataset_info.json")
        with open(info_file, "w") as f:
            json.dump(dataset_info, f, indent=2)

        rank_0_print(f"Individual cache saved to {cache_dir}")

    def _get_config_hash(self) -> str:
        """Generate a hash of the relevant config parameters."""
        import hashlib

        config_str = str(self.config.max_frames_for_preprocessing)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _load_individual_cache(self, cache_dir: str, cache_key: str):
        """Load a pre-processed dataset and its index mappings from a cache directory."""
        dataset_cache_dir = os.path.join(cache_dir, "processed_dataset")
        if not os.path.exists(dataset_cache_dir):
            raise FileNotFoundError(f"Processed dataset not found at {dataset_cache_dir}")

        # Load the processed dataset
        self.datasets[cache_key] = Dataset.load_from_disk(dataset_cache_dir)

        # Load index mappings
        mappings_file = os.path.join(cache_dir, "index_mappings.json")
        if not os.path.exists(mappings_file):
            raise FileNotFoundError(f"Index mappings not found at {mappings_file}")

        with open(mappings_file) as f:
            self.dataset_indices[cache_key] = json.load(f)

        # Log loaded cache
        rank_0_print(f"  📂 Loaded individual cache from {cache_dir}")

    def get_combined_indices(self):
        """Get combined index mappings from all individual datasets."""
        if not self.dataset_indices:
            return {}

        # Combine indices from all datasets
        combined_indices = {
            "robot_trajectories": [],
            "human_trajectories": [],
            "optimal_by_task": {},
            "suboptimal_by_task": {},
            "quality_indices": {},
            "task_indices": {},
            "source_indices": {},
            "partial_success_indices": {},
        }

        # Track offset for each dataset
        offset = 0

        for cache_key, indices in self.dataset_indices.items():
            # Adjust indices by adding offset
            for key in combined_indices:
                if key in indices:
                    if isinstance(indices[key], list):
                        # For list indices, add offset
                        combined_indices[key].extend([idx + offset for idx in indices[key]])
                    elif isinstance(indices[key], dict):
                        # For dict indices, add offset to values
                        if key not in combined_indices[key]:
                            combined_indices[key] = {}
                        for subkey, subindices in indices[key].items():
                            if subkey not in combined_indices[key]:
                                combined_indices[key][subkey] = []
                            combined_indices[key][subkey].extend([idx + offset for idx in subindices])

            # Update offset for next dataset
            if cache_key in self.datasets:
                offset += len(self.datasets[cache_key])

        return combined_indices

    def _load_dataset_from_path(self, dataset_path: str, subset: str):
        """Load dataset from path with proper video handling."""
        if "/" in dataset_path and not os.path.exists(dataset_path):
            # Loading from HuggingFace Hub - handle video paths
            rank_0_print(f"Loading dataset: {dataset_path}")

            dataset_root = os.environ.get("ROBOMETER_DATASET_PATH", "")
            if not dataset_root:
                raise ValueError(
                    "ROBOMETER_DATASET_PATH not set. "
                    "Set it to the directory containing your downloaded datasets. "
                    "Example: export ROBOMETER_DATASET_PATH=/path/to/your/datasets"
                )

            dataset_name = dataset_path.split("/")[-1]

            def patch_path(old_path):
                root_dir = f"{dataset_root}/{dataset_name}"
                return f"{root_dir}/{old_path}"  # e.g., "./videos/trajectory_0000.mp4"

            # Load dataset
            dataset = load_dataset(dataset_path, name=subset, split="train")

            # dataset = dataset.select(range(100))

            # Just patch the paths, don't decode videos yet
            dataset = dataset.map(
                lambda x: {"frames_video": patch_path(x["frames"]), "frames_path": patch_path(x["frames"])}
            )
            return dataset
        else:
            # Load from local disk
            dataset = load_dataset(dataset_path)
            return dataset

    def _show_preprocessed_datasets(self, all_datasets: list[str]):
        """
        Show which datasets are already preprocessed and which are not.
        This helps avoid re-processing already cached datasets.
        """
        rank_0_print("\n🔍 Checking for pre-existing caches...")

        cached_count = 0
        total_count = len(all_datasets)

        for dataset_path, subset in all_datasets:
            cache_key = f"{dataset_path}/{subset}"
            individual_cache_dir = os.path.join(self.config.cache_dir, cache_key.replace("/", "_").replace(":", "_"))

            if os.path.exists(individual_cache_dir):
                cached_count += 1
                # Try to load cache info to show details
                info_file = os.path.join(individual_cache_dir, "dataset_info.json")
                if os.path.exists(info_file):
                    try:
                        with open(info_file) as f:
                            info = json.load(f)
                        trajectories = info.get("total_trajectories", "unknown")
                        timestamp = info.get("cache_timestamp", "unknown")
                        rank_0_print(
                            f"  ✅ {dataset_path}/{subset}: {trajectories} trajectories (cached at {timestamp})"
                        )
                    except:
                        rank_0_print(f"  ✅ {dataset_path}/{subset}: Cache exists but info file corrupted")
                else:
                    rank_0_print(f"  ✅ {dataset_path}/{subset}: Cache exists (no info file)")
            else:
                rank_0_print(f"  ❌ {dataset_path}/{subset}: No cache found")

        # Show summary
        rank_0_print("\n📊 Cache Status Summary:")
        rank_0_print(f"  ✅ Already cached: {cached_count}/{total_count} dataset/subset pairs")
        rank_0_print(f"  🔄 Need processing: {total_count - cached_count}/{total_count} dataset/subset pairs")

        if cached_count == total_count:
            rank_0_print("  🎉 All dataset/subset pairs are already cached! Use --force-reprocess to reprocess.")
        elif cached_count > 0:
            rank_0_print("  💡 Some dataset/subset pairs are cached. Only uncached ones will be processed.")
        else:
            rank_0_print("  🚀 No dataset/subset pairs are cached. All will be processed.")

    def _show_final_status_summary(self, all_datasets: list[tuple[str, str]]):
        """
        Show a summary of which datasets were processed and which were loaded from cache.
        """
        rank_0_print("\n📊 Final Status Summary for Dataset Preprocessing:")

        processed_count = 0
        loaded_count = 0
        total_count = len(all_datasets)

        for dataset_path, subset in all_datasets:
            cache_key = f"{dataset_path}/{subset}"
            individual_cache_dir = os.path.join(self.config.cache_dir, cache_key.replace("/", "_").replace(":", "_"))

            if cache_key in self.datasets:
                if os.path.exists(individual_cache_dir):
                    loaded_count += 1
                    rank_0_print(
                        f"  ✅ {dataset_path}/{subset}: Loaded from cache ({len(self.datasets[cache_key])} trajectories)"
                    )
                else:
                    processed_count += 1
                    rank_0_print(
                        f"  🔄 {dataset_path}/{subset}: Newly processed ({len(self.datasets[cache_key])} trajectories)"
                    )
            else:
                rank_0_print(f"  ❌ {dataset_path}/{subset}: Failed to load/process")

        # Show summary counts
        rank_0_print("\n📈 Processing Summary:")
        rank_0_print(f"  🔄 Newly processed: {processed_count} dataset/subset pairs")
        rank_0_print(f"  ✅ Loaded from cache: {loaded_count} dataset/subset pairs")
        rank_0_print(f"  ❌ Failed: {total_count - processed_count - loaded_count} dataset/subset pairs")
        rank_0_print(f"  📊 Total available: {processed_count + loaded_count}/{total_count} dataset/subset pairs")


@wrap()
def main(config: DataPreprocessConfig):
    """Main preprocessing function."""
    # Try to increase the soft limit for open files to reduce 'Too many open files'
    try:
        import resource  # type: ignore

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target_soft = min(hard, 65535)
        if soft < target_soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard))
    except Exception:
        pass

    # Show dataset structure info
    print("\n🏗️  Dataset Configuration Structure:")
    print("Training datasets:")
    for dataset_path, dataset_subsets in zip(config.train_datasets, config.train_subsets, strict=False):
        if isinstance(dataset_subsets, str):
            dataset_subsets = [dataset_subsets]
        print(f"  📚 {dataset_path}: {len(dataset_subsets)} subset(s)")
        for subset in dataset_subsets:
            print(f"    📂 {subset}")

    print("Evaluation datasets:")
    for dataset_path, dataset_subsets in zip(config.eval_datasets, config.eval_subsets, strict=False):
        if isinstance(dataset_subsets, str):
            dataset_subsets = [dataset_subsets]
        print(f"  📚 {dataset_path}: {len(dataset_subsets)} subset(s)")
        for subset in dataset_subsets:
            print(f"    📂 {subset}")

    print("\n💡 Note: Each dataset can now have multiple subsets!")
    print("   - Single subset: ['subset1'] or 'subset1'")
    print("   - Multiple subsets: ['subset1', 'subset2', 'subset3']")

    # Create unified preprocessor for all datasets
    preprocessor = DatasetPreprocessor(config)

    # Preprocess all datasets
    print("\n=== Processing All Datasets ===")

    preprocessor.preprocess_datasets()

    # Test the caches
    print("\n=== Testing Caches ===")

    if preprocessor.datasets:
        print("\n📚 All Datasets:")
        total_trajectories = sum(len(dataset) for dataset in preprocessor.datasets.values())
        print(f"  Total trajectories: {total_trajectories}")

        # Get combined indices
        combined_indices = preprocessor.get_combined_indices()
        if combined_indices:
            print(f"  Robot trajectories: {len(combined_indices.get('robot_trajectories', []))}")
            print(f"  Human trajectories: {len(combined_indices.get('human_trajectories', []))}")
            print(f"  Tasks: {list(combined_indices.get('task_indices', {}).keys())}")

        # Test direct access to first dataset
        if preprocessor.datasets:
            first_dataset = next(iter(preprocessor.datasets.values()))
            if len(first_dataset) > 0:
                test_traj = first_dataset[0]
                print(f"  Sample trajectory: {test_traj['id']} - {test_traj['task']}")

    # Show individual dataset info
    print("\n📊 Individual Dataset Summary:")
    print(f"Total datasets processed: {len(preprocessor.datasets)}")
    for cache_key, dataset in preprocessor.datasets.items():
        print(f"  ✅ {cache_key}: {len(dataset)} trajectories")

    # Show dataset structure
    print("\n🏗️  Dataset Structure:")
    print("Training datasets:")
    for dataset_path, dataset_subsets in zip(config.train_datasets, config.train_subsets, strict=False):
        if isinstance(dataset_subsets, str):
            dataset_subsets = [dataset_subsets]
        print(f"  📚 {dataset_path}: {len(dataset_subsets)} subset(s)")
        for subset in dataset_subsets:
            cache_key = f"{dataset_path}/{subset}"
            if cache_key in preprocessor.datasets:
                print(f"    ✅ {subset}: {len(preprocessor.datasets[cache_key])} trajectories")
            else:
                print(f"    ❌ {subset}: Failed to load")

    print("Evaluation datasets:")
    for dataset_path, dataset_subsets in zip(config.eval_datasets, config.eval_subsets, strict=False):
        if isinstance(dataset_subsets, str):
            dataset_subsets = [dataset_subsets]
        print(f"  📚 {dataset_path}: {len(dataset_subsets)} subset(s)")
        for subset in dataset_subsets:
            cache_key = f"{dataset_path}/{subset}"
            if cache_key in preprocessor.datasets:
                print(f"    ✅ {subset}: {len(preprocessor.datasets[cache_key])} trajectories")
            else:
                print(f"    ❌ {subset}: Failed to load")

    print("\n✅ Dataset preprocessing complete!")
    print(f"Unified cache: {config.cache_dir}")
    print(f"Please set ROBOMETER_PROCESSED_DATASETS_PATH to {config.cache_dir} by running:")
    print(f"export ROBOMETER_PROCESSED_DATASETS_PATH={config.cache_dir}")


if __name__ == "__main__":
    main()
