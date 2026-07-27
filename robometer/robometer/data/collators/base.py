import torch
from transformers import AutoProcessor, AutoTokenizer

from robometer.data.dataset_types import PreferenceSample, ProgressSample, SampleType


class BaseCollator:
    def __init__(
        self,
        processor: AutoProcessor,
        tokenizer: AutoTokenizer = None,
        max_length: int = 1024,
        resized_height: int = 128,
        resized_width: int = 128,
        base_model_id: str = None,
        load_embeddings: bool = False,
        **kwargs,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.tokenizer = tokenizer
        self.base_model_id = base_model_id
        self.load_embeddings = load_embeddings

        # Update processor based on base model id
        # if "SmolVLM" in self.base_model_id:
        #     # For image processor
        #     self.processor.image_processor.max_image_size = {"longest_edge": self.resized_height}
        #     self.processor.image_processor.size = {"longest_edge": self.resized_height}
        #     self.processor.image_processor.video_sampling["video_size"] = {"longest_edge": self.resized_height}

        #     # for video processor
        #     self.processor.video_processor.max_image_size = {"longest_edge": self.resized_height}
        #     self.processor.video_processor.size = {"longest_edge": self.resized_height}
        #     self.processor.video_processor.video_sampling["video_size"] = {"longest_edge": self.resized_height}

    def __call__(
        self,
        samples: list[SampleType],
    ) -> dict[str, torch.Tensor]:
        """
        Collate a list of samples into separate batches for preferences and progress.

        Args:
            samples: List of Sample objects or dictionaries that can be converted to Sample objects

        Returns:
            Dictionary containing separate batches for preferences and progress
        """
        # Convert dictionaries to Sample objects if needed
        sample_objects = []
        for sample in samples:
            if isinstance(sample, dict):
                sample_type = sample.get("sample_type", "unknown")
                if sample_type == "preference":
                    sample_obj = PreferenceSample(**sample)
                elif sample_type == "progress":
                    sample_obj = ProgressSample(**sample)
                else:
                    raise ValueError(f"Unknown sample_type: {sample_type}. Must be 'preference' or 'progress'")
                sample_objects.append(sample_obj)
            elif isinstance(sample, (PreferenceSample, ProgressSample)):
                sample_objects.append(sample)
            else:
                raise ValueError(f"Expected Sample object or dict, got {type(sample)}")

        preference_samples = [s for s in sample_objects if s.sample_type == "preference"]
        progress_samples = [s for s in sample_objects if s.sample_type == "progress"]

        preference_inputs = {}
        if preference_samples:
            preference_inputs = self._process_preference_batch(preference_samples)

        progress_inputs = {}
        if progress_samples:
            progress_inputs = self._process_progress_batch(progress_samples)

        return {
            "preference_inputs": preference_inputs,
            "progress_inputs": progress_inputs,
            "num_preferences": len(preference_samples),
            "num_progress": len(progress_samples),
        }
