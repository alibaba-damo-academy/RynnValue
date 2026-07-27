#!/usr/bin/env python3
"""
Evaluation configuration for RBM.
This file contains evaluation configuration classes:
- EvalServerConfig: For evaluation server runs (eval_server.py)
- OfflineEvalConfig: For standalone evaluation runs (run_eval_only.py)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

from hydra.core.config_store import ConfigStore

from robometer.configs.experiment_configs import CustomEvaluationConfig, DataConfig


@dataclass
class EvalServerConfig:
    """Configuration for evaluation server runs (eval_server.py)."""

    # Model path to load training config from
    model_path: str = field(
        default="./rbm_model_output/checkpoint-1000",
        metadata={"help": "Path to the trained model checkpoint (will load training_config.yaml from here)"},
    )

    # GPU Pool settings
    num_gpus: int = field(default=1, metadata={"help": "Number of GPUs to use (None for all available)"})
    max_workers: int = field(
        default=1, metadata={"help": "Max worker threads (None for auto, typically same as num_gpus)"}
    )

    # Evaluation parameters
    batch_size: int = field(default=4, metadata={"help": "Batch size for evaluation"})
    server_url: str = field(default="0.0.0.0", metadata={"help": "Evaluation server URL"})
    server_port: int = field(default=8000, metadata={"help": "Evaluation server port"})
    use_frame_steps: bool = field(
        default=False,
        metadata={"help": "Server-side default for use_frame_steps when a request does not set it"},
    )


@dataclass
class OfflineEvalConfig:
    """Configuration for standalone evaluation runs (run_eval_only.py)."""

    # Model path (HuggingFace model ID or local checkpoint path)
    model_path: str = field(
        default="",
        metadata={"help": "HuggingFace model ID (e.g., 'aliangdw/rbm_model') or local checkpoint path"},
    )

    # Output directory for evaluation results
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for evaluation results (defaults to checkpoint_path/eval_output)"},
    )

    # Custom evaluation configuration
    custom_eval: CustomEvaluationConfig = field(
        default_factory=CustomEvaluationConfig,
        metadata={"help": "Custom evaluation configuration (reused from experiment_configs)"},
    )

    def __post_init__(self):
        """Convert nested dict configs to dataclass instances."""
        if isinstance(self.custom_eval, dict):
            self.custom_eval = CustomEvaluationConfig(**self.custom_eval)


# Model-specific configuration classes
@dataclass
class RLVLMFConfig:
    """Configuration for RL-VLM-F baseline model."""

    vlm_provider: str = field(
        default="gemini",
        metadata={"help": "VLM provider for RL-VLM-F: 'gemini' or 'openai'"},
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for RL-VLM-F"},
    )


@dataclass
class GVLConfig:
    """Configuration for GVL baseline model."""

    api_key: Optional[str] = field(
        default=None,
        metadata={"help": "API key (defaults to GEMINI_API_KEY or OPENAI_API_KEY env var based on provider)"},
    )
    provider: str = field(
        default="gemini",
        metadata={"help": "API provider: 'gemini' or 'openai'"},
    )
    model_name: str = field(
        default="gemini-2.5-flash-lite",
        metadata={"help": "Model name (e.g., 'gemini-2.0-flash' for Gemini, 'gpt-4o' for OpenAI)"},
    )
    offset: float = field(
        default=0.5,
        metadata={"help": "Frame offset for GVL"},
    )
    # Retry settings for API throttling
    max_retries: int = field(
        default=5,
        metadata={"help": "Max retry attempts on 429/5xx errors"},
    )
    base_delay: float = field(
        default=1.0,
        metadata={"help": "Base delay in seconds for exponential backoff"},
    )
    max_delay: float = field(
        default=60.0,
        metadata={"help": "Maximum delay between retries"},
    )


@dataclass
class VLACConfig:
    """Configuration for VLAC baseline model."""

    device: str = field(
        default="cuda:0",
        metadata={"help": "Device for VLAC model"},
    )
    model_type: str = field(
        default="internvl2",
        metadata={"help": "VLAC model type"},
    )
    temperature: float = field(
        default=0.5,
        metadata={"help": "Temperature for VLAC"},
    )
    batch_size: int = field(
        default=5,
        metadata={"help": "Batch size for VLAC processing"},
    )
    skip: int = field(
        default=5,
        metadata={"help": "Pair-wise step size for VLAC"},
    )
    frame_skip: bool = field(
        default=True,
        metadata={"help": "Whether to skip frames for VLAC efficiency"},
    )
    use_images: bool = field(
        default=False,
        metadata={
            "help": "If True, use image mode (get_trajectory_critic). If False, use video mode (web_trajectory_critic)"
        },
    )


@dataclass
class RynnValueConfig:
    """Configuration for RynnValue baseline model (Qwen3VL-based)."""

    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to fine-tuned checkpoint dir containing model.pt (loads HF artifacts from sibling huggingface/ dir)"},
    )

    stride: int = field(
        default=1,
        metadata={"help": "Stride between context frames in process_episode"},
    )

    num_frames: int = field(
        default=8,
        metadata={"help": "Number of context frames per prediction window"},
    )

    mode: str = field(
        default="absolute",
        metadata={
            "help": "Prediction mode: 'absolute' (negated remaining seconds), "
            "'normalized' (1 - t/max), or 'remaining_time' (raw seconds)"
        },
    )

    camera_desc_lookup_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to JSON file with per-trajectory camera_description lookup (id -> camera_description)"},
    )

    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Max tokens generated for the Analysis block (description/match/success) in compute_progress"},
    )

    confusion_score_mode: str = field(
        default="match_binary",
        metadata={
            "help": "Confusion-matrix scoring on RynnValue: 'match_binary' (1.0 if Analysis Match:Yes else 0.0) "
            "or 'normalized_value' (value head normalized to [0,1] via 1 - t/t_max on Match:Yes, else 0.0). "
            "Only affects compute_progress on confusion-matrix samples."
        },
    )

    use_fuse: bool = field(
        default=False,
        metadata={
            "help": "If True, fuse the absolute remaining-time head with the relative per-step head via "
            "bidirectional TD(lambda) (fuse_td_lambda) in compute_progress. Requires a checkpoint that "
            "exposes a relative value head; no-ops otherwise."
        },
    )

    fuse_lambda: float = field(
        default=0.5,
        metadata={
            "help": "Lambda for TD(lambda) fusion when use_fuse=True. 0 leans on the absolute head one step "
            "away; 1 integrates the relative head from each anchor. Ignored when use_fuse=False."
        },
    )


@dataclass
class RBMConfig:
    """Configuration for RBM/ReWiND baseline model."""

    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for RBM/ReWiND model inference"},
    )

@dataclass
class RoboRewardConfig:
    """Configuration for RoboReward baseline model."""

    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum number of tokens to generate for RoboReward"},
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Number of samples per model.generate() call in compute_progress_batched"},
    )


@dataclass
class RoboDopamineConfig:
    """Configuration for Robo-Dopamine GRM baseline model."""

    frame_interval: int = field(
        default=1,
        metadata={"help": "Step between sampled frames for before/after pairs (1 = every frame)"},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for vLLM inference"},
    )
    eval_mode: str = field(
        default="incremental",
        metadata={"help": "Evaluation mode: 'incremental', 'forward', or 'backward'"},
    )


@dataclass
class TopRewardConfig:
    """Configuration for TOPReward baseline (token-probability zero-shot rewards)."""

    max_frames: int = field(
        default=64,
        metadata={"help": "Maximum frames per trajectory (subsampled if longer)"},
    )
    num_prefix_samples: int = field(
        default=15,
        metadata={"help": "Number of trajectory prefix lengths to evaluate for progress curve"},
    )
    reduction: str = field(
        default="mean",
        metadata={"help": "Reduction over instruction tokens: 'mean' or 'sum'"},
    )
    add_chat_template: bool = field(
        default=True,
        metadata={"help": "Use model chat template for instruction prompt"},
    )
    fps: float = field(
        default=2.0,
        metadata={"help": "Frames per second for video input to VLM"},
    )


@dataclass
class BaselineEvalConfig:
    """Configuration for baseline evaluation runs (run_baseline_eval.py)."""

    # Reward model discriminator: "gvl", "vlac", "rlvlmf", "rbm", "rewind", "roboreward", "robodopamine", or "topreward"
    reward_model: str = field(
        default="rlvlmf",
        metadata={
            "help": "Reward model: 'gvl', 'vlac', 'robodopamine', 'topreward' for progress; 'rlvlmf' for preference; 'rbm', 'rewind' for trained models; 'roboreward' for RoboReward baseline"
        },
    )

    model_config: Any = field(
        default=None,
        metadata={"help": "Model-specific configuration (automatically selected based on reward_model)"},
    )

    # Shared configuration
    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to model checkpoint (HuggingFace repo ID or local path). Used by vlac, rbm, rewind, and roboreward models."
        },
    )
    max_frames: int = field(
        default=15,
        metadata={"help": "Maximum frames for models (used by GVL and potentially others)"},
    )

    # Custom evaluation configuration
    custom_eval: CustomEvaluationConfig = field(
        default_factory=CustomEvaluationConfig,
        metadata={"help": "Custom evaluation configuration"},
    )

    # Output directory for evaluation results
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for evaluation results"},
    )

    # Whether to render per-trajectory progress MP4 visualizations
    save_videos: bool = field(
        default=True,
        metadata={"help": "If False, skip rendering per-trajectory progress MP4 visualizations"},
    )

    # Server settings (for baseline_eval_server.py)
    server_url: str = field(
        default="0.0.0.0",
        metadata={"help": "Evaluation server URL"},
    )
    server_port: int = field(
        default=8001,
        metadata={"help": "Evaluation server port"},
    )
    use_frame_steps: bool = field(
        default=False,
        metadata={"help": "Server-side default for use_frame_steps when a request does not set it"},
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for batched model inference (e.g. rynnvalue frame-step sub-samples)"},
    )

    def __post_init__(self):
        """Convert nested dict configs to dataclass instances and initialize model_config."""
        # Handle custom_eval conversion
        if isinstance(self.custom_eval, dict):
            self.custom_eval = CustomEvaluationConfig(**self.custom_eval)

        # Handle model_config conversion based on reward_model
        # If model_config is None or a dict, convert to appropriate dataclass
        if self.model_config is None or isinstance(self.model_config, dict):
            if self.reward_model == "rlvlmf":
                self.model_config = RLVLMFConfig(**(self.model_config or {}))
            elif self.reward_model == "gvl":
                self.model_config = GVLConfig(**(self.model_config or {}))
            elif self.reward_model == "vlac":
                self.model_config = VLACConfig(**(self.model_config or {}))
            elif self.reward_model in ["rewind", "rbm"]:
                self.model_config = RBMConfig(**(self.model_config or {}))
            elif self.reward_model == "roboreward":
                self.model_config = RoboRewardConfig(**(self.model_config or {}))
            elif self.reward_model == "robodopamine":
                self.model_config = RoboDopamineConfig(**(self.model_config or {}))
            elif self.reward_model == "topreward":
                self.model_config = TopRewardConfig(**(self.model_config or {}))
            elif self.reward_model == "rynnvalue":
                self.model_config = RynnValueConfig(**(self.model_config or {}))
            else:
                raise ValueError(
                    f"Unknown reward_model: {self.reward_model}. "
                    f"Must be 'rlvlmf', 'gvl', 'vlac', 'rbm', 'rewind', 'roboreward', 'robodopamine', 'topreward', or 'rynnvalue'"
                )


# Register structured configs with Hydra
cs = ConfigStore.instance()
cs.store(name="eval_server_config", node=EvalServerConfig)
cs.store(name="eval_only_config", node=OfflineEvalConfig)
cs.store(name="baseline_eval_config", node=BaselineEvalConfig)
