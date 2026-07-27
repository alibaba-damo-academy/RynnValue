import collections
import copy
import io
import json
import os
import random
from typing import Dict, List, Tuple, Optional, Any, final

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer

from robometer.data.datasets.name_mapping import DS_SHORT_NAME_MAPPING
from robometer.evals.compile_results import (
    run_quality_preference_eval,
    run_reward_alignment_eval_per_trajectory,
    run_confusion_matrix_eval,
    run_policy_ranking_eval,
)
from robometer.models.utils import ModelOutput, convert_bins_to_continuous, convert_discrete_target_to_continuous
from robometer.utils.distributed import banner, get_rank, is_rank_0, log_fsdp_diagnostics
from robometer.utils.logger import Logger, get_logger, log_memory_usage
from robometer.utils.metrics import compute_spearman_correlation
from robometer.utils.setup_utils import setup_batch_collator, setup_custom_eval_dataset
from robometer.utils.tensor_utils import t2n
from robometer.utils.timer import _timer
from robometer.utils.video_utils import create_policy_ranking_grid

logger = get_logger()


def seed_worker(worker_id):
    """Set random seed for dataloader workers."""
    import random

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reduce_metrics_with_accelerate(metrics: Dict[str, Any], accelerator, aggregate_method="sum"):
    """
    Reduce multiple scalar metrics using Accelerate's built-in methods.
    Handles cases where different processes have different metric keys.
    metrics: dict of {name: float or tensor}
    Returns dict with averaged metrics across all ranks.
    """
    if not metrics:
        return metrics

    # Step 1: Gather all metric keys from all processes
    local_keys = list(metrics.keys())
    all_keys_gathered = accelerator.gather_for_metrics(local_keys)

    # Step 2: Create union of all keys across all processes
    all_unique_keys = set()
    for keys_from_process in all_keys_gathered:
        if isinstance(keys_from_process, list):
            all_unique_keys.update(keys_from_process)
        else:
            # Handle single key case
            all_unique_keys.add(keys_from_process)

    all_unique_keys = sorted(all_unique_keys)

    # Step 3: Create synchronized metrics dict with 0.0 for missing keys
    synchronized_metrics = {}
    for key in all_unique_keys:
        if key in metrics:
            synchronized_metrics[key] = metrics[key]
        else:
            # This process doesn't have this metric, use 0.0
            synchronized_metrics[key] = 0.0

    # Step 4: Now reduce all metrics (all processes have same keys)
    result_metrics = {}

    for key, value in synchronized_metrics.items():
        try:
            # Convert to tensor on accelerator device
            if torch.is_tensor(value):
                tensor_val = value.to(accelerator.device, dtype=torch.float32)
            else:
                tensor_val = torch.tensor(float(value), dtype=torch.float32, device=accelerator.device)

            # Check for NaN values before reduction
            if torch.isnan(tensor_val).any():
                logger.warning(f"NaN detected in metric '{key}', using 0.0")
                tensor_val = torch.tensor(0.0, dtype=torch.float32, device=accelerator.device)

            # Check for infinity values
            if torch.isinf(tensor_val).any():
                logger.warning(f"Infinity detected in metric '{key}', using 0.0")
                tensor_val = torch.tensor(0.0, dtype=torch.float32, device=accelerator.device)

            # Use accelerator's reduce method - all processes participate
            reduced_val = accelerator.reduce(tensor_val, reduction=aggregate_method)

            # Final check for NaN in reduced result
            if torch.isnan(reduced_val).any():
                logger.warning(f"NaN in reduced result for metric '{key}', using fallback")
                result_metrics[key] = 0.0
            else:
                result_metrics[key] = reduced_val.item()

        except Exception as metric_error:
            # If individual metric fails, keep original value (or 0.0 if missing)
            logger.warning(f"Failed to reduce metric '{key}': {metric_error}")
            if key in metrics:
                original_val = float(metrics[key]) if not torch.is_tensor(metrics[key]) else metrics[key].item()
                result_metrics[key] = 0.0 if np.isnan(original_val) else original_val
            else:
                result_metrics[key] = 0.0

    # Step 5: Return all reduced metrics (all processes should have the same keys after reduction)
    # Return all keys from result_metrics to ensure we get all metrics across all processes
    return result_metrics


class RBMHeadsTrainer(Trainer):
    def __init__(self, config, *args, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.log_metadata = collections.defaultdict(float)
        self.global_metadata = collections.defaultdict(float)
        self.timing_raw = collections.defaultdict(float)
        self._ddp_static_graph_set = False  # Flag to track if DDP static graph has been set
        self._fsdp_diagnostics_logged = False  # Flag to track if FSDP diagnostics have been logged

        if logger is not None:
            self.logger = logger
        else:
            log_level = self.config.logging.log_level
            self.logger = Logger(
                log_to=self.config.logging.log_to,
                output_dir=getattr(self.args, "output_dir", "./logs"),
                is_main_process=is_rank_0(),
                log_level=log_level,
            )

        # Use loguru logger after it's been initialized
        loguru_logger = get_logger()
        loguru_logger.info(f"DDP find_unused_parameters: {getattr(self.args, 'ddp_find_unused_parameters', 'N/A')}")

    def create_optimizer(self):
        """
        Override to create optimizer with separate parameter groups for vision encoder layers.
        If vision_encoder_lr is set, the last N vision encoder layers will use that LR,
        while all other parameters use the default learning rate.
        """
        # Check if we need to create parameter groups for vision encoder
        vision_encoder_lr = self.config.training.vision_encoder_lr
        vision_encoder_num_layers = self.config.training.vision_encoder_num_layers

        if vision_encoder_lr is None or vision_encoder_lr <= 0:
            # No special vision encoder LR, use default optimizer
            return super().create_optimizer()

        # Get the model
        model = self.model
        if not hasattr(model, "model") or not hasattr(model.model, "visual"):
            logger.warning(
                "vision_encoder_lr is set but model doesn't have visual encoder. "
                "Using default optimizer without parameter groups."
            )
            return super().create_optimizer()

        # Get vision encoder blocks
        visual_encoder = model.model.visual
        if not hasattr(visual_encoder, "blocks"):
            logger.warning(
                "vision_encoder_lr is set but visual encoder doesn't have blocks. "
                "Using default optimizer without parameter groups."
            )
            return super().create_optimizer()

        blocks = visual_encoder.blocks
        total_blocks = len(blocks)

        if vision_encoder_num_layers > total_blocks:
            logger.warning(
                f"vision_encoder_num_layers ({vision_encoder_num_layers}) is greater than "
                f"total blocks ({total_blocks}). Using all blocks for vision encoder LR."
            )
            vision_encoder_num_layers = total_blocks

        # Identify parameters for last N layers
        vision_encoder_params = []
        other_params = []

        # Get all parameters and their names
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if this parameter belongs to the last N vision encoder blocks
            is_vision_encoder_param = False
            if "visual.blocks" in name:
                # Extract block index from parameter name
                # Format: model.visual.blocks.{idx}.{rest}
                try:
                    parts = name.split("visual.blocks.")
                    if len(parts) > 1:
                        block_part = parts[1].split(".")[0]
                        block_idx = int(block_part)
                        # Check if this is one of the last N blocks
                        if block_idx >= (total_blocks - vision_encoder_num_layers):
                            is_vision_encoder_param = True
                except (ValueError, IndexError):
                    # If we can't parse the block index, skip this parameter
                    pass

            if is_vision_encoder_param:
                vision_encoder_params.append(param)
            else:
                other_params.append(param)

        if not vision_encoder_params:
            logger.warning(
                "No vision encoder parameters found for parameter groups. "
                "Using default optimizer without parameter groups."
            )
            return super().create_optimizer()

        # Use AdamW as default (same as HuggingFace Trainer)
        optimizer_kwargs = {
            "betas": (
                self.args.adam_beta1 if hasattr(self.args, "adam_beta1") else 0.9,
                self.args.adam_beta2 if hasattr(self.args, "adam_beta2") else 0.999,
            ),
            "eps": self.args.adam_epsilon if hasattr(self.args, "adam_epsilon") else 1e-8,
            "weight_decay": self.args.weight_decay,
        }

        # Create parameter groups with different learning rates
        param_groups = [
            {
                "params": other_params,
                "lr": self.args.learning_rate,
                **optimizer_kwargs,
            },
            {
                "params": vision_encoder_params,
                "lr": vision_encoder_lr,
                **optimizer_kwargs,
            },
        ]

        optimizer = torch.optim.AdamW(param_groups)

        logger.info(
            f"Created optimizer with parameter groups: "
            f"{len(other_params)} params at LR={self.args.learning_rate}, "
            f"{len(vision_encoder_params)} vision encoder params (last {vision_encoder_num_layers} blocks) at LR={vision_encoder_lr}"
        )
        self.optimizer = optimizer

        return optimizer

    def _post_checkpoint_load_reset(self):
        """
        Reset model and optimizer state after loading from checkpoint.
        This addresses issues where checkpoint loading can leave stale gradients
        or computational graph state that causes crashes during training.
        """
        logger.info("Performing post-checkpoint load reset...")

        # Ensure model is in training mode
        self.model.train()

        # Clear any cached gradients or computational graph state
        # NOTE: We don't clear optimizer.state or param_groups as that breaks the lr_scheduler
        try:
            # Zero out any existing gradients
            if hasattr(self, "optimizer") and self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
        except Exception as e:
            logger.warning(f"Could not clear gradients: {e}")

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Post-checkpoint load reset complete")

    def _normalize_list_like(self, value):
        """Convert None/scalars/tuples to a list so we can safely gather across ranks."""
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _gather_list_across_processes(self, value):
        """Gather Python lists (or list-like) across all ranks."""
        normalized = self._normalize_list_like(value)
        if not dist.is_initialized():
            return normalized

        world_size = dist.get_world_size()
        gathered = [None] * world_size
        dist.all_gather_object(gathered, normalized)

        flattened = []
        for proc_list in gathered:
            if not proc_list:
                continue
            flattened.extend(proc_list)
        return flattened

    def _gather_metadata_fields(self, sample_inputs: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        """Gather heterogeneous metadata fields (lists, tensors) across processes."""
        gathered = {}
        for field in fields:
            field_value = sample_inputs.get(field)
            if torch.is_tensor(field_value):
                gathered[field] = self.accelerator.gather_for_metrics(field_value)
            else:
                gathered[field] = self._gather_list_across_processes(field_value)
        return gathered

    def _truncate_metadata_lists(self, metadata: dict, max_len: int) -> dict:
        """Ensure metadata lists align with tensor batch sizes without altering tensors."""
        if max_len < 0:
            return metadata
        for key, value in metadata.items():
            if isinstance(value, list):
                metadata[key] = value[:max_len]
        return metadata

    def _get_learning_rate(self):
        """
        Override to safely get learning rate, handling cases where scheduler hasn't been stepped yet.
        """
        try:
            if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                last_lrs = self.lr_scheduler.get_last_lr()
                if last_lrs:
                    return last_lrs[0]
            # Fallback to optimizer's learning rate
            if hasattr(self, "optimizer") and self.optimizer is not None:
                if self.optimizer.param_groups:
                    return self.optimizer.param_groups[0]["lr"]
            # Last resort: return configured learning rate
            return self.args.learning_rate
        except Exception as e:
            logger.warning(f"Could not get learning rate: {e}")
            return self.args.learning_rate

    def train(self, resume_from_checkpoint=None, **kwargs):
        """
        Override train method to perform post-checkpoint reset.
        """
        # If resuming from checkpoint, set flag for reset in first training step
        if resume_from_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            self._just_resumed_from_checkpoint = True

        # Call parent train method
        result = super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step and log custom losses.
        """
        logger.debug("training_step: Starting")

        if not self._fsdp_diagnostics_logged:
            log_fsdp_diagnostics(model, accelerator=self.accelerator, logger=logger)
            self._fsdp_diagnostics_logged = True

        # Check if we just resumed from checkpoint (first step after resume)
        if hasattr(self, "_just_resumed_from_checkpoint") and self._just_resumed_from_checkpoint:
            self._post_checkpoint_load_reset()
            self._just_resumed_from_checkpoint = False

        self.timing_raw = {}

        # Initialize log_metadata
        self.log_metadata = {}

        # Safety check: ensure model is in training mode and gradients are properly set up
        if not model.training:
            logger.warning("Model not in training mode, setting to train mode")
            model.train()

        # Clear any stale gradients before starting
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        with _timer("time/training_step", timing_raw=self.timing_raw):
            loss = super().training_step(model, inputs, num_items_in_batch)

        # Extract the separate batches
        preference_inputs = inputs.get("preference_inputs", {})
        progress_inputs = inputs.get("progress_inputs", {})
        num_preferences = inputs.get("num_preferences", 0)
        num_progress = inputs.get("num_progress", 0)

        logger.trace(f"num_preferences: {num_preferences}, num_progress: {num_progress}")

        if num_preferences > 0 and preference_inputs:
            rejected_data_gen_strategy = preference_inputs["rejected_data_gen_strategy"]
            if isinstance(rejected_data_gen_strategy, list) and len(rejected_data_gen_strategy) > 0:
                for s in rejected_data_gen_strategy:
                    self.global_metadata[f"pref_{s}"] += 1

            data_sources = preference_inputs.get("data_source", None)
            if data_sources is not None:
                for ds in data_sources:
                    self.global_metadata[f"total_{ds}"] += 1.0

        if num_progress > 0 and progress_inputs:
            data_gen_strategy = progress_inputs["data_gen_strategy"]
            if isinstance(data_gen_strategy, list) and len(data_gen_strategy) > 0:
                for s in data_gen_strategy:
                    self.global_metadata[f"prog_{s}"] += 1

            data_sources = progress_inputs.get("data_source", None)
            if data_sources is not None:
                for ds in data_sources:
                    self.global_metadata[f"total_{ds}"] += 1.0

        # Update global metadata for training
        # add to total batch size and sum across all processes
        self.global_metadata["total_samples"] += num_preferences + num_progress
        self.global_metadata["total_preferences"] += num_preferences
        self.global_metadata["total_progress"] += num_progress

        logger.trace("finished updating global metadata")

        # self._update_resample_attempt_metrics(inputs)

        # logger.trace("update resample attempt metrics")

        # Log custom losses at specified intervals (using our custom logger only)
        if self.state.global_step % self.args.logging_steps == 0:
            try:
                self._log_metadata()
            except Exception as e:
                logger.warning(f"Error logging metadata: {e}")

        # Log GPU memory usage at every training step for diagnostics
        log_memory_usage(f"Step {self.state.global_step}")

        return loss

    def _get_optimizer_stats(self):
        """Get optimizer and gradient statistics for logging."""
        optim_stats = {}

        if not hasattr(self, "optimizer") or self.optimizer is None:
            return optim_stats

        # Get learning rates for each parameter group
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = param_group.get("lr", 0.0)
            optim_stats[f"optim/lr_group_{i}"] = lr

        # If only one param group, also log as optim/lr for convenience
        if len(self.optimizer.param_groups) == 1:
            optim_stats["optim/lr"] = self.optimizer.param_groups[0].get("lr", 0.0)

        # Compute gradient norms across all model parameters
        total_norm = 0.0
        num_params_with_grad = 0
        max_grad_norm = 0.0
        min_grad_norm = float("inf")

        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm**2
                num_params_with_grad += 1
                max_grad_norm = max(max_grad_norm, param_norm)
                min_grad_norm = min(min_grad_norm, param_norm)

        if num_params_with_grad > 0:
            total_norm = total_norm**0.5
            optim_stats["optim/preclip_grad_norm"] = total_norm
            optim_stats["optim/preclip_grad_norm_max"] = max_grad_norm
            optim_stats["optim/preclip_grad_norm_min"] = min_grad_norm if min_grad_norm != float("inf") else 0.0
            optim_stats["optim/num_params_with_grad"] = num_params_with_grad

        # Compute parameter norms across all model parameters
        total_param_norm = 0.0
        max_param_norm = 0.0
        min_param_norm = float("inf")
        param_norms = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.data is None:
                continue
            param_norm = p.data.norm(2).item()
            total_param_norm += param_norm**2
            max_param_norm = max(max_param_norm, param_norm)
            min_param_norm = min(min_param_norm, param_norm)
            param_norms.append((name, param_norm))

        if param_norms:
            total_param_norm = total_param_norm**0.5
            optim_stats["optim/param_norm"] = total_param_norm
            optim_stats["optim/param_norm_max"] = max_param_norm
            optim_stats["optim/param_norm_min"] = min_param_norm if min_param_norm != float("inf") else 0.0

        # Get optimizer state statistics (e.g., momentum, variance for Adam)
        if hasattr(self.optimizer, "state") and len(self.optimizer.state) > 0:
            # For Adam-like optimizers, log average momentum and variance
            exp_avg_norms = []
            exp_avg_sq_norms = []

            for state in self.optimizer.state.values():
                if "exp_avg" in state:
                    exp_avg_norms.append(state["exp_avg"].norm(2).item())
                if "exp_avg_sq" in state:
                    exp_avg_sq_norms.append(state["exp_avg_sq"].norm(2).item())

            if exp_avg_norms:
                optim_stats["optim/exp_avg_norm_mean"] = np.mean(exp_avg_norms)
                optim_stats["optim/exp_avg_norm_max"] = np.max(exp_avg_norms)
            if exp_avg_sq_norms:
                optim_stats["optim/exp_avg_sq_norm_mean"] = np.mean(exp_avg_sq_norms)
                optim_stats["optim/exp_avg_sq_norm_max"] = np.max(exp_avg_sq_norms)

        # Log top 10 parameters with largest gradient norms
        param_grad_norms = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            grad_norm = p.grad.data.norm(2).item()
            param_grad_norms.append((name, grad_norm))

        if param_grad_norms:
            # Sort by gradient norm (descending) and take top 10
            param_grad_norms.sort(key=lambda x: x[1], reverse=True)
            for i, (name, grad_norm) in enumerate(param_grad_norms[:5]):
                # Shorten parameter name for cleaner logging
                short_name = name.replace("model.", "").replace("module.", "")
                optim_stats[f"optim/top_preclip_grad_norm_{i + 1}_{short_name}"] = grad_norm

        if param_norms:
            # Sort by parameter norm (descending) and take top 10
            param_norms.sort(key=lambda x: x[1], reverse=True)
            for i, (name, param_norm) in enumerate(param_norms[:5]):
                short_name = name.replace("model.", "").replace("module.", "")
                optim_stats[f"optim/top_param_norm_{i + 1}_{short_name}"] = param_norm

        return optim_stats

    def _update_resample_attempt_metrics(self, inputs: Dict[str, Any]) -> None:
        """Aggregate resample attempt statistics across processes."""
        if not hasattr(self, "accelerator"):
            return

        local_pairs: List[Tuple[str, float]] = []

        for key in ("preference_inputs", "progress_inputs"):
            sample_inputs = inputs.get(key) or {}
            resample_attempts = sample_inputs.get("resample_attempts")
            if resample_attempts is None:
                continue

            if torch.is_tensor(resample_attempts):
                attempts_tensor = resample_attempts.to(self.accelerator.device, dtype=torch.float32).view(-1)
            else:
                attempts_tensor = torch.tensor(
                    resample_attempts, dtype=torch.float32, device=self.accelerator.device
                ).view(-1)

            if attempts_tensor.numel() == 0:
                continue

            sample_category = key.replace("_inputs", "")
            strategies = sample_inputs.get("data_gen_strategy")
            if strategies is None:
                raise ValueError(
                    f"Expected data_gen_strategy for {sample_category} samples when logging resample attempts."
                )

            if len(strategies) != attempts_tensor.numel():
                raise ValueError(
                    f"Mismatch between resample attempts ({attempts_tensor.numel()}) and strategies "
                    f"({len(strategies)}) for {sample_category} samples."
                )

            strategy_labels = [f"{sample_category}/{str(strategy)}" for strategy in strategies]

            for attempt_value, strategy_label in zip(attempts_tensor.tolist(), strategy_labels):
                local_pairs.append((strategy_label, float(attempt_value)))

        if dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_lists: List[List[Tuple[str, float]]] = [None] * world_size
            dist.all_gather_object(gathered_lists, local_pairs)
            flat_pairs = [pair for proc_pairs in gathered_lists for pair in proc_pairs]
        else:
            flat_pairs = local_pairs

        if not flat_pairs:
            return

        all_attempts = [attempt for _, attempt in flat_pairs]
        self.log_metadata["data/resample_min"] = float(min(all_attempts))
        self.log_metadata["data/resample_max"] = float(max(all_attempts))
        self.log_metadata["data/resample_mean"] = float(sum(all_attempts) / len(all_attempts))

        strategy_values: Dict[str, List[float]] = collections.defaultdict(list)
        for label, attempt in flat_pairs:
            strategy_values[label].append(attempt)

        for label, values in strategy_values.items():
            if not values:
                continue
            safe_label = label.replace("/", "_").replace(" ", "_")
            strategy_min = float(min(values))
            strategy_max = float(max(values))
            strategy_mean = float(sum(values) / len(values))
            self.log_metadata[f"data/resample_min_{safe_label}"] = strategy_min
            self.log_metadata[f"data/resample_max_{safe_label}"] = strategy_max
            self.log_metadata[f"data/resample_mean_{safe_label}"] = strategy_mean

    def _log_metadata(self):
        """Log custom RBM losses to wandb and console."""
        if not self.log_metadata:
            return

        logger.trace("logging metadata, starting to aggregate metrics")

        # Use local metrics (no aggregation needed for individual GPU metrics)
        log_metadata = reduce_metrics_with_accelerate(self.log_metadata, self.accelerator, aggregate_method="mean")

        logger.trace("finished aggregating metrics")

        training_step_time = self.timing_raw.get("time/training_step", 0.0)
        it_per_sec = 1.0 / training_step_time if training_step_time > 0 else 0.0

        # Prepare logging data using aggregated losses
        log_data = {
            "step": self.state.global_step,
            "epoch": self.state.epoch,
            "train/it_per_sec": it_per_sec,
            **self.timing_raw,
            **log_metadata,
        }

        # Log global metadata
        logger.trace("logging global metadata")
        global_metadata = reduce_metrics_with_accelerate(self.global_metadata, self.accelerator, aggregate_method="sum")
        logger.trace("finished aggregating global metadata")

        # Convert counts to fractions of total samples
        total_samples = global_metadata["total_samples"]
        log_global = {
            f"counts/{key}": value / total_samples for key, value in global_metadata.items() if key != "total_samples"
        }

        log_data.update(log_global)

        # Log optimizer and gradient statistics
        optim_stats = self._get_optimizer_stats()
        log_data.update(optim_stats)

        # make sure values are floats so they are loggable into wandb reports
        log_data = {k: float(v) for k, v in log_data.items()}

        self.logger.log_scalars(log_data, step=self.state.global_step + 1)

        if is_rank_0():
            logger.info(f"Step {self.state.global_step}, Epoch {self.state.epoch:.2f}:")
            logger.info("-" * 50)
            logger.info(f"  train/it_per_sec: {it_per_sec:.4f}")
            for key in log_global:
                logger.info(f"  {key}: {log_global[key]}")

            rounded_times = {k: round(v, 2) for k, v in self.timing_raw.items()}
            logger.info(f"Timing raw: {rounded_times}")

            # Log optimizer stats to console
            # if optim_stats:
            #     logger.info(f"Optimizer stats: {optim_stats}")

    def _make_eval_dataloader(self, dataset):
        """Create a distributed evaluation dataloader with proper sampling."""
        collator = setup_batch_collator(self.model.processor, self.model.tokenizer, self.config, is_eval=True)

        dl = DataLoader(
            dataset,
            batch_size=self.config.training.per_device_eval_batch_size,
            collate_fn=collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=False,
            # Force persistent_workers=False for eval to prevent memory leaks across datasets
            persistent_workers=False,
            worker_init_fn=seed_worker,
        )
        prepared_dl = self.accelerator.prepare(dl)
        return prepared_dl

    def _setup_eval_dataset(self, eval_type, eval_dataset):
        """Setup dataset and dataloader for evaluation."""
        eval_cfg = copy.deepcopy(self.config.data)

        # explicitly set dataset type to rbm for custom eval datasets
        eval_cfg.dataset_type = "rbm"

        if isinstance(eval_dataset, list):
            eval_cfg.eval_datasets = eval_dataset
        else:
            eval_cfg.eval_datasets = [eval_dataset]

        # Create custom eval dataset with the appropriate sampler
        sampler_kwargs = {}
        sampler_kwargs["random_seed"] = self.config.custom_eval.custom_eval_random_seed

        if eval_type == "reward_alignment":
            sampler_kwargs["max_trajectories"] = self.config.custom_eval.reward_alignment_max_trajectories
            sampler_kwargs["frame_step"] = (
                2 if (self.config.trainer_cls == "rbm_heads" and not self.config.data.use_multi_image) else 1
            )
            sampler_kwargs["use_frame_steps"] = self.config.custom_eval.use_frame_steps
        elif eval_type == "policy_ranking":
            sampler_kwargs["num_examples_per_quality_pr"] = self.config.custom_eval.num_examples_per_quality_pr
            sampler_kwargs["num_partial_successes"] = self.config.custom_eval.num_partial_successes
            sampler_kwargs["max_tasks"] = self.config.custom_eval.policy_ranking_max_tasks
            sampler_kwargs["frame_step"] = (
                2 if (self.config.trainer_cls == "rbm_heads" and not self.config.data.use_multi_image) else 1
            )
            # sampler_kwargs["use_frame_steps"] = self.config.custom_eval.use_frame_steps
            # we only care about the final predicted progress for ranking
            sampler_kwargs["use_frame_steps"] = False
        elif eval_type == "quality_preference":
            sampler_kwargs["comparisons_per_task"] = self.config.custom_eval.comparisons_per_task
            sampler_kwargs["max_comparisons"] = self.config.custom_eval.max_comparisons

        dataset = setup_custom_eval_dataset(
            eval_cfg, sampler_type=eval_type, verbose=False, sampler_kwargs=sampler_kwargs
        )
        # Explicitly delete eval_cfg after dataset creation to free memory
        del eval_cfg

        logger.info(f"  Dataset size: {len(dataset)}")
        # log_memory_usage(f"After creating dataset")

        dataloader = self._make_eval_dataloader(dataset)
        logger.info(f"  Dataloader created with {len(dataloader)} batches")
        # log_memory_usage(f"After creating dataloader")

        # Ensure model is in eval mode and clear any gradient buffers
        self.model.eval()
        # Explicitly zero any gradients that might exist (shouldn't, but safety measure)
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        # Clear cache before starting evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # log_memory_usage(f"After clearing cache, before eval loop")

        return dataset, dataloader

    def _process_batch_progress_eval(self, batch, eval_type):
        """Process a batch for progress-based evaluations (reward_alignment, policy_ranking, confusion_matrix)."""
        logger.trace(f"    Processing {eval_type} batch")
        progress_samples = batch["progress_inputs"]
        logger.trace(f"    Calling forward_model for progress")
        with torch.no_grad():
            outputs, _ = self.forward_model(self.model, progress_samples, sample_type="progress")
        logger.trace(f"    Forward pass complete")

        progress_logits = outputs.progress_logits
        progress_pred = progress_logits["A"]

        # Gather everything
        progress_pred = self.accelerator.gather_for_metrics(progress_pred)
        target_progress = self.accelerator.gather_for_metrics(progress_samples["target_progress"])

        # Gather metadata fields
        metadata_fields = [
            "task",
            "data_source",
            "data_gen_strategy",
            "quality_labels",
            "metadata",
            "partial_success",
        ]
        gathered_metadata_dict = self._gather_metadata_fields(progress_samples, metadata_fields)
        num_progress_samples = progress_pred.shape[0] if progress_pred is not None else 0
        gathered_metadata_dict = self._truncate_metadata_lists(gathered_metadata_dict, num_progress_samples)

        # Handle success predictions if needed
        success_pred_gathered = None
        success_probs_gathered = None
        success_labels_gathered = None
        if self.config.model.train_success_head:
            success_pred = outputs.success_logits["A"]
            success_probs = torch.sigmoid(success_pred)
            success_binary = (success_probs > 0.5).float()
            success_labels = progress_samples.get("success_labels")
            success_pred_gathered = self.accelerator.gather_for_metrics(success_binary)
            success_probs_gathered = self.accelerator.gather_for_metrics(success_probs)
            success_labels_gathered = self.accelerator.gather_for_metrics(success_labels)

            # Clean up intermediate tensors (but keep gathered tensors for eval_results)
            del (
                success_pred,
                success_binary,
                success_probs,
                success_labels,
            )

        # Build eval_results on all processes for compute_eval_metrics
        batch_results = []
        for i in range(len(progress_pred)):
            metadata = gathered_metadata_dict["metadata"][i]
            sample_result = {
                "task": gathered_metadata_dict["task"][i],
                "target_progress": t2n(target_progress[i]),
                "progress_pred": t2n(progress_pred[i]),
                "data_source": gathered_metadata_dict["data_source"][i],
                "data_gen_strategy": gathered_metadata_dict["data_gen_strategy"][i],
                "quality_label": gathered_metadata_dict["quality_labels"][i],
                "metadata": metadata,
                "id": metadata["id"],
                "video_path": metadata["video_path"],
                "partial_success": gathered_metadata_dict["partial_success"][i],
            }
            if success_pred_gathered is not None:
                sample_result["success_pred"] = t2n(success_pred_gathered[i])
            if success_probs_gathered is not None:
                sample_result["success_probs"] = t2n(success_probs_gathered[i])
            if success_labels_gathered is not None:
                sample_result["success_labels"] = t2n(success_labels_gathered[i])
            batch_results.append(sample_result)

        # Clean up gathered tensors and metadata after building results
        del progress_pred, target_progress, gathered_metadata_dict
        if success_pred_gathered is not None:
            del success_pred_gathered
        if success_probs_gathered is not None:
            del success_probs_gathered
        if success_labels_gathered is not None:
            del success_labels_gathered

        return batch_results, outputs

    def _process_batch_preference_eval(self, batch):
        """Process a batch for preference-based evaluations (quality_preference)."""
        logger.trace(f"    Processing quality_preference batch")
        preference_samples = batch["preference_inputs"]
        logger.trace(f"    Calling forward_model for preference")
        with torch.no_grad():
            outputs, _ = self.forward_model(self.model, preference_samples, sample_type="preference")
        logger.trace(f"    Forward pass complete")
        pref_logits = outputs.pref_logits

        # Gather predictions and labels across all ranks
        pref_logits = self.accelerator.gather_for_metrics(pref_logits)
        preference_labels = self.accelerator.gather_for_metrics(preference_samples["preference_labels"])

        # Convert logits to binary predictions (0/1): apply sigmoid, then threshold at 0.5
        pref_probs = torch.sigmoid(pref_logits)
        binary_preds = (pref_probs > 0.5).float()

        # Gather non-tensor metadata using helper (handles single and multi GPU)
        gathered_pref_metadata = self._gather_metadata_fields(
            preference_samples,
            [
                "task",
                "data_source",
                "chosen_data_gen_strategy",
                "rejected_data_gen_strategy",
                "metadata",
            ],
        )
        num_pref_samples = binary_preds.shape[0] if binary_preds is not None else 0
        gathered_pref_metadata = self._truncate_metadata_lists(gathered_pref_metadata, num_pref_samples)

        # Build eval_results on all processes for compute_eval_metrics
        batch_results = []
        for i in range(len(binary_preds)):
            if binary_preds[i] is None:
                continue
            sample_result = {
                "task": gathered_pref_metadata["task"][i],
                "preference_pred": t2n(binary_preds[i]),
                "preference_logits": t2n(pref_logits[i]),
                "preference_labels": t2n(preference_labels[i]),
                "data_source": gathered_pref_metadata["data_source"][i],
                "chosen_data_gen_strategy": gathered_pref_metadata["chosen_data_gen_strategy"][i],
                "rejected_data_gen_strategy": gathered_pref_metadata["rejected_data_gen_strategy"][i],
                "metadata": gathered_pref_metadata["metadata"][i],
            }
            batch_results.append(sample_result)

        # Clean up gathered tensors and metadata after building results
        del pref_logits, pref_probs, binary_preds, preference_labels, gathered_pref_metadata

        return batch_results, outputs

    def _compute_and_log_eval_metrics(self, eval_type, eval_results, ds_name, eval_step):
        """Compute metrics and create visualizations for evaluation results."""
        # Initialize variables to None to ensure they exist for cleanup
        plots = None
        video_frames_list = None
        trajectory_progress_data = None
        task_groups = None
        task_details = None
        confusion_plot = None
        confusion_matrix = None

        # if dataset name is too long, truncate it
        if len(ds_name) > 90:
            ds_name = ds_name[:90] + "..."

        is_discrete_mode = self.config.loss.progress_loss_type.lower() == "discrete"
        num_bins = self.config.loss.progress_discrete_bins if is_discrete_mode else None

        data_source = None
        if eval_results and len(eval_results) > 0:
            data_source = eval_results[0]["data_source"]

        if eval_type == "reward_alignment":
            eval_metrics, plots, video_frames_list, trajectory_progress_data = run_reward_alignment_eval_per_trajectory(
                eval_results,
                self.config.data.progress_pred_type,
                is_discrete_mode,
                num_bins,
                data_source,
                use_frame_steps=self.config.custom_eval.use_frame_steps,
                train_success_head=self.config.model.train_success_head,
                last_frame_only=False,
            )
            # log_memory_usage(f"After compute_eval_metrics (reward_alignment)")

            banner(
                f"{eval_type} evaluation: {len(eval_results)} samples",
                f"Metrics: {eval_metrics}",
                inner_padding=1,
            )

            # Build rows of (video, figure)
            rows = []
            for plot, frames in zip(plots, video_frames_list):
                if frames is not None:
                    rows.append((frames, plot))

            if video_frames_list and plots:
                # Log individual images to wandb: first frame + plot side-by-side
                if self.logger.enabled("wandb"):
                    # Filter valid pairs and limit to 10
                    valid_pairs = [
                        (frames, plot)
                        for frames, plot in zip(video_frames_list, plots)
                        if frames is not None and plot is not None
                    ]
                    valid_pairs = valid_pairs[:10]

                    combined_images = []
                    for idx, (frames, plot) in enumerate(valid_pairs):
                        # Convert frames from (T, C, H, W) to (T, H, W, C) for display
                        if len(frames.shape) == 4 and frames.shape[1] == 3:
                            frames_rgb = frames.transpose(0, 2, 3, 1)
                        else:
                            frames_rgb = frames

                        # Ensure frames are uint8 in [0, 255]
                        if frames_rgb.max() <= 1.0:
                            frames_rgb = (frames_rgb * 255).astype(np.uint8)
                        else:
                            frames_rgb = np.clip(frames_rgb, 0, 255).astype(np.uint8)

                        # Convert plot to image at original resolution
                        plot_fig = plot
                        buf = io.BytesIO()
                        plot_fig.savefig(buf, format="png", dpi=plot_fig.dpi, bbox_inches="tight")
                        buf.seek(0)
                        from PIL import Image

                        plot_img_pil = Image.open(buf)
                        # Convert RGBA to RGB if necessary
                        if plot_img_pil.mode == "RGBA":
                            plot_img_pil = plot_img_pil.convert("RGB")
                        plot_img = np.array(plot_img_pil)
                        buf.close()

                        # Get plot dimensions
                        plot_h, plot_w = plot_img.shape[:2]

                        # Get first and last frames and resize to match plot height
                        first_frame = frames_rgb[0]
                        last_frame = frames_rgb[-1]

                        # Calculate width to maintain aspect ratio, or use plot height
                        frame_h, frame_w = first_frame.shape[:2]
                        aspect_ratio = frame_w / frame_h
                        video_h = plot_h
                        video_w = int(plot_h * aspect_ratio)

                        # Resize first frame
                        first_frame_pil = Image.fromarray(first_frame)
                        first_frame_resized = first_frame_pil.resize((video_w, video_h), Image.Resampling.LANCZOS)
                        first_frame_resized = np.array(first_frame_resized)

                        # Resize last frame
                        last_frame_pil = Image.fromarray(last_frame)
                        last_frame_resized = last_frame_pil.resize((video_w, video_h), Image.Resampling.LANCZOS)
                        last_frame_resized = np.array(last_frame_resized)

                        # Combine first frame, last frame, and plot side-by-side
                        combined_image = np.hstack([first_frame_resized, last_frame_resized, plot_img])
                        combined_images.append(combined_image)

                        # Log individual image to wandb
                        tag = f"reward_alignment_samples/{ds_name}/reward_sample_{idx}"
                        self.logger.log_image(tag, combined_image, step=eval_step)

                    # Create combined figure with all samples stacked vertically
                    if combined_images:
                        # Find the maximum width to ensure all images have the same width
                        max_width = max(img.shape[1] for img in combined_images)
                        # Resize all images to have the same width
                        resized_combined = []
                        for img in combined_images:
                            if img.shape[1] != max_width:
                                img_pil = Image.fromarray(img)
                                # Maintain aspect ratio by calculating new height
                                aspect = img.shape[0] / img.shape[1]
                                new_height = int(max_width * aspect)
                                img_resized = img_pil.resize((max_width, new_height), Image.Resampling.LANCZOS)
                                resized_combined.append(np.array(img_resized))
                            else:
                                resized_combined.append(img)

                        # Stack all images vertically
                        combined_figure = np.vstack(resized_combined)

                        # Log combined figure to wandb
                        combined_tag = f"reward_alignment_samples/{ds_name}/all_samples_combined"
                        self.logger.log_image(combined_tag, combined_figure, step=eval_step)

            # if rows and self.logger.enabled("wandb"):
            #     self.logger.log_video_table(
            #         f"reward_alignment_samples/{ds_name}",
            #         videos_and_figures=rows,
            #         columns=["video", "progress_plot"],
            #         step=eval_step,
            #     )

            # # Create and log 3x3 grid of videos with progress overlays
            # if video_frames_list and self.logger.enabled("wandb"):
            #     grid_video = create_video_grid_with_progress(
            #         video_frames_list,
            #         trajectory_progress_data,
            #         grid_size=(3, 3),
            #         max_videos=9,
            #         is_discrete_mode=is_discrete_mode,
            #     )
            #     if grid_video is not None:
            #         self.logger.log_video(
            #             f"reward_alignment_grid/{ds_name}",
            #             grid_video,
            #             fps=2,
            #             step=eval_step,
            #         )
            #         del grid_video

            # For tensorboard (no table support), log each video and its figure separately
            # if self.logger.enabled("tensorboard"):
            #     for idx, frames in enumerate(video_frames_list):
            #         if frames is not None:
            #             self.logger.log_video(
            #                 f"reward_alignment_video/{ds_name}/{idx}",
            #                 frames,
            #                 fps=2,
            #                 step=eval_step,
            #             )
            #     for idx, plot in enumerate(plots):
            #         self.logger.log_figure(f"{ds_name}/reward_alignment_plot/{idx}", plot, step=eval_step)

            # Close all plots to avoid accumulating open figures
            for plot in plots:
                plt.close(plot)

            # Explicitly delete to free memory and set to None for outer cleanup
            # log_memory_usage(f"Before deleting plots/videos")
            del plots, video_frames_list, trajectory_progress_data, rows
            plots = None
            video_frames_list = None
            trajectory_progress_data = None
            # log_memory_usage(f"After deleting plots/videos")
        elif eval_type == "policy_ranking":
            eval_metrics, task_groups, task_details = run_policy_ranking_eval(
                eval_results,
                self.config.data.progress_pred_type,
                is_discrete_mode,
                num_bins,
                data_source,
                correlation_method="kendall",
            )
            # log_memory_usage(f"After compute_eval_metrics (policy_ranking)")

            banner(
                f"{eval_type} evaluation: {len(eval_results)} samples",
                f"Metrics: {eval_metrics}",
                inner_padding=1,
            )

            # Check if any trajectory has partial_success to determine visualization type
            use_partial_success = False
            if task_groups:
                # Check first task group to see if it has partial_success
                first_group = next(iter(task_groups.values()))
                if first_group and len(first_group) > 0:
                    use_partial_success = first_group[0].get("partial_success") is not None

            data = []
            if use_partial_success:
                # Visualization for datasets with partial_success: show partial vs predicted rewards for each aggregation type
                for task, group in task_groups.items():
                    partial_successes = np.array([t["partial_success"] for t in group]).round(2)
                    predicted_rewards_last = np.array([t["final_predicted_reward_last"] for t in group]).round(2)
                    predicted_rewards_avg = np.array([t["final_predicted_reward_avg"] for t in group]).round(2)
                    predicted_rewards_sum = np.array([t["final_predicted_reward_sum"] for t in group]).round(2)
                    partial_successes = partial_successes.tolist()
                    predicted_rewards_last = predicted_rewards_last.tolist()
                    predicted_rewards_avg = predicted_rewards_avg.tolist()
                    predicted_rewards_sum = predicted_rewards_sum.tolist()
                    data.append([
                        task,
                        f"partial:{partial_successes}",
                        f"predicted_last:{predicted_rewards_last}",
                        f"predicted_avg:{predicted_rewards_avg}",
                        f"predicted_sum:{predicted_rewards_sum}",
                    ])
                columns = [
                    "task",
                    "partial_successes",
                    "predicted_rewards_last",
                    "predicted_rewards_avg",
                    "predicted_rewards_sum",
                ]
            else:
                # Standard policy ranking visualization: show quality labels and rewards for each aggregation type
                for task, group in task_groups.items():
                    quality_to_rews_last = collections.defaultdict(list)
                    quality_to_rews_avg = collections.defaultdict(list)
                    quality_to_rews_sum = collections.defaultdict(list)
                    for t in group:
                        rew_last = t["final_predicted_reward_last"]
                        rew_avg = t["final_predicted_reward_avg"]
                        rew_sum = t["final_predicted_reward_sum"]
                        quality_label = t["quality_label"]
                        quality_to_rews_last[quality_label].append(rew_last)
                        quality_to_rews_avg[quality_label].append(rew_avg)
                        quality_to_rews_sum[quality_label].append(rew_sum)

                    for q, r in quality_to_rews_last.items():
                        quality_to_rews_last[q] = np.array(r).round(2).tolist()
                    for q, r in quality_to_rews_avg.items():
                        quality_to_rews_avg[q] = np.array(r).round(2).tolist()
                    for q, r in quality_to_rews_sum.items():
                        quality_to_rews_sum[q] = np.array(r).round(2).tolist()

                    quality_to_rews_last_str = ",".join([f"{q}:{r}" for q, r in quality_to_rews_last.items()])
                    quality_to_rews_avg_str = ",".join([f"{q}:{r}" for q, r in quality_to_rews_avg.items()])
                    quality_to_rews_sum_str = ",".join([f"{q}:{r}" for q, r in quality_to_rews_sum.items()])

                    # Get task details for differences (using last aggregation for differences)
                    task_detail = task_details.get(task, {})
                    succ_subopt = task_detail.get("succ_subopt_diff")
                    subopt_fail = task_detail.get("subopt_fail_diff")
                    succ_fail = task_detail.get("succ_fail_diff")

                    # Format differences
                    diff_str = []
                    if succ_subopt is not None:
                        diff_str.append(f"succ-subopt:{succ_subopt:.2f}")
                    if subopt_fail is not None:
                        diff_str.append(f"subopt-fail:{subopt_fail:.2f}")
                    if succ_fail is not None:
                        diff_str.append(f"succ-fail:{succ_fail:.2f}")
                    diff_str = ",".join(diff_str) if diff_str else "N/A"

                    data.append([
                        task,
                        quality_to_rews_last_str,
                        quality_to_rews_avg_str,
                        quality_to_rews_sum_str,
                        diff_str,
                    ])

                columns = [
                    "task",
                    "quality_and_rews_last",
                    "quality_and_rews_avg",
                    "quality_and_rews_sum",
                    "avg_differences",
                ]

            table_name = f"policy_ranking_samples/{ds_name}"

            self.logger.log_table(
                table_name,
                data=data,
                columns=columns,
                step=eval_step,
            )

            # Save policy ranking samples as JSON metadata
            # Convert table data (list of lists) to list of dictionaries
            samples_metadata = []
            for row in data:
                sample_dict = {col: val for col, val in zip(columns, row)}
                samples_metadata.append(sample_dict)

            # Save to policy_ranking_samples folder
            output_dir = self.args.output_dir
            samples_dir = os.path.join(output_dir, "policy_ranking_samples", f"step_{eval_step}")
            os.makedirs(samples_dir, exist_ok=True)

            filename = f"{ds_name}.json"
            filepath = os.path.join(samples_dir, filename)

            with open(filepath, "w") as f:
                json.dump(samples_metadata, f, indent=2)
            logger.info(f"Saved {len(samples_metadata)} policy ranking samples to {filepath}")

            # # Create and log grid of frame pairs with progress annotations
            # if self.logger.enabled("wandb"):
            #     grid_image = create_policy_ranking_grid(
            #         eval_results, grid_size=(2, 2), max_samples=4, is_discrete_mode=is_discrete_mode
            #     )
            #     if grid_image is not None:
            #         self.logger.log_image(
            #             f"policy_ranking_grid/{ds_name}",
            #             grid_image,
            #             step=eval_step,
            #         )
            #         del grid_image

            # log_memory_usage(f"Before deleting policy_ranking data")
            del data, task_groups, task_details
            task_groups = None
            task_details = None
            # log_memory_usage(f"After deleting policy_ranking data")
        elif eval_type == "confusion_matrix":
            confusion_plot, confusion_matrix, eval_metrics = run_confusion_matrix_eval(
                eval_results, self.config.data.progress_pred_type, is_discrete_mode, num_bins
            )
            # log_memory_usage(f"After compute_eval_metrics (confusion_matrix)")

            banner(
                f"{eval_type} evaluation: {len(eval_results)} samples",
                f"Metrics: {eval_metrics}",
                inner_padding=1,
            )

            self.logger.log_figure(f"eval_cm/{ds_name}", confusion_plot, step=eval_step)
            plt.close(confusion_plot)
            # log_memory_usage(f"Before deleting confusion_matrix data")
            del confusion_plot, confusion_matrix
            confusion_plot = None
            confusion_matrix = None
            # log_memory_usage(f"After deleting confusion_matrix data")
        elif "quality_preference" in eval_type:
            eval_metrics, task_groups, task_details = run_quality_preference_eval(eval_results, data_source=data_source)
            # log_memory_usage(f"After compute_eval_metrics (quality_preference)")

            banner(
                "Completed evaluation",
                f"{eval_type} evaluation: {len(eval_results)} samples",
                "Metrics",
                f"{eval_metrics}",
                inner_padding=1,
            )

            data = []
            for task, details in task_details.items():
                task_acc = details["preference_accuracy"]
                quality_accs = details["quality_accuracies"]
                quality_accs_str = ",".join([f"{k}:{round(v, 3)}" for k, v in quality_accs.items()])
                num_correct = details["num_correct"]
                num_total = details["num_total"]
                data.append([
                    task,
                    round(task_acc, 3),
                    quality_accs_str if quality_accs_str else "N/A",
                    f"{num_correct}/{num_total}",
                ])
            columns = ["task", "preference_accuracy", "quality_accuracies", "num_correct/total"]

            table_name = f"quality_preference_samples/{ds_name}"

            self.logger.log_table(
                table_name,
                data=data,
                columns=columns,
                step=eval_step,
            )

            # Save quality preference samples as JSON metadata
            # Convert table data (list of lists) to list of dictionaries
            samples_metadata = []
            for row in data:
                sample_dict = {col: val for col, val in zip(columns, row)}
                samples_metadata.append(sample_dict)

            # Save to quality_preference_samples folder
            output_dir = self.args.output_dir
            samples_dir = os.path.join(output_dir, "quality_preference_samples", f"step_{eval_step}")
            os.makedirs(samples_dir, exist_ok=True)

            filename = f"{ds_name}.json"
            filepath = os.path.join(samples_dir, filename)

            with open(filepath, "w") as f:
                json.dump(samples_metadata, f, indent=2)
            logger.info(f"Saved {len(samples_metadata)} quality preference samples to {filepath}")

            # log_memory_usage(f"Before deleting quality_preference data")
            del data, task_groups, task_details
            task_groups = None
            task_details = None
            # log_memory_usage(f"After deleting quality_preference data")
        else:
            raise ValueError(f"Unsupported eval type: {eval_type}")

        # Clean up eval-specific outputs
        if plots is not None:
            del plots
        if video_frames_list is not None:
            del video_frames_list
        if trajectory_progress_data is not None:
            del trajectory_progress_data
        if task_groups is not None:
            del task_groups
        if task_details is not None:
            del task_details
        if confusion_plot is not None:
            del confusion_plot
        if confusion_matrix is not None:
            del confusion_matrix

        return eval_metrics

    def _save_eval_results_json(self, eval_results, eval_type, ds_name):
        """Save eval_results as JSON file.

        Args:
            eval_results: List of evaluation result dictionaries
            eval_type: Type of evaluation (e.g., "reward_alignment", "policy_ranking")
            ds_name: Dataset name
        """

        def serialize_value(value):
            """Recursively serialize a value to JSON-compatible format."""
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                return float(value)
            elif isinstance(value, (np.bool_, bool)):
                return bool(value)
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [serialize_value(v) for v in value]
            elif isinstance(value, (int, float, str, type(None))):
                return value
            else:
                # Fallback: try to convert to string
                return str(value)

        # Serialize eval_results
        serialized_results = [serialize_value(result) for result in eval_results]

        # Create output directory if it doesn't exist
        output_dir = self.args.output_dir
        eval_results_dir = os.path.join(output_dir, "eval_results")
        os.makedirs(eval_results_dir, exist_ok=True)

        # Create filename: {eval_type}_{ds_name}.json
        filename = f"{eval_type}_{ds_name}.json"
        filepath = os.path.join(eval_results_dir, filename)

        # Save to JSON file
        with open(filepath, "w") as f:
            json.dump(serialized_results, f, indent=2)
        logger.info(f"Saved {len(eval_results)} eval results to: {filepath}")

    def _cleanup_eval_dataset(self, dataset, dataloader, eval_results):
        """Clean up dataset, dataloader, and eval_results after evaluation."""
        logger.info(f"  [Rank {get_rank()}] Cleaning up dataset and eval_results")
        # log_memory_usage(f"Before cleanup")

        # Aggressive cleanup to prevent memory leaks
        # First, delete eval_results which can be large
        logger.debug(f"  [Rank {get_rank()}] Deleting eval_results")
        del eval_results

        # For the dataloader, we need to ensure worker processes are shut down
        # The accelerator.prepare() wraps the dataloader, so we need to clean both
        # Access the underlying dataloader if it exists and has workers
        logger.debug(f"  [Rank {get_rank()}] Shutting down dataloader workers")
        try:
            if hasattr(dataloader, "_loader"):
                # Accelerator-wrapped dataloader
                underlying_dl = dataloader._loader
            else:
                underlying_dl = dataloader

            # Shutdown workers if they exist
            if hasattr(underlying_dl, "_iterator") and underlying_dl._iterator is not None:
                logger.debug(f"  [Rank {get_rank()}] Calling _shutdown_workers()")
                underlying_dl._iterator._shutdown_workers()
                underlying_dl._iterator = None
                logger.debug(f"  [Rank {get_rank()}] Workers shut down successfully")
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"  [Rank {get_rank()}] Could not explicitly shutdown workers: {e}")

        # Delete dataloader and dataset
        logger.debug(f"  [Rank {get_rank()}] Deleting dataloader and dataset")
        del dataloader, dataset
        # log_memory_usage(f"After deleting dataloader and dataset")

        # Force garbage collection
        import gc

        logger.debug(f"  [Rank {get_rank()}] Running garbage collection")
        gc.collect()
        # log_memory_usage(f"After first gc.collect()")
        gc.collect()  # Call twice for cyclic references
        # log_memory_usage(f"After second gc.collect()")

        # Clear GPU cache
        if torch.cuda.is_available():
            logger.debug(f"  [Rank {get_rank()}] Clearing CUDA cache")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.debug(f"  [Rank {get_rank()}] Cleanup complete")

    def _run_single_eval_dataset(self, eval_type, eval_dataset, eval_step):
        """Run evaluation for a single dataset."""
        logger.info(f"  Processing dataset: {eval_dataset}")
        # log_memory_usage(f"Before dataset {eval_dataset}")

        dataset_for_mapping = eval_dataset[0] if isinstance(eval_dataset, list) else eval_dataset
        ds_name = DS_SHORT_NAME_MAPPING.get(dataset_for_mapping, dataset_for_mapping)
        timing_key = f"time/eval_dataset/{eval_type}/{ds_name}"

        with _timer(timing_key, timing_raw=self.timing_raw):
            # Setup dataset and dataloader
            dataset, dataloader = self._setup_eval_dataset(eval_type, eval_dataset)

            eval_results = []
            batch_idx = 0
            # Create tqdm iterator explicitly so we can close it properly
            dataloader_iter = tqdm(
                dataloader,
                desc=f"Running {eval_type}, ds: {eval_dataset}, batch size: {self.config.training.per_device_eval_batch_size}",
                disable=not is_rank_0(),
            )

            for batch in dataloader_iter:
                logger.trace(f"  Processing batch {batch_idx}")
                # if batch_idx % 10 == 0:  # Log memory every 10 batches
                #     log_memory_usage(f"Batch {batch_idx}/{len(dataloader)}")

                batch = self._prepare_inputs(batch)

                # Log batch composition
                num_pref = batch.get("num_preferences", 0)
                num_prog = batch.get("num_progress", 0)
                num_sim = batch.get("num_similarities", 0)
                logger.trace(f"  Batch {batch_idx}: pref={num_pref}, prog={num_prog}, sim={num_sim}")
                batch_idx += 1

                # Process batch based on eval type
                if eval_type in ["reward_alignment", "policy_ranking", "confusion_matrix"]:
                    batch_results, outputs = self._process_batch_progress_eval(batch, eval_type)
                    eval_results.extend(batch_results)
                elif "quality_preference" in eval_type:
                    batch_results, outputs = self._process_batch_preference_eval(batch)
                    eval_results.extend(batch_results)
                # Clean up batch tensors and free memory after each batch
                # Free memory after each batch to prevent OOM
                del batch, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # # Log memory after cleanup every 10 batches
                # if (batch_idx - 1) % 10 == 0:
                #     log_memory_usage(f"After batch {batch_idx - 1} cleanup")

            # Close tqdm iterator to release any held references
            dataloader_iter.close()
            del dataloader_iter

            logger.info(f"  [Rank {get_rank()}] Finished processing {len(eval_results)} eval results")
            # log_memory_usage(f"After eval loop, before compute_eval_metrics")

            # Compute metrics and create visualizations (only on main process)
            eval_metrics = {}
            if self.accelerator.is_main_process:
                logger.info(f"  [Rank {get_rank()}] Starting metric computation")
                eval_metrics = self._compute_and_log_eval_metrics(eval_type, eval_results, ds_name, eval_step)

                # Save eval_results as JSON
                self._save_eval_results_json(eval_results, eval_type, ds_name)
                logger.info(f"  [Rank {get_rank()}] Finished metric computation")
            else:
                logger.info(f"  [Rank {get_rank()}] Skipping metric computation (not main process)")

            # Cleanup
            logger.info(f"  [Rank {get_rank()}] Starting dataset cleanup")
            self._cleanup_eval_dataset(dataset, dataloader, eval_results)
            logger.info(f"  [Rank {get_rank()}] Finished dataset cleanup")

            # log_memory_usage(f"After cleanup for {eval_dataset}")

            # Store timing for this eval_dataset
            eval_dataset_time = self.timing_raw.get(timing_key, 0.0)
            logger.info(
                f"  [Rank {get_rank()}] Finished {eval_type} for {eval_dataset} (took {eval_dataset_time:.2f} seconds)"
            )
            logger.info("-" * 80)

            return eval_metrics, ds_name

    def _run_custom_evaluations(self, eval_step=None):
        """
        Run custom evaluations.

        Args:
            eval_step: Step number to use for logging. If None, uses self.state.global_step.
                      This ensures consistent step logging to prevent wandb warnings.
        """
        if eval_step is None:
            eval_step = self.state.global_step

        logger.info("=" * 100)
        logger.info("STARTING CUSTOM EVALUATIONS")
        # log_memory_usage("Before custom evaluations")
        logger.info("=" * 100)

        metrics = collections.defaultdict(dict)
        eval_types = self.config.custom_eval.eval_types

        EVAL_TYPE_SHORT = {
            "reward_alignment": "rew_align",
            "confusion_matrix": "cm",
            "policy_ranking": "p_rank",
            "quality_preference": "pref",
            "quality_preference_roboarena": "pref_robo",
        }

        banner("Running custom evaluations", f"Custom evaluations: {eval_types}")

        eval_type_timings = {}
        eval_dataset_timings = {}

        for eval_type in eval_types:
            logger.info("=" * 80)
            logger.info(f"Running evaluation for: {eval_type}")
            # log_memory_usage(f"Before {eval_type}")
            logger.info("=" * 80)

            eval_datasets_name = getattr(self.config.custom_eval, eval_type)

            with _timer(f"time/eval_type/{eval_type}", timing_raw=self.timing_raw):
                for eval_dataset in eval_datasets_name:
                    eval_metrics, ds_name = self._run_single_eval_dataset(eval_type, eval_dataset, eval_step)
                    metrics[ds_name][eval_type] = eval_metrics

                    # Store timing for this eval_dataset
                    dataset_for_mapping = eval_dataset[0] if isinstance(eval_dataset, list) else eval_dataset
                    ds_name_mapped = DS_SHORT_NAME_MAPPING.get(dataset_for_mapping, dataset_for_mapping)
                    timing_key = f"time/eval_dataset/{eval_type}/{ds_name_mapped}"
                    eval_dataset_time = self.timing_raw.get(timing_key, 0.0)
                    eval_dataset_timings[timing_key] = eval_dataset_time

                # log_memory_usage(f"After completing all datasets for {eval_type}")

            eval_type_time = self.timing_raw.get(f"time/eval_type/{eval_type}", 0.0)
            eval_type_timings[f"time/eval_type/{eval_type}"] = eval_type_time
            logger.info(f"Finished eval_type: {eval_type} (took {eval_type_time:.2f} seconds)")
            logger.info("=" * 80)

        flat_metrics = {}
        for ds_name, eval_type_metric in metrics.items():
            for eval_type, metric in eval_type_metric.items():
                eval_type_short = EVAL_TYPE_SHORT[eval_type]
                # Add to flat metrics dict with full names
                for k, v in metric.items():
                    if isinstance(v, (int, float)):
                        metric_name = f"eval_{eval_type_short}/{k}_{ds_name}"
                        flat_metrics[metric_name] = v

        # Prepare metrics for callbacks (all processes should have the same metrics)
        callback_metrics = flat_metrics

        # Prepare wandb metrics and log (only on main process)
        if self.accelerator.is_main_process:
            # Convert callback_metrics to float for wandb logging
            to_log = {k: float(v) for k, v in callback_metrics.items()}
            to_log["epoch"] = self.state.epoch

            # Add timing metrics
            for timing_key, timing_value in eval_type_timings.items():
                to_log[timing_key] = float(timing_value)
            for timing_key, timing_value in eval_dataset_timings.items():
                to_log[timing_key] = float(timing_value)

            self.logger.log_scalars(to_log, step=eval_step)

            # Log timing summary to console
            if is_rank_0():
                logger.info("=" * 80)
                logger.info("Custom Evaluation Timing Summary")
                logger.info("=" * 80)
                logger.info("Per eval_type:")
                for timing_key, timing_value in sorted(eval_type_timings.items()):
                    logger.info(f"  {timing_key}: {timing_value:.2f} seconds")
                logger.info("Per eval_dataset:")
                for timing_key, timing_value in sorted(eval_dataset_timings.items()):
                    logger.info(f"  {timing_key}: {timing_value:.2f} seconds")
                logger.info("=" * 80)

        banner("Finished running custom evaluations!")
        # log_memory_usage("After all evaluations, before cleanup")

        # Reset model to training mode and clear any cached states to prevent leakage
        self.model.train()
        # Ensure gradients are cleared before returning to training
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

        # Aggressive cleanup to prevent OOM after evaluation
        import gc

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Call twice
        gc.collect()
        gc.collect()

        # Clean up large objects
        del metrics

        # log_memory_usage("After final cleanup")
        logger.info("=" * 100)
        logger.info("FINISHED CUSTOM EVALUATIONS")
        logger.info("=" * 100)

        # Final synchronization barrier to ensure all processes finish together
        if dist.is_initialized():
            logger.info(f"  [Rank {get_rank()}] Waiting at barrier in _run_custom_evaluations")
            dist.barrier()
            logger.info(f"  [Rank {get_rank()}] Passed barrier in _run_custom_evaluations")

        return callback_metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None) -> Dict[str, float]:
        """
        Override evaluate method to implement custom RBM evaluation metrics.
        """
        eval_step = self.state.global_step + 1

        # Save current training mode and set to eval mode
        was_training = self.model.training
        self.model.eval()
        metrics = {}

        # Run evaluation
        if self.config.training.run_default_eval:
            # Get the evaluation dataset
            eval_dataloader = self.get_eval_dataloader(eval_dataset)

            outputs = []
            with _timer("time/evaluate", timing_raw=self.timing_raw):
                with torch.no_grad():
                    for _step, inputs in tqdm(
                        enumerate(eval_dataloader),
                        total=len(eval_dataloader),
                        desc="Evaluating",
                        disable=not is_rank_0(),
                    ):
                        # Move inputs to device
                        inputs = self._prepare_inputs(inputs)

                        _, loss_dicts = self.compute_loss(self.model, inputs, return_outputs=True, training=False)
                        outputs.append(loss_dicts)

            # assume that we already called .item() on the outputs
            keys = list(outputs[0].keys())
            for key in keys:
                metrics[key] = [output[key] for output in outputs if key in output]
                metrics[key] = np.array(metrics[key]).mean()

            # Aggregate metrics across all processes using accelerator
            metrics = reduce_metrics_with_accelerate(metrics, self.accelerator, aggregate_method="mean")
            metrics["time/evaluate"] = self.timing_raw["time/evaluate"]

        # Run the custom evaluations
        custom_eval_should_run = (
            self.config.training.custom_eval_steps
            and self.state.global_step % self.config.training.custom_eval_steps == 0
        )
        if custom_eval_should_run:
            with _timer("time/custom_evaluations", timing_raw=self.timing_raw):
                custom_metrics = self._run_custom_evaluations(eval_step=eval_step)

            metrics.update(custom_metrics)
            # Add custom evaluation time
            metrics["time/custom_evaluations"] = self.timing_raw["time/custom_evaluations"]

            if is_rank_0():
                logger.info(f"Custom evaluations took {self.timing_raw['time/custom_evaluations']:.2f} seconds")

        if metrics:
            if is_rank_0():
                banner("Evaluation Results (Aggregated)", inner_padding=1)
                for key, value in metrics.items():
                    logger.info(f"{key}: {value:.6f}")
                logger.info("=" * 50)

            if is_rank_0():
                self.logger.log_scalars(metrics, step=eval_step)

            # Trigger the callback handler with all metrics
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        # CRITICAL: Final barrier OUTSIDE the if metrics block to ensure ALL ranks wait
        # This is the absolute final barrier before returning from evaluate(), ensuring no training can start
        # until all evaluation is completely done, regardless of whether metrics were computed
        if dist.is_initialized():
            logger.debug(f"[Rank {get_rank()}] Waiting at final barrier before returning from evaluate()")
            dist.barrier()
            logger.debug(f"[Rank {get_rank()}] Passed final barrier, about to return from evaluate()")

        # Restore original training mode to prevent state leakage
        self.model.train(was_training)
        # Ensure gradients are cleared before returning to training
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        # Clear any cached states that might persist from evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()

        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, training=True, **kwargs):
        """Compute loss for separate preference and progress batches."""
        logger.trace("compute_loss: Starting")

        # Set static graph for DDP on first training step to handle multiple forward passes
        # Preference and progress losses are computed in separate forward passes.
        if (
            training
            and not self._ddp_static_graph_set
            and getattr(self.accelerator.gradient_state, "sync_gradients", True)
            and hasattr(model, "module")
        ):
            if hasattr(model.module, "_set_static_graph"):
                logger.info("Setting DDP static graph mode for multiple forward passes")
                model.module._set_static_graph()
                self._ddp_static_graph_set = True
            elif hasattr(model, "_set_static_graph"):
                logger.info("Setting DDP static graph mode for multiple forward passes")
                model._set_static_graph()
                self._ddp_static_graph_set = True

        # Extract the separate batches
        preference_inputs = inputs.get("preference_inputs", {})
        progress_inputs = inputs.get("progress_inputs", {})

        num_preferences = inputs.get("num_preferences", 0)
        num_progress = inputs.get("num_progress", 0)

        total_loss = 0
        log_metadata = {}

        logger.trace(f"Num preferences: {num_preferences}, Num progress: {num_progress}")

        # Compute preference loss if we have preference samples
        if num_preferences > 0 and preference_inputs and self.config.model.train_preference_head:
            with _timer("time/compute_preference_loss", timing_raw=self.timing_raw):
                preference_loss, loss_dict = self._compute_preference_loss(
                    model, preference_inputs, return_outputs=True, training=training
                )
                if not torch.isnan(preference_loss).any():
                    total_loss += preference_loss
                else:
                    logger.warning(f"NaN detected in preference loss, replacing with 0.0")
                log_metadata.update(loss_dict)

        # Compute progress loss if we have progress samples
        if num_progress > 0 and progress_inputs and self.config.model.train_progress_head:
            with _timer("time/compute_progress_loss", timing_raw=self.timing_raw):
                progress_loss, loss_dict = self._compute_progress_loss(
                    model, progress_inputs, return_outputs=True, training=training
                )
                if not torch.isnan(progress_loss).any():
                    total_loss += progress_loss
                else:
                    logger.warning(f"NaN detected in progress loss, replacing with 0.0")
                log_metadata.update(loss_dict)

        for key, value in log_metadata.items():
            logger.trace(f"{key}: {value}, type: {type(value)}")
            if isinstance(value, torch.Tensor):
                logger.trace(f"\t{key}: shape={value.shape}")
        # Check for NaN in total loss before returning
        if torch.isnan(total_loss).any():
            logger.warning(f"NaN detected in total_loss, replacing with 0.0")
            total_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)

        # Always store custom losses for logging (even when return_outputs=False)
        self.log_metadata = log_metadata

        if return_outputs:
            # Combine outputs from all loss functions
            extra_info = {**log_metadata, "total_loss": total_loss.item()}
            return total_loss, extra_info

        return total_loss

    def _compute_success_loss_helper(
        self, success_logits, target_progress, success_labels, progress_loss_mask=None, quality_labels=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Helper function to compute success prediction loss.

        Computes binary cross-entropy loss for frames with:
        - progress < min_success (label=0, failure)
        - progress > max_success (label=1, success)
        - ignores frames in between

        The loss is automatically balanced by applying a weight to the minority class
        (whichever has fewer samples - positives or negatives) so both classes contribute equally.

        Args:
            success_logits: Success prediction logits (can be tensor or list of tensors)
            target_progress: Target progress tensors (can be tensor or list of tensors)
            success_labels: Success labels from batch (computed in collator) (can be tensor or list of tensors)
            progress_loss_mask: Per-sample mask tensor of shape (batch_size,) with 1.0 for samples
                where we should compute progress/success loss (e.g., successful, rewound, different_task)
            quality_labels: Optional list of quality labels (e.g., "successful", "suboptimal", "failure") for each sample.
                If a trajectory has quality_label in ["suboptimal", "failure", "failed"], we always predict success=0
                and verify that success_labels are all 0s.

        Returns:
            tuple: (success_loss, success_accuracy, success_auprc, metrics)
                   The loss is already balanced via per-sample weighting of the minority class
        """
        # Get base thresholds from config
        min_success = self.config.data.min_success

        # Handle Qwen/Molmo downsampling: take every 2nd frame if using Qwen/Molmo and NOT using multi_image
        # In multi_image mode, we already get one embedding per frame, so no downsampling needed
        # Ensure success_logits matches target_progress length after downsampling
        if (
            "Qwen" in self.config.model.base_model_id or "Molmo" in self.config.model.base_model_id
        ) and not self.config.data.use_multi_image:
            success_logits = success_logits[:, ::2]
            target_progress = target_progress[:, ::2]
            success_labels = success_labels[:, ::2]

        # Handle suboptimal/failure trajectories: always predict success=0 and verify labels are all 0s
        # Create a mask for suboptimal/failure trajectories: always include all frames for these trajectories
        quality_mask = None
        if quality_labels is not None:
            batch_size = success_logits.shape[0]
            seq_len = success_logits.shape[1]
            quality_mask = torch.zeros(batch_size, seq_len, device=success_logits.device, dtype=torch.float32)

            for i in range(batch_size):
                quality_label = quality_labels[i]
                if quality_label is not None and quality_label.lower() in ("suboptimal", "failure", "failed"):
                    # Verify that success_labels are all 0s for this trajectory
                    sample_success_labels = success_labels[i]
                    if not (sample_success_labels == 0.0).all():
                        logger.debug(
                            f"Trajectory {i} has quality_label='{quality_label}' but success_labels are not all 0s. "
                            f"Found non-zero labels: {(sample_success_labels != 0.0).sum().item()} out of {len(sample_success_labels)}"
                        )
                        import ipdb

                        ipdb.set_trace()

                    # Include all frames for this trajectory in the mask
                    quality_mask[i, :] = 1.0

        if self.config.loss.progress_loss_type.lower() == "discrete":
            target_progress = convert_discrete_target_to_continuous(
                target_progress, num_bins=self.config.loss.progress_discrete_bins
            )

        # We predict success for frames where progress < min_success or the frame is a success
        combined_mask = ((target_progress < min_success) | (success_labels > 0.5)).float()

        # Incorporate quality mask: always include all frames for suboptimal/failure trajectories
        if quality_mask is not None:
            combined_mask = torch.maximum(combined_mask, quality_mask)

        # if progress_loss_mask is not None:
        #     combined_mask = combined_mask * progress_loss_mask

        # Clamp logits to prevent extreme values and gradient issues
        success_logits = torch.clamp(success_logits, min=-50.0, max=50.0)

        # Compute class counts for balancing
        num_positives = (success_labels * combined_mask).sum()
        num_negatives = ((1 - success_labels) * combined_mask).sum()

        # Compute per-sample weights to balance classes
        # Weight the minority class so both classes contribute equally to the loss
        # success_loss_weight = max(num_pos, num_neg) / min(num_pos, num_neg)
        # Applied to whichever class has fewer samples
        if num_positives > 0 and num_negatives > 0:
            if num_positives < num_negatives:
                # Positives are minority - weight them up
                success_loss_weight = (num_negatives / num_positives).detach()
                sample_weights = torch.where(
                    success_labels > 0.5,
                    success_loss_weight * combined_mask,
                    combined_mask,
                )
            else:
                # Negatives are minority (or equal) - weight them up
                success_loss_weight = (num_positives / num_negatives).detach()
                sample_weights = torch.where(
                    success_labels > 0.5,
                    combined_mask,
                    success_loss_weight * combined_mask,
                )
        else:
            success_loss_weight = torch.tensor(1.0, device=success_logits.device, dtype=success_logits.dtype)
            sample_weights = combined_mask

        # Compute BCE loss with per-sample weights (includes combined_mask)
        loss = F.binary_cross_entropy_with_logits(
            success_logits,
            success_labels,
            weight=sample_weights,
            reduction="none",
        )
        combined_mask_index = combined_mask.bool()
        loss = (loss * combined_mask) / (sample_weights + 1e-8)
        success_loss = loss[combined_mask_index].mean()

        # Compute accuracy per sample
        success_preds = (torch.sigmoid(success_logits) > 0.5).float()
        correct = (success_preds == success_labels).float()
        masked_correct = correct * combined_mask

        # Compute per-sample positive and negative accuracy tensors (like masked_correct)
        # Only include values for samples of the corresponding class, use NaN for other class
        # positive_correct: correct (0 or 1) for positive samples (label==1), NaN for negative samples
        positive_mask = (success_labels == 1) & (combined_mask > 0)
        positive_correct_tensor = torch.where(
            positive_mask, correct, torch.tensor(float("nan"), device=correct.device, dtype=torch.float32)
        )

        # negative_correct: correct (0 or 1) for negative samples (label==0), NaN for positive samples
        negative_mask = (success_labels == 0) & (combined_mask > 0)
        negative_correct_tensor = torch.where(
            negative_mask, correct, torch.tensor(float("nan"), device=correct.device, dtype=torch.float32)
        )

        # Compute weighted accuracy (balanced accuracy) - scalar values for logging
        # Weight each class's accuracy by inverse of its frequency
        positive_correct_sum = (correct[combined_mask_index] * success_labels[combined_mask_index]).sum()
        negative_correct_sum = (correct[combined_mask_index] * (1 - success_labels[combined_mask_index])).sum()

        if num_positives > 0 and num_negatives > 0:
            # Balanced accuracy: average of recall for each class
            positive_acc_scalar = positive_correct_sum / (num_positives + 1e-8)
            negative_acc_scalar = negative_correct_sum / (num_negatives + 1e-8)
            weighted_acc = (positive_acc_scalar + negative_acc_scalar) / 2.0
        else:
            weighted_acc = masked_correct.sum() / (combined_mask.sum() + 1e-8)
            # Set accuracies to 0.0 when we can't compute them properly
            positive_acc_scalar = torch.tensor(0.0, device=success_loss.device, dtype=torch.float32)
            negative_acc_scalar = torch.tensor(0.0, device=success_loss.device, dtype=torch.float32)

        success_acc = masked_correct.sum() / (combined_mask.sum() + 1e-8)

        # Compute AUPRC (Area Under Precision-Recall Curve)
        success_probs = torch.sigmoid(success_logits)
        success_probs_flat = success_probs[combined_mask > 0]
        success_labels_flat = success_labels[combined_mask > 0]

        # Compute AUPRC across all valid frames
        if success_probs_flat.numel() > 0 and len(torch.unique(success_labels_flat)) > 1:
            auprc = average_precision_score(
                t2n(success_labels_flat),
                t2n(success_probs_flat),
            )
            batch_auprc = torch.tensor(auprc, device=success_loss.device, dtype=torch.float32)
        else:
            batch_auprc = torch.tensor(0.0, device=success_loss.device, dtype=torch.float32)

        metrics = {
            "masked_correct": masked_correct,
            "masked_loss": loss,
            "weighted_accuracy": weighted_acc,
            "positive_accuracy": positive_correct_tensor,  # Per-sample tensor like masked_correct
            "negative_accuracy": negative_correct_tensor,  # Per-sample tensor like masked_correct
            "positive_accuracy_scalar": positive_acc_scalar,  # Scalar for logging
            "negative_accuracy_scalar": negative_acc_scalar,  # Scalar for logging
            "success_loss_weight": success_loss_weight,
            "num_positives": num_positives,
            "num_negatives": num_negatives,
        }

        return success_loss, success_acc, batch_auprc, metrics

    def _compute_progress_loss_helper(
        self,
        progress_pred: torch.Tensor,
        target_progress: torch.Tensor,
        mask: torch.Tensor,
        predict_last_frame_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Helper function to compute progress loss.

        Args:
            progress_pred: Progress prediction tensors of shape (batch_size, seq_len) for L1/L2 loss,
                          or (batch_size, seq_len, num_bins) for discrete loss (logits)
            target_progress: Target progress tensors of shape (batch_size, seq_len) with values in [0, 1]
            mask: Per-sample mask tensor of shape (batch_size, seq_len) with 1.0 for samples where we should compute loss
            predict_last_frame_mask: Optional mask tensor of shape (batch_size, seq_len) with 1.0 for frames with partial_success.
                                    If provided, this takes precedence over config.

        Returns:
            tuple: (masked_loss, spearman_correlations, metrics)
        """
        # Handle Qwen downsampling: take every 2nd frame if using Qwen and NOT using multi_image
        # In multi_image mode, we already get one embedding per frame, so no downsampling needed
        if (
            "Qwen" in self.config.model.base_model_id or "Molmo" in self.config.model.base_model_id
        ) and not self.config.data.use_multi_image:
            target_progress = target_progress[:, ::2]
            mask = mask[:, ::2]
            if predict_last_frame_mask is not None:
                predict_last_frame_mask = predict_last_frame_mask[:, ::2]

        # Apply predict_last_frame_mask if provided
        # The mask defaults to all 1s (include all frames) unless explicitly set to 0
        if predict_last_frame_mask is not None:
            # Use the mask from batch inputs (already handles partial_success)
            # Ensure shapes match
            if predict_last_frame_mask.shape != mask.shape:
                # If predict_last_frame_mask is [batch_size, seq_len] and mask is [batch_size, 1], expand mask first
                if mask.shape[1] == 1 and predict_last_frame_mask.shape[1] > 1:
                    mask = mask.expand_as(predict_last_frame_mask)
                elif predict_last_frame_mask.shape[1] == 1 and mask.shape[1] > 1:
                    predict_last_frame_mask = predict_last_frame_mask.expand_as(mask)

            mask = mask * predict_last_frame_mask
        elif self.config.loss.predict_last_frame_progress:
            # Fallback to config-based logic: create a mask that only selects the last frame for each sequence
            last_frame_mask = torch.zeros_like(target_progress, dtype=torch.float32)
            last_frame_mask[:, -1] = 1.0  # Set last frame to 1.0 for all sequences
            mask = mask * last_frame_mask

        # Determine loss type from config
        loss_type = self.config.loss.progress_loss_type.lower()

        masked_correct = None

        # Set loss function based on loss type
        if loss_type == "discrete":
            # Discrete loss: target progress is already binned in data sampling
            num_bins = self.config.loss.progress_discrete_bins

            # Target progress is already discrete bins [0, num_bins-1] from data sampling
            # Convert to long tensor
            if len(target_progress.shape) == 2:
                target_bins = target_progress.long()  # [batch_size, seq_len]
                # Ensure bins are in valid range [0, num_bins-1]
                target_bins = torch.clamp(target_bins, 0, num_bins - 1)

                # progress_pred should be [batch_size, seq_len, num_bins] logits
                # Reshape for cross-entropy: [batch_size * seq_len, num_bins] and [batch_size * seq_len]
                batch_size, seq_len = target_bins.shape
                target_bins_flat = target_bins.view(batch_size * seq_len)  # [B*T]
            else:
                target_bins = target_progress
                # if we're using C51-style soft bins
                batch_size, seq_len, num_bins = target_bins.shape
                target_bins_flat = target_bins.view(batch_size * seq_len, num_bins)

            # Check if progress_pred has the correct shape for discrete mode
            if len(progress_pred.shape) == 2:
                # Model is outputting [batch_size, seq_len] instead of [batch_size, seq_len, num_bins]
                # This means the model wasn't configured for discrete mode
                raise ValueError(
                    f"Discrete loss requires progress_pred shape [batch_size, seq_len, num_bins], "
                    f"but got shape {progress_pred.shape}. "
                    f"The model's progress head may not be configured for discrete mode. "
                    f"Check that loss.progress_loss_type='discrete' is set before model initialization."
                )

            if progress_pred.shape[:2] != (batch_size, seq_len) or progress_pred.shape[2] != num_bins:
                raise ValueError(
                    f"Shape mismatch: progress_pred has shape {progress_pred.shape}, "
                    f"but expected [batch_size={batch_size}, seq_len={seq_len}, num_bins={num_bins}]"
                )

            progress_pred_flat = progress_pred.view(batch_size * seq_len, num_bins)  # [B*T, num_bins]
            # Mask shape may be [B, 1] or [B, seq_len] depending on downsampling/last_frame_mask
            # Ensure it matches target_bins shape [B, seq_len] before flattening
            if mask.shape[1] != seq_len:
                # Mask is [B, 1], expand to [B, seq_len]
                mask_expanded = mask.expand(batch_size, seq_len)  # [B, seq_len]
            else:
                # Mask is already [B, seq_len]
                mask_expanded = mask
            mask_flat = mask_expanded.flatten()  # [B*T]

            # Compute cross-entropy loss per sample
            loss_per_sample_flat = F.cross_entropy(progress_pred_flat, target_bins_flat, reduction="none")  # [B*T]

            # Compute accuracy: compare predicted bins (argmax) with target bins
            pred_bins_flat = torch.argmax(progress_pred_flat, dim=-1)  # [B*T]
            if len(target_bins_flat.shape) == 2:
                correct_flat = (pred_bins_flat == torch.argmax(target_bins_flat, dim=-1)).float()  # [B*T]
            else:
                correct_flat = (pred_bins_flat == target_bins_flat).float()  # [B*T]

            # Apply mask and reshape back
            masked_loss_flat = loss_per_sample_flat * mask_flat  # [B*T]
            masked_correct_flat = correct_flat * mask_flat  # [B*T]
            loss_per_sample = loss_per_sample_flat.view(batch_size, seq_len)  # [B, T]
            masked_loss = masked_loss_flat.view(batch_size, seq_len)  # [B, T]
            masked_correct = masked_correct_flat.view(batch_size, seq_len)  # [B, T]
        elif loss_type == "l1":
            loss_fn = F.l1_loss
        else:
            loss_fn = F.mse_loss

        # Compute loss_per_sample and masked_loss for L1/L2
        if loss_type != "discrete":
            loss_per_sample = loss_fn(progress_pred.float(), target_progress.float(), reduction="none")
            masked_loss = loss_per_sample * mask

        # For discrete mode, convert predictions back to continuous for spearman correlation
        # For L1/L2, use predictions as-is
        if loss_type == "discrete":
            progress_pred_for_corr = convert_bins_to_continuous(progress_pred)
            target_progress_for_corr = convert_discrete_target_to_continuous(target_progress, num_bins=num_bins)
        else:
            progress_pred_for_corr = progress_pred
            target_progress_for_corr = target_progress

        if mask.shape[1] != target_progress_for_corr.shape[1]:
            repeated_mask = mask.repeat(1, target_progress_for_corr.shape[1])
        else:
            repeated_mask = mask
        masked_spearman_corr = compute_spearman_correlation(
            progress_pred_for_corr, target_progress_for_corr, aggregate=False, mask=repeated_mask
        )
        masked_spearman_corr = masked_spearman_corr.detach()

        # Average per sample, then take mean across batch
        # TODO: might need to change this if the mask is per timestep too
        progress_loss = masked_loss.mean(dim=1).sum(dim=0) / (mask.sum() + 1e-8)
        spearman_corr = masked_spearman_corr.mean()

        # Keep track of the per-sample metrics
        metrics = {"masked_loss": masked_loss, "masked_spearman_corr": masked_spearman_corr}

        # Add progress accuracy for discrete mode
        if loss_type == "discrete" and masked_correct is not None:
            metrics["masked_progress_accuracy"] = masked_correct

        return progress_loss, spearman_corr, metrics

    def _add_stratified_metrics(
        self,
        outputs_dict: Dict[str, Any],
        prefix: str,
        strategy_values: Optional[List[str]],
        data_source_values: List[str],
        metrics: Dict[str, torch.Tensor],
        loss_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Add stratified metrics (by strategy and data source) to outputs_dict.

        Args:
            outputs_dict: Dictionary to update with metrics
            prefix: Prefix for metric keys (e.g., "train" or "eval")
            strategy_values: List of strategy values to split by (e.g., data_gen_strategy), or None to skip
            data_source_values: List of data source values to split by
            metrics: Dictionary of metric tensors, e.g., {"acc": tensor, "loss": tensor, "margin": tensor}
            loss_mask: Optional[torch.Tensor] = None,
        """
        device = self.accelerator.device

        # Flatten loss_mask to 1D if it has extra dimensions (e.g., [batch_size, 1] -> [batch_size])
        if loss_mask is not None:
            loss_mask = loss_mask.squeeze()

        # Split by strategy
        if strategy_values is not None:
            strats = set(strategy_values)
            for strat in strats:
                mask = torch.tensor(
                    [1 if s == strat else 0 for s in strategy_values], device=device, requires_grad=False
                )
                # Apply mask to each metric and compute mean
                for metric_name, metric_tensor in metrics.items():
                    if len(metric_tensor.shape) == 0:
                        continue
                    if loss_mask is not None:
                        masked_metric = metric_tensor[(mask == 1) & (loss_mask == 1)].detach()
                    else:
                        masked_metric = metric_tensor[mask == 1].detach()

                    # get mean over non-nan values
                    non_nan_masked_metric = masked_metric[~torch.isnan(masked_metric)]
                    if non_nan_masked_metric.numel() > 0:
                        mean_value = non_nan_masked_metric.mean().item()
                        outputs_dict[f"{prefix}_strat_{metric_name}/{strat}"] = mean_value

        # Split by data source
        data_sources = set(data_source_values)
        for data_source in data_sources:
            mask = torch.tensor(
                [1 if s == data_source else 0 for s in data_source_values], device=device, requires_grad=False
            )
            # Apply mask to each metric and compute mean
            for metric_name, metric_tensor in metrics.items():
                if len(metric_tensor.shape) == 0:
                    continue
                if loss_mask is not None:
                    masked_metric = metric_tensor[(mask == 1) & (loss_mask == 1)].detach()
                else:
                    masked_metric = metric_tensor[mask == 1].detach()
                non_nan_masked_metric = masked_metric[~torch.isnan(masked_metric)]
                if non_nan_masked_metric.numel() > 0:
                    mean_value = non_nan_masked_metric.mean().item()
                    outputs_dict[f"{prefix}_ds_{metric_name}/{data_source}"] = mean_value

    def forward_model(self, model, inputs, sample_type="progress"):
        """Forward pass for the model."""
        logger.trace(f"forward_model: Starting forward pass for sample_type={sample_type}")

        with _timer("time/forward", timing_raw=self.timing_raw):
            if "rewind" in self.config.model.base_model_id:
                logger.trace("forward_model: Using ReWiND model path")
                model_output, model_timing_raw = model(
                    input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask"),
                    pixel_values=inputs.get("pixel_values", None),
                    pixel_values_videos=inputs.get("pixel_values_videos", None),
                    video_embeddings=inputs.get("video_embeddings", None),
                    text_embeddings=inputs.get("text_embeddings", None),
                    sample_type=sample_type,
                    timing_raw=self.timing_raw,
                )
            else:
                logger.trace("forward_model: Using Qwen/Molmo/RBM model path, calling model forward")
                logger.trace(
                    f"forward_model: input_ids shape: {inputs['input_ids'].shape if 'input_ids' in inputs else 'N/A'}"
                )

                # Build model kwargs - include both Qwen and Molmo2 specific parameters
                model_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "pixel_values": inputs.get("pixel_values", None),
                    "pixel_values_videos": inputs.get("pixel_values_videos", None),
                    # Qwen-specific parameters
                    "image_grid_thw": inputs.get("image_grid_thw", None),
                    "video_grid_thw": inputs.get("video_grid_thw", None),
                    "second_per_grid_ts": inputs.get("second_per_grid_ts", None),
                    # Molmo2-specific parameters
                    "image_grids": inputs.get("image_grids", None),
                    "image_token_pooling": inputs.get("image_token_pooling", None),
                    "image_num_crops": inputs.get("image_num_crops", None),
                    "video_grids": inputs.get("video_grids", None),
                    "video_token_pooling": inputs.get("video_token_pooling", None),
                    # Common parameters
                    "sample_type": sample_type,
                    "timing_raw": self.timing_raw,
                }
                model_output, model_timing_raw = model(**model_kwargs)
                logger.trace("forward_model: Model forward pass completed")

            logger.trace("forward_model: Updating timing and returning")
            self.timing_raw.update(model_timing_raw)
            return model_output, model_timing_raw

    def _compute_progress_loss(self, model, inputs, return_outputs=False, training=True, stratify_by_strategy=True):
        """
        Compute progress prediction loss.

        Args:
            model: The model to use for forward pass
            inputs: Input dictionary containing progress data
            return_outputs: Whether to return detailed outputs dict
            training: Whether in training mode
            stratify_by_strategy: Whether to stratify metrics by data_gen_strategy (default: True)
                                 Set to False for single-frame training where strategies aren't used
        """
        model_output, _ = self.forward_model(model, inputs, sample_type="progress")
        progress_logits = model_output.progress_logits
        progress_pred = progress_logits["A"]
        progress_target = inputs["target_progress"]
        progress_target_mask = inputs["target_progress_mask"].unsqueeze(-1)
        predict_last_frame_mask = inputs["predict_last_frame_mask"]

        progress_loss, spearman_corr, progress_metrics = self._compute_progress_loss_helper(
            progress_pred, progress_target, progress_target_mask, predict_last_frame_mask=predict_last_frame_mask
        )
        final_loss = 0

        final_loss += progress_loss
        if self.config.model.train_success_head:
            success_logits = model_output.success_logits
            success_pred = success_logits["A"]
            success_labels = inputs["success_labels"]

            quality_labels = inputs.get("quality_labels", None)
            success_loss, success_accuracy, success_auprc, success_metrics = self._compute_success_loss_helper(
                success_pred,
                progress_target,
                success_labels,
                progress_loss_mask=progress_target_mask,
                quality_labels=quality_labels,
            )
            # success_loss is already balanced via per-sample weighting of minority class
            if not torch.isnan(success_loss).any():
                final_loss += success_loss
            else:
                logger.warning(f"NaN detected in success loss")

        # Check for NaN in final loss
        if torch.isnan(final_loss).any():
            if training:
                import ipdb

                ipdb.set_trace()
            logger.warning(f"NaN detected in progress loss, replacing with 0.0")
            final_loss = torch.tensor(0.0, device=final_loss.device, dtype=final_loss.dtype)

        if return_outputs:
            outputs_dict = {}

            prefix = "train" if training else "eval"
            stratified_metrics = {
                "spearman_corr": progress_metrics["masked_spearman_corr"],
                "prog_loss": progress_metrics["masked_loss"],
            }

            strategy_values = inputs.get("data_gen_strategy") if stratify_by_strategy else None
            self._add_stratified_metrics(
                outputs_dict,
                prefix,
                strategy_values,
                inputs["data_source"],
                stratified_metrics,
                progress_target_mask,
            )

            outputs_dict.update({
                f"{prefix}/prog_loss": progress_loss.item(),
                f"{prefix}/spearman_corr": spearman_corr.item(),
            })

            # Add progress accuracy for discrete mode
            if "masked_progress_accuracy" in progress_metrics:
                # Expand mask to match masked_progress_accuracy shape [batch_size, seq_len]
                masked_progress_accuracy = progress_metrics["masked_progress_accuracy"]
                batch_size, seq_len = masked_progress_accuracy.shape
                if progress_target_mask.shape[1] != seq_len:
                    mask_expanded = progress_target_mask.expand(batch_size, seq_len)
                else:
                    mask_expanded = progress_target_mask
                progress_accuracy = masked_progress_accuracy.sum() / (mask_expanded.sum() + 1e-8)
                outputs_dict[f"{prefix}/prog_accuracy"] = progress_accuracy.item()

            if self.config.model.train_success_head:
                weighted_accuracy = success_metrics["weighted_accuracy"]
                positive_accuracy_scalar = success_metrics["positive_accuracy_scalar"]
                negative_accuracy_scalar = success_metrics["negative_accuracy_scalar"]
                success_loss_weight = success_metrics["success_loss_weight"]
                outputs_dict.update({
                    f"{prefix}/success_loss": success_loss.item(),
                    f"{prefix}/success_accuracy": success_accuracy.item(),
                    f"{prefix}/success_auprc": success_auprc.item(),
                    f"{prefix}/weighted_success_accuracy": weighted_accuracy.item()
                    if torch.is_tensor(weighted_accuracy)
                    else weighted_accuracy,
                    f"{prefix}/positive_success_accuracy": positive_accuracy_scalar.item()
                    if torch.is_tensor(positive_accuracy_scalar)
                    else positive_accuracy_scalar,
                    f"{prefix}/negative_success_accuracy": negative_accuracy_scalar.item()
                    if torch.is_tensor(negative_accuracy_scalar)
                    else negative_accuracy_scalar,
                    f"{prefix}/success_loss_weight": success_loss_weight.item()
                    if torch.is_tensor(success_loss_weight)
                    else success_loss_weight,
                    f"{prefix}/success_num_positives": success_metrics["num_positives"].item(),
                    f"{prefix}/success_num_negatives": success_metrics["num_negatives"].item(),
                })

        if not return_outputs:
            return final_loss

        return final_loss, outputs_dict

    def _compute_preference_loss(self, model, inputs, return_outputs=False, training=True):
        """Compute preference prediction loss using Bradley-Terry model."""
        model_outputs, model_timing_raw = self.forward_model(model, inputs, sample_type="preference")
        progress_logits = model_outputs.progress_logits

        # Get preference labels (1 if first trajectory is preferred, 0 if second trajectory is preferred)
        preference_labels = inputs["preference_labels"]

        # Get preference scores from the preference head
        preference_scores = model_outputs.pref_logits.squeeze(-1)  # [batch_size]

        # Clamp logits to prevent extreme values and gradient issues
        preference_scores = torch.clamp(preference_scores, min=-50.0, max=50.0)

        # Binary cross entropy loss for preference prediction
        preference_loss_all = F.binary_cross_entropy_with_logits(
            preference_scores, preference_labels.float(), reduction="none"
        )
        preference_loss = preference_loss_all.mean()

        final_loss = 0

        if not torch.isnan(preference_loss).any():
            final_loss += preference_loss
        else:
            logger.warning(f"NaN detected in preference loss")

        # =========================================================================================
        # Compute progress and success loss for the first trajectory in the paired samples
        # =========================================================================================
        target_progress_A = inputs["target_progress_A"]
        target_progress_A_mask = inputs["target_progress_A_mask"].unsqueeze(-1)
        data_gen_strat = inputs["trajectory_A_data_gen_strategy"]
        logger.warning(f"DATA GEN STRAT FOR TRAJ A: {data_gen_strat}")
        logger.warning(f"DATA SOURCE FOR TRAJ A: {inputs['trajectory_A_data_source']}")
        # logger.warning(f"PREFERENCE LABELS: {inputs['preference_labels']}")

        if self.config.model.train_progress_head and self.config.training.predict_pref_progress:
            progress_pred_A = progress_logits["A"]
            predict_last_frame_mask_A = inputs["predict_last_frame_mask_A"]
            progress_loss_A, spearman_corr_A, progress_metrics_A = self._compute_progress_loss_helper(
                progress_pred_A,
                target_progress_A,
                target_progress_A_mask,
                predict_last_frame_mask=predict_last_frame_mask_A,
            )
            final_loss += progress_loss_A

        if self.config.model.train_success_head:
            success_logits = model_outputs.success_logits
            success_logits = success_logits["A"]
            success_labels_A = inputs["success_labels_A"]

            quality_labels_A = inputs.get("trajectory_A_quality_label", None)
            success_loss, success_accuracy, success_auprc, success_metrics_A = self._compute_success_loss_helper(
                success_logits,
                target_progress_A,
                success_labels_A,
                progress_loss_mask=target_progress_A_mask,
                quality_labels=quality_labels_A,
            )
            # success_loss is already balanced via per-sample weighting of minority class
            if not torch.isnan(success_loss).any():
                final_loss += success_loss
            else:
                logger.warning(f"NaN detected in success loss")

        # Check for NaN in final loss
        if torch.isnan(final_loss).any():
            logger.warning(f"NaN detected in preference loss")

        if return_outputs:
            outputs_dict = {}

            prefix = "train" if training else "eval"
            rejected_data_gen_strategy = inputs["rejected_data_gen_strategy"]

            if self.config.model.train_progress_head and self.config.training.predict_pref_progress:
                outputs_dict.update({
                    f"{prefix}/pref_prog_loss": progress_loss_A.item(),
                    f"{prefix}/pref_prog_spearman_corr": spearman_corr_A.item(),
                })

                # Add progress accuracy for discrete mode
                if "masked_progress_accuracy" in progress_metrics_A:
                    # Expand mask to match masked_progress_accuracy shape [batch_size, seq_len]
                    masked_progress_accuracy = progress_metrics_A["masked_progress_accuracy"]
                    batch_size, seq_len = masked_progress_accuracy.shape
                    if target_progress_A_mask.shape[1] != seq_len:
                        mask_expanded = target_progress_A_mask.expand(batch_size, seq_len)
                    else:
                        mask_expanded = target_progress_A_mask
                    progress_accuracy_A = masked_progress_accuracy.sum() / (mask_expanded.sum() + 1e-8)
                    outputs_dict[f"{prefix}/pref_prog_accuracy"] = progress_accuracy_A.item()

                stratified_progress_metrics = {
                    "spearman_corr": progress_metrics_A["masked_spearman_corr"],
                    "prog_loss": progress_metrics_A["masked_loss"],
                }

                self._add_stratified_metrics(
                    outputs_dict,
                    prefix,
                    inputs["trajectory_A_data_gen_strategy"],
                    inputs["trajectory_A_data_source"],
                    stratified_progress_metrics,
                    target_progress_A_mask,
                )

            if self.config.model.train_success_head:
                weighted_accuracy = success_metrics_A["weighted_accuracy"]
                positive_accuracy_scalar = success_metrics_A["positive_accuracy_scalar"]
                negative_accuracy_scalar = success_metrics_A["negative_accuracy_scalar"]
                success_loss_weight = success_metrics_A["success_loss_weight"]
                outputs_dict.update({
                    f"{prefix}/pref_success_loss": success_loss.item(),
                    f"{prefix}/pref_success_accuracy": success_accuracy.item(),
                    f"{prefix}/pref_success_auprc": success_auprc.item(),
                    f"{prefix}/pref_weighted_success_accuracy": weighted_accuracy.item()
                    if torch.is_tensor(weighted_accuracy)
                    else weighted_accuracy,
                    f"{prefix}/pref_positive_success_accuracy": positive_accuracy_scalar.item()
                    if torch.is_tensor(positive_accuracy_scalar)
                    else positive_accuracy_scalar,
                    f"{prefix}/pref_negative_success_accuracy": negative_accuracy_scalar.item()
                    if torch.is_tensor(negative_accuracy_scalar)
                    else negative_accuracy_scalar,
                    f"{prefix}/pref_success_loss_weight": success_loss_weight.item()
                    if torch.is_tensor(success_loss_weight)
                    else success_loss_weight,
                })

                stratified_success_metrics = {
                    "success_loss": success_metrics_A["masked_loss"],
                    "success_acc": success_metrics_A["masked_correct"],
                    "success_pos_acc": success_metrics_A["positive_accuracy"],
                    "success_neg_acc": success_metrics_A["negative_accuracy"],
                }
                self._add_stratified_metrics(
                    outputs_dict,
                    prefix,
                    inputs["trajectory_A_data_gen_strategy"],
                    inputs["trajectory_A_data_source"],
                    stratified_success_metrics,
                    target_progress_A_mask,
                )

            if preference_loss is not None:
                # Compute preference accuracy for training monitoring
                preference_probs = torch.sigmoid(preference_scores)
                preference_predictions = (preference_probs > 0.5).float()
                preference_accuracy = (preference_predictions == preference_labels).float()

                # Prepare metrics for stratification
                stratified_metrics = {
                    "pref_acc": preference_accuracy,
                    "pref_loss": preference_loss_all,
                }

                self._add_stratified_metrics(
                    outputs_dict,
                    prefix,
                    rejected_data_gen_strategy,
                    inputs["data_source"],
                    stratified_metrics,
                )

                outputs_dict.update({
                    f"{prefix}/preference_loss": preference_loss.item(),
                    f"{prefix}/preference_accuracy": preference_accuracy.mean().item(),
                })
                return final_loss, outputs_dict

        return final_loss
