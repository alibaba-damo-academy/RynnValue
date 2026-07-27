#!/usr/bin/env python3
"""
Robometer implementation.
Contains the RBM class with three prediction heads for different objectives.

Note: make sure that the forward pass uses all of the
heads or there will be some problems with FSDP sharding.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, Qwen2_5_VLModel

try:
    from transformers import Qwen3VLModel
except ImportError:
    Qwen3VLModel = None

# from transformers import AutoModelForImageTextToText as Molmo2VLModel  # Molmo2 uses AutoModelForImageTextToText
from transformers import SmolVLMModel
import torch.distributed as dist

from robometer.models.utils import ModelOutput
from robometer.models.heads import PredictionHeadsMixin
from robometer.utils.timer import _timer
from robometer.utils.logger import get_logger

logger = get_logger()


def squeeze_last_safe(x: torch.Tensor) -> torch.Tensor:
    if x.ndim > 1 and x.shape[-1] == 1:
        return x.squeeze(-1)
    return x


class RBM(PredictionHeadsMixin, PreTrainedModel):
    """Robometer (RBM) with three prediction heads for different objectives.

    Supports multiple base model architectures:
    - Qwen2.5-VL (Qwen2_5_VLModel)
    - SmolVLM (AutoModelForImageTextToText)
    """

    # unused param i think
    config_class = Qwen2_5_VLModel.config_class

    # Declare support for SDPA and Flash Attention (will delegate to underlying model), needed for Qwen3
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def __init__(self, config, processor, tokenizer, base_model=None, base_model_id=None, model_config=None):
        if "SmolVLM" in base_model_id:
            hidden_size = config.text_config.hidden_size
            self.model_cls = SmolVLMModel
        elif "Qwen2.5" in base_model_id:
            hidden_size = config.hidden_size
            self.model_cls = Qwen2_5_VLModel
        elif "Qwen3" in base_model_id:
            hidden_size = config.text_config.hidden_size
            self.model_cls = Qwen3VLModel
        elif "Molmo" in base_model_id:
            # Molmo2 is based on Qwen3 architecture
            hidden_size = config.text_config.hidden_size
            self.model_cls = Qwen3VLModel
            # self.model_cls = Molmo2VLModel
        else:
            raise ValueError(f"Unsupported base model: {base_model_id}")

        super().__init__(
            config,
            hidden_dim=hidden_size,
            model_config=model_config,
            dropout=0.1,
        )

        if base_model is not None:
            self.model = base_model
        else:
            self.model = self.model_cls(config)

        self.config_class = self.model_cls.config_class
        self.base_model_id = base_model_id

        self.model_dtype = self.model.dtype
        self.progress_head = self.progress_head.to(dtype=self.model_dtype)
        self.preference_head = self.preference_head.to(dtype=self.model_dtype)
        self.success_head = self.success_head.to(dtype=self.model_dtype)

        self.processor = processor
        self.tokenizer = tokenizer
        self.model_config = model_config

        self.average_temporal_patches = self.model_config.average_temporal_patches
        self.use_per_frame_progress_token = self.model_config.use_per_frame_progress_token
        self.use_multi_image = self.model_config.use_multi_image

        # Frame pooling strategy for multi-image mode (used when NOT using per-frame progress tokens).
        # - mean: average pool patch tokens in the frame span
        # - boundary: use the last patch token in the frame span
        # - attention: learned attention pooling over patch tokens in the frame span
        self.frame_pooling = getattr(self.model_config, "frame_pooling", "mean")
        self.frame_pooling_attn_temperature = float(getattr(self.model_config, "frame_pooling_attn_temperature", 1.0))
        if self.frame_pooling_attn_temperature <= 0:
            raise ValueError("frame_pooling_attn_temperature must be > 0")
        # Always create the attention pooling projection so checkpoints can be loaded across pooling modes.
        self.frame_pool_attn = nn.Linear(hidden_size, 1, bias=False).to(dtype=self.model_dtype)

        # Validate that use_per_frame_progress_token requires use_multi_image
        if self.use_per_frame_progress_token and not self.use_multi_image:
            raise ValueError(
                "use_per_frame_progress_token=True requires use_multi_image=True. "
                "Per-frame progress tokens can only be used in multi-image mode."
            )

        # Molmo2 only supports multi-image mode, not video
        if "Molmo" in self.base_model_id and not self.use_multi_image:
            raise ValueError(
                "Molmo2 does not support video mode (use_multi_image=False). "
                "Please set data.use_multi_image=True to use Molmo2 with multi-image input."
            )

    def gradient_checkpointing_enable(self, **kwargs):
        """Delegates gradient checkpointing enabling to the base model."""
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        """Delegates gradient checkpointing disabling to the base model."""
        self.model.gradient_checkpointing_disable(**kwargs)

    def generate(self, *args, **kwargs):
        """Delegates generation to the base model."""
        return self.model.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Delegates input preparation for generation to the base model."""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def _extract_hidden_states_from_token_pairs(
        self,
        hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract image/video frame embeddings from hidden states by finding token pairs and mean pooling.

        This is a general function that works for both SmolVLM and Qwen multi-image mode.
        It automatically detects which model is being used based on the base_model_id:
        - SmolVLM: Uses the same token for start and end: <fake_token_around_image>
        - Qwen: Uses different tokens: <|vision_start|> and <|vision_end|>

        Args:
            hidden_state: Hidden states tensor [seq_len, hidden_dim]
            input_ids: Input token IDs [seq_len]

        Returns:
            frame_embeddings: Tensor [num_frames, hidden_dim] containing mean-pooled
                            embeddings for each frame/image between token pairs
        """
        # Detect model type and get appropriate tokenizer and tokens
        is_molmo = "Molmo" in self.base_model_id
        if "SmolVLM" in self.base_model_id:
            # SmolVLM mode: same token appears in pairs
            tokenizer = self.tokenizer
            start_token = "<fake_token_around_image>"
            end_token = None  # Same token for both start and end
            use_same_token = True
            use_molmo_mode = False
        elif is_molmo:
            # Molmo2 mode: <low_res_im_start> followed by <im_patch> tokens
            tokenizer = self.processor.tokenizer
            start_token = "<low_res_im_start>"
            end_token = None  # No explicit end token
            patch_token = "<im_patch>"
            use_same_token = False
            use_molmo_mode = True
        else:
            # Qwen mode: different start and end tokens
            tokenizer = self.processor.tokenizer
            start_token = "<|vision_start|>"
            end_token = "<|vision_end|>"
            use_same_token = False
            use_molmo_mode = False

        # Get token IDs
        start_token_id = tokenizer.convert_tokens_to_ids(start_token)

        # Find all positions where start tokens appear
        start_positions = (input_ids == start_token_id).nonzero(as_tuple=True)[0]

        if len(start_positions) == 0:
            raise ValueError(
                f"No {start_token} tokens found in input_ids. Token ID {start_token_id} not found in sequence."
            )

        # Handle different pairing modes
        if use_same_token:
            # SmolVLM mode: same token appears in pairs
            if len(start_positions) % 2 != 0:
                raise ValueError(
                    f"Expected even number of {start_token} tokens (pairs), but found {len(start_positions)} tokens."
                )

            # Group tokens into pairs (every two consecutive tokens form a pair)
            token_pairs = []
            for i in range(0, len(start_positions), 2):
                token_pairs.append((start_positions[i].item(), start_positions[i + 1].item()))
        elif use_molmo_mode:
            # Molmo2 mode: <low_res_im_start> followed by <im_patch> tokens
            patch_token_id = tokenizer.convert_tokens_to_ids(patch_token)
            im_patch_positions = (input_ids == patch_token_id).nonzero(as_tuple=True)[0]

            token_pairs = []
            for start_idx, start_pos in enumerate(start_positions):
                start_pos_val = start_pos.item()
                # Find the last consecutive im_patch token after this start
                patches_after_start = im_patch_positions[im_patch_positions > start_pos]
                if len(patches_after_start) > 0:
                    # Find where patches stop (at next image start or end of sequence)
                    if start_idx + 1 < len(start_positions):
                        next_start = start_positions[start_idx + 1].item()
                        patches_for_this_image = patches_after_start[patches_after_start < next_start]
                    else:
                        patches_for_this_image = patches_after_start
                    if len(patches_for_this_image) > 0:
                        end_pos = patches_for_this_image[-1].item()
                        token_pairs.append((start_pos_val, end_pos))
        else:
            # Qwen mode: different start and end tokens
            end_token_id = tokenizer.convert_tokens_to_ids(end_token)

            # Find all positions where end tokens appear
            end_positions = (input_ids == end_token_id).nonzero(as_tuple=True)[0]

            if len(end_positions) == 0:
                raise ValueError(
                    f"No {end_token} tokens found in input_ids. Token ID {end_token_id} not found in sequence."
                )

            if len(start_positions) != len(end_positions):
                raise ValueError(
                    f"Mismatched number of tokens: "
                    f"found {len(start_positions)} {start_token} tokens "
                    f"and {len(end_positions)} {end_token} tokens."
                )

            # Pair up start and end tokens (they should appear in order: start, end, start, end, ...)
            token_pairs = []
            for i in range(len(start_positions)):
                start_pos = start_positions[i].item()
                end_pos = end_positions[i].item()

                if start_pos >= end_pos:
                    raise ValueError(
                        f"Invalid token pair at index {i}: "
                        f"{start_token} at {start_pos}, {end_token} at {end_pos}. "
                        f"Start must come before end."
                    )
                token_pairs.append((start_pos, end_pos))

        # Extract hidden states between token pairs
        frame_embeddings = []
        for start_pos, end_pos in token_pairs:
            # Extract hidden states between the token pair (exclusive of the tokens themselves)
            # Add 1 to start_pos to exclude the start token, end_pos is exclusive
            frame_tokens = hidden_state[start_pos + 1 : end_pos]

            if frame_tokens.shape[0] == 0:
                # If no tokens between the pair, use the token positions themselves
                # This shouldn't happen normally, but handle it gracefully
                frame_embedding = (hidden_state[start_pos] + hidden_state[end_pos]) / 2.0
            else:
                if self.frame_pooling == "mean":
                    frame_embedding = frame_tokens.mean(dim=0)  # [hidden_dim]
                elif self.frame_pooling == "boundary":
                    # Use the last patch token as a boundary summary of the frame span.
                    frame_embedding = frame_tokens[-1]
                elif self.frame_pooling == "attention":
                    # Learned attention pooling over patch tokens in this frame span.
                    # scores: [num_tokens]
                    scores = self.frame_pool_attn(frame_tokens).squeeze(-1) / self.frame_pooling_attn_temperature
                    weights = torch.softmax(scores, dim=0).unsqueeze(-1)  # [num_tokens, 1]
                    frame_embedding = (weights * frame_tokens).sum(dim=0)
                else:
                    raise ValueError(f"Unsupported frame_pooling: {self.frame_pooling}")

            frame_embeddings.append(frame_embedding)

        if len(frame_embeddings) == 0:
            return torch.empty(0, hidden_state.shape[-1], device=hidden_state.device, dtype=hidden_state.dtype)

        return torch.stack(frame_embeddings)  # [num_frames, hidden_dim]

    def _extract_progress_from_trajectory(
        self,
        hidden_state: torch.Tensor,
        start_position: int,
        video_grid_thw: list[int],  # [T, H, W]
        merge_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract progress and success predictions from a trajectory's hidden states.

        Args:
            hidden_state: Hidden states tensor [seq_len, hidden_dim]
            start_position: Starting position in the sequence for this trajectory
            video_grid_thw: Video grid dimensions [T, H, W] where T is number of temporal patch groups,
                           H and W are spatial grid dimensions
            merge_size: Merge size for patch grouping

        Returns:
            tuple: (progress_logits [T], success_logits [T])
        """
        T, H, W = video_grid_thw

        if T == 0:
            return torch.empty(0, device=hidden_state.device), torch.empty(0, device=hidden_state.device)

        # Calculate tokens per frame: (H * W) // merge_size^2
        tokens_per_frame = (H * W) // (merge_size**2)

        if self.average_temporal_patches:
            # Average all tokens within each temporal patch group
            temporal_patch_tokens = []
            current_pos = start_position
            for t_idx in range(T):
                start_idx = current_pos
                end_idx = current_pos + tokens_per_frame
                patch_tokens = hidden_state[start_idx:end_idx]  # [tokens_per_frame, hidden_dim]
                patch_embedding = patch_tokens.mean(dim=0)  # [hidden_dim] - averaged
                temporal_patch_tokens.append(patch_embedding)
                current_pos = end_idx
            boundary_hidden_states = torch.stack(temporal_patch_tokens)  # [T, hidden_dim]
        else:
            # Use last token (boundary) of each temporal patch group
            frame_boundary_positions = []
            current_pos = start_position
            for _frame_idx in range(T):
                frame_end = current_pos + tokens_per_frame
                frame_boundary_positions.append(frame_end)
                current_pos = frame_end

            trajectory_boundaries = torch.tensor(frame_boundary_positions, device=hidden_state.device)
            boundary_hidden_states = hidden_state[trajectory_boundaries]  # [T, hidden_dim]

        assert boundary_hidden_states.shape[0] == T, f"Expected {T} frames, got {boundary_hidden_states.shape[0]}"
        progress_output = self.progress_head(boundary_hidden_states)  # [T, 1] or [T, num_bins] for discrete
        if self.use_discrete_progress:
            progress = progress_output  # [T, num_bins] - keep logits
        else:
            progress = squeeze_last_safe(progress_output)  # [T]
        success = squeeze_last_safe(self.success_head(boundary_hidden_states))  # [T]

        return progress, success

    def _extract_hidden_state_from_token(
        self,
        hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        token_name: str,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Extract hidden states at specific token positions.

        Args:
            hidden_state: Hidden states tensor [B, seq_len, hidden_dim] or [seq_len, hidden_dim]
            input_ids: Input token IDs [B, seq_len] or [seq_len]
            token_name: Name of the token to find (e.g., "<|prog_token|>", "<|pref_token|>")

        Returns:
            If exactly one token per sequence: tensor [B, hidden_dim]
            If multiple tokens per sequence (variable counts): list of tensors, one per batch item [num_tokens_i, hidden_dim]
        """
        # Handle both batched and unbatched inputs
        is_batched = hidden_state.dim() == 3
        if not is_batched:
            hidden_state = hidden_state.unsqueeze(0)  # [1, seq_len, hidden_dim]
            input_ids = input_ids.unsqueeze(0)  # [1, seq_len]

        B = input_ids.shape[0]

        # Get tokenizer (works for both SmolVLM and Qwen)
        if "SmolVLM" in self.base_model_id:
            tokenizer = self.tokenizer
        else:
            tokenizer = self.processor.tokenizer

        # Get token ID
        token_id = tokenizer.convert_tokens_to_ids(token_name)

        # Find all token positions across batch - vectorized
        token_mask = input_ids == token_id  # [B, seq_len]
        batch_indices, positions = token_mask.nonzero(as_tuple=True)  # both [total_tokens]

        if len(positions) == 0:
            raise ValueError(f"{token_name} not found in any sequence")

        # Extract all hidden states at token positions at once - vectorized
        all_hidden = hidden_state[batch_indices, positions]  # [total_tokens, hidden_dim]

        # Count tokens per batch sample
        token_counts = torch.bincount(batch_indices, minlength=B)  # [B]

        # Check if all sequences have exactly one token
        if (token_counts == 1).all():
            # Return as [B, hidden_dim] tensor
            return all_hidden.reshape(B, -1)
        else:
            # Split by batch index into list of tensors (variable token counts)
            return list(torch.split(all_hidden, token_counts.tolist()))

    def _forward_smolvlm(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        sample_type,
        timing_raw,
        **kwargs,
    ):
        """Forward pass for SmolVLM model. Returns (ModelOutput, timing_raw)."""
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            **kwargs,
        }
        with _timer("time/rbm_forward", timing_raw=timing_raw):
            outputs = self.model(**model_kwargs, output_hidden_states=True, return_dict=True)

        hidden_state = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]
        progress_logits = {"A": None, "B": None}
        success_logits = {"A": None, "B": None}

        # Create output
        output = ModelOutput()

        # Handle token-based extraction first if using per-frame progress tokens
        if self.use_per_frame_progress_token:
            progress_logits, success_logits, pref_or_sim_logits = self._process_token_extraction(
                hidden_state, input_ids, sample_type
            )
            output.progress_logits = progress_logits
            output.success_logits = success_logits
            if pref_or_sim_logits is not None:
                output.pref_logits = pref_or_sim_logits
        else:
            # Process frames normally
            with _timer("time/progress_logits", timing_raw=timing_raw):
                progress_logits, success_logits = self._process_smolvlm_frames(hidden_state, input_ids, sample_type)
            output.progress_logits = progress_logits
            output.success_logits = success_logits

            # Handle preference token if needed
            if sample_type == "preference":
                token_hidden = self._extract_hidden_state_from_token(hidden_state, input_ids, "<|pref_token|>")
                output.pref_logits = self.preference_head(token_hidden)

        return output, timing_raw

    def _process_smolvlm_frames(self, hidden_state, input_ids, sample_type):
        """Process SmolVLM frames and return progress/success logits."""
        B = input_ids.shape[0]
        progress_logits_A, progress_logits_B = [], []
        success_logits_A, success_logits_B = [], []

        for i in range(B):
            frame_embeddings = self._extract_hidden_states_from_token_pairs(hidden_state[i], input_ids[i])
            if frame_embeddings.shape[0] == 0:
                raise ValueError(f"No frame embeddings extracted for sample {i}")

            if sample_type == "progress":
                traj_A, traj_B = frame_embeddings, None
            else:
                mid = frame_embeddings.shape[0] // 2
                traj_A, traj_B = frame_embeddings[:mid], frame_embeddings[mid:]

            # Trajectory A
            prog_A = self.progress_head(traj_A)
            progress_logits_A.append(prog_A if self.use_discrete_progress else squeeze_last_safe(prog_A))
            success_logits_A.append(squeeze_last_safe(self.success_head(traj_A)))

            # Trajectory B
            if traj_B is not None:
                prog_B = self.progress_head(traj_B)
                progress_logits_B.append(prog_B if self.use_discrete_progress else squeeze_last_safe(prog_B))
                success_logits_B.append(squeeze_last_safe(self.success_head(traj_B)))
            else:
                progress_logits_B.append(None)
                success_logits_B.append(None)

        return (
            {
                "A": torch.stack(progress_logits_A) if progress_logits_A else None,
                "B": torch.stack(progress_logits_B) if progress_logits_B and progress_logits_B[0] is not None else None,
            },
            {
                "A": torch.stack(success_logits_A) if success_logits_A else None,
                "B": torch.stack(success_logits_B) if success_logits_B and success_logits_B[0] is not None else None,
            },
        )

    def _forward_qwen(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_values_videos,
        image_grid_thw,
        video_grid_thw,
        sample_type,
        timing_raw,
        second_per_grid_ts=None,
        **kwargs,
    ):
        """Forward pass for Qwen2.5/Qwen3 models. Returns (ModelOutput, timing_raw)."""
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "second_per_grid_ts": second_per_grid_ts,
            **kwargs,
        }
        with _timer("time/rbm_forward", timing_raw=timing_raw):
            # Qwen3 models may need output_hidden_states=True and use hidden_states instead of last_hidden_state
            is_qwen3 = "Qwen3" in self.base_model_id or (
                hasattr(self.model, "config") and "Qwen3" in str(type(self.model))
            )
            if is_qwen3:
                outputs = self.model(**model_kwargs, output_hidden_states=True, return_dict=True)
                # Qwen3 uses hidden_states tuple, take the last layer
                hidden_state = (
                    outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs.last_hidden_state
                )
            else:
                outputs = self.model(**model_kwargs)
                hidden_state = outputs.last_hidden_state  # [B, seq_len, hidden_dim]

        progress_logits = {"A": None, "B": None}
        success_logits = {"A": None, "B": None}

        # Create output
        output = ModelOutput()

        if self.use_per_frame_progress_token:
            progress_logits, success_logits, pref_or_sim_logits = self._process_token_extraction(
                hidden_state, input_ids, sample_type
            )
            output.progress_logits = progress_logits
            output.success_logits = success_logits
            if pref_or_sim_logits is not None:
                output.pref_logits = pref_or_sim_logits
        else:
            # Process frames normally
            vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            vision_end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            split_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|split_token|>")

            tps = getattr(getattr(self.processor, "video_processor", None), "temporal_patch_size", 2)
            merge_size = getattr(getattr(self.processor, "video_processor", None), "merge_size", 14)

            if self.use_multi_image:
                progress_logits, success_logits = self._process_multi_image_frames(
                    hidden_state,
                    input_ids,
                    sample_type,
                    vision_start_token_id,
                    vision_end_token_id,
                    split_token_id,
                    timing_raw,
                )
            else:
                progress_logits, success_logits = self._process_video_frames(
                    hidden_state,
                    input_ids,
                    video_grid_thw,
                    sample_type,
                    vision_start_token_id,
                    split_token_id,
                    tps,
                    merge_size,
                    timing_raw,
                )

            output.progress_logits = progress_logits
            output.success_logits = success_logits

            # Handle preference token if needed
            if sample_type == "preference":
                token_hidden = self._extract_hidden_state_from_token(hidden_state, input_ids, "<|pref_token|>")
                output.pref_logits = self.preference_head(token_hidden)

        return output, timing_raw

    def _forward_molmo(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        sample_type,
        timing_raw,
        image_grids=None,
        image_token_pooling=None,
        image_num_crops=None,
        **kwargs,
    ):
        """Forward pass for Molmo2 models (multi-image only). Returns (ModelOutput, timing_raw)."""
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "image_grids": image_grids,
            "image_token_pooling": image_token_pooling,
            "image_num_crops": image_num_crops,
            **kwargs,
        }
        with _timer("time/rbm_forward", timing_raw=timing_raw):
            # Qwen3 models may need output_hidden_states=True and use hidden_states instead of last_hidden_state
            is_qwen3 = "Qwen3" in self.base_model_id or (
                hasattr(self.model, "config") and "Qwen3" in str(type(self.model))
            )
            if is_qwen3:
                outputs = self.model(**model_kwargs, output_hidden_states=True, return_dict=True)
                # Qwen3 uses hidden_states tuple, take the last layer
                hidden_state = (
                    outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs.last_hidden_state
                )
            else:
                outputs = self.model(**model_kwargs)
                hidden_state = outputs.last_hidden_state  # [B, seq_len, hidden_dim]

        progress_logits = {"A": None, "B": None}
        success_logits = {"A": None, "B": None}

        # Create output
        output = ModelOutput()

        # Handle token-based extraction first if using per-frame progress tokens
        if self.use_per_frame_progress_token:
            progress_logits, success_logits, pref_or_sim_logits = self._process_token_extraction(
                hidden_state, input_ids, sample_type
            )
            output.progress_logits = progress_logits
            output.success_logits = success_logits
            if pref_or_sim_logits is not None:
                output.pref_logits = pref_or_sim_logits
        else:
            # Process frames normally
            vision_start_token_id = self.processor.tokenizer.convert_tokens_to_ids("<low_res_im_start>")
            split_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|split_token|>")

            progress_logits, success_logits = self._process_multi_image_frames(
                hidden_state, input_ids, sample_type, vision_start_token_id, None, split_token_id, timing_raw
            )

            output.progress_logits = progress_logits
            output.success_logits = success_logits

            # Handle preference token if needed
            if sample_type == "preference":
                token_hidden = self._extract_hidden_state_from_token(hidden_state, input_ids, "<|pref_token|>")
                output.pref_logits = self.preference_head(token_hidden)

        return output, timing_raw

    def _process_multi_image_frames(
        self,
        hidden_state,
        input_ids,
        sample_type,
        vision_start_token_id,
        vision_end_token_id,
        split_token_id,
        timing_raw,
    ):
        """Process frames in multi-image mode (shared by Qwen and Molmo)."""
        progress_logits_A = []
        progress_logits_B = []
        success_logits_A = []
        success_logits_B = []

        all_trajectory_A_frames = []
        all_trajectory_B_frames = []
        trajectory_A_lengths = []
        trajectory_B_lengths = []
        has_trajectory_B = sample_type != "progress"

        with _timer("time/progress_logits", timing_raw=timing_raw):
            # First pass: extract all frame embeddings
            for i, seq_ids in enumerate(input_ids):
                vision_start_positions = (seq_ids == vision_start_token_id).nonzero(as_tuple=True)[0]
                if len(vision_start_positions) == 0:
                    raise ValueError(f"vision_start_token (id={vision_start_token_id}) not found in sequence {i}")

                # Extract embeddings using _extract_hidden_states_from_token_pairs
                # (handles both Qwen and Molmo token patterns)
                frame_embeddings = self._extract_hidden_states_from_token_pairs(hidden_state[i], seq_ids)

                if frame_embeddings.shape[0] == 0:
                    raise ValueError(f"No frame embeddings extracted for sample {i}")

                # Split into trajectories
                if sample_type == "progress":
                    trajectory_A_frames = frame_embeddings
                    trajectory_B_frames = None
                else:
                    split_positions = (seq_ids == split_token_id).nonzero(as_tuple=True)[0]
                    if len(split_positions) == 0:
                        raise ValueError(f"split_token not found in sequence {i}")
                    split_pos = split_positions[0].item()
                    traj_a_count = sum(1 for pos in vision_start_positions if pos.item() < split_pos)
                    trajectory_A_frames = frame_embeddings[:traj_a_count]
                    trajectory_B_frames = frame_embeddings[traj_a_count:]

                all_trajectory_A_frames.append(trajectory_A_frames)
                trajectory_A_lengths.append(trajectory_A_frames.shape[0])

                if trajectory_B_frames is not None:
                    all_trajectory_B_frames.append(trajectory_B_frames)
                    trajectory_B_lengths.append(trajectory_B_frames.shape[0])
                else:
                    all_trajectory_B_frames.append(None)
                    trajectory_B_lengths.append(0)

            # Batch process trajectory A
            if all_trajectory_A_frames:
                batched_A = torch.cat(all_trajectory_A_frames, dim=0)
                progress_A_out = self.progress_head(batched_A)
                success_A_out = squeeze_last_safe(self.success_head(batched_A))

                if self.use_discrete_progress:
                    progress_A_split = torch.split(progress_A_out, trajectory_A_lengths, dim=0)
                else:
                    progress_A_split = torch.split(squeeze_last_safe(progress_A_out), trajectory_A_lengths, dim=0)
                success_A_split = torch.split(success_A_out, trajectory_A_lengths, dim=0)

                for prog, succ in zip(progress_A_split, success_A_split):
                    progress_logits_A.append(prog)
                    success_logits_A.append(succ)

            # Batch process trajectory B
            if has_trajectory_B:
                valid_B = [(f, l) for f, l in zip(all_trajectory_B_frames, trajectory_B_lengths) if f is not None]
                if valid_B:
                    valid_B_frames, valid_B_lengths = zip(*valid_B)
                    batched_B = torch.cat(valid_B_frames, dim=0)
                    progress_B_out = self.progress_head(batched_B)
                    success_B_out = squeeze_last_safe(self.success_head(batched_B))

                    if self.use_discrete_progress:
                        progress_B_split = torch.split(progress_B_out, list(valid_B_lengths), dim=0)
                    else:
                        progress_B_split = torch.split(squeeze_last_safe(progress_B_out), list(valid_B_lengths), dim=0)
                    success_B_split = torch.split(success_B_out, list(valid_B_lengths), dim=0)

                    valid_idx = 0
                    for frame in all_trajectory_B_frames:
                        if frame is not None:
                            progress_logits_B.append(progress_B_split[valid_idx])
                            success_logits_B.append(success_B_split[valid_idx])
                            valid_idx += 1
                        else:
                            progress_logits_B.append(None)
                            success_logits_B.append(None)
                else:
                    progress_logits_B = [None] * len(all_trajectory_A_frames)
                    success_logits_B = [None] * len(all_trajectory_A_frames)
            else:
                progress_logits_B = [None] * len(all_trajectory_A_frames)
                success_logits_B = [None] * len(all_trajectory_A_frames)

        progress_logits = {
            "A": torch.stack(progress_logits_A) if progress_logits_A else None,
            "B": torch.stack(progress_logits_B) if progress_logits_B and progress_logits_B[0] is not None else None,
        }
        success_logits = {
            "A": torch.stack(success_logits_A) if success_logits_A else None,
            "B": torch.stack(success_logits_B) if success_logits_B and success_logits_B[0] is not None else None,
        }
        return progress_logits, success_logits

    def _process_video_frames(
        self,
        hidden_state,
        input_ids,
        video_grid_thw,
        sample_type,
        vision_start_token_id,
        split_token_id,
        tps,
        merge_size,
        timing_raw,
    ):
        """Process frames in video mode (Qwen only). Returns (progress_logits, success_logits)."""
        progress_logits_A = []
        progress_logits_B = []
        success_logits_A = []
        success_logits_B = []

        with _timer("time/progress_logits", timing_raw=timing_raw):
            for i, seq_ids in enumerate(input_ids):
                vision_start_positions = (seq_ids == vision_start_token_id).nonzero(as_tuple=True)[0]
                if len(vision_start_positions) == 0:
                    raise ValueError(f"vision_start_token not found in sequence {i}")

                if video_grid_thw is None or i >= len(video_grid_thw):
                    raise ValueError(f"video_grid_thw required for video mode")

                # Trajectory A
                grid_idx_A = i if sample_type == "progress" else i * tps
                progress_A, success_A = self._extract_progress_from_trajectory(
                    hidden_state[i], vision_start_positions[0].item(), video_grid_thw[grid_idx_A], merge_size
                )
                progress_logits_A.append(progress_A)
                success_logits_A.append(success_A)

                # Trajectory B (if not progress sample)
                if sample_type != "progress":
                    grid_idx_B = i * tps + 1
                    if grid_idx_B >= len(video_grid_thw):
                        raise ValueError(f"video_grid_thw index {grid_idx_B} out of bounds")
                    progress_B, success_B = self._extract_progress_from_trajectory(
                        hidden_state[i], vision_start_positions[1].item(), video_grid_thw[grid_idx_B], merge_size
                    )
                    progress_logits_B.append(progress_B)
                    success_logits_B.append(success_B)
                else:
                    progress_logits_B.append(None)
                    success_logits_B.append(None)

        progress_logits = {
            "A": torch.stack(progress_logits_A) if progress_logits_A else None,
            "B": torch.stack(progress_logits_B) if progress_logits_B[0] is not None else None,
        }
        success_logits = {
            "A": torch.stack(success_logits_A) if success_logits_A else None,
            "B": torch.stack(success_logits_B) if success_logits_B[0] is not None else None,
        }
        return progress_logits, success_logits

    def _apply_heads_to_hidden_states(
        self, hidden_states_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply progress and success heads to a list of hidden states.

        Args:
            hidden_states_list: List of tensors, one per batch item [num_tokens_i, hidden_dim]

        Returns:
            tuple: (progress_logits, success_logits)
                - progress_logits: [B, max_tokens] or [B, max_tokens, num_bins] for discrete (padded)
                - success_logits: [B, max_tokens] (padded)
        """
        progress_list = []
        success_list = []
        for hidden in hidden_states_list:
            if hidden.shape[0] > 0:
                progress_output = self.progress_head(hidden)
                if self.use_discrete_progress:
                    progress_list.append(progress_output)
                else:
                    progress_list.append(squeeze_last_safe(progress_output))
                success_list.append(squeeze_last_safe(self.success_head(hidden)))
            else:
                progress_list.append(torch.empty(0, device=hidden.device))
                success_list.append(torch.empty(0, device=hidden.device))

        progress = torch.stack(progress_list) if progress_list else None
        success = torch.stack(success_list) if success_list else None
        return progress, success

    def _process_token_extraction(
        self,
        hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        sample_type: str,
    ) -> tuple[dict, dict, torch.Tensor | None]:
        """
        Process token-based extraction for progress and preference predictions.

        Returns:
            tuple: (progress_logits, success_logits, pref_logits)
                - progress_logits: dict with "A" and/or "B" keys
                - success_logits: dict with "A" and/or "B" keys
                - pref_logits: preference logits, or None
        """
        progress_logits = {"A": None, "B": None}
        success_logits = {"A": None, "B": None}
        pref_or_sim_logits = None

        # Extract all <|prog_token|> hidden states (returns list of tensors when multiple tokens)
        all_prog_token_hidden = self._extract_hidden_state_from_token(hidden_state, input_ids, "<|prog_token|>")

        if sample_type == "progress":
            # For progress samples, all tokens belong to trajectory A
            hidden_states_A = all_prog_token_hidden
            progress_logits["A"], success_logits["A"] = self._apply_heads_to_hidden_states(hidden_states_A)
        elif sample_type == "preference":
            # For preference, assume equal number of tokens for A and B
            hidden_states_A = []
            hidden_states_B = []
            for tokens in all_prog_token_hidden:
                num_tokens = tokens.shape[0]
                if num_tokens % 2 != 0:
                    raise ValueError(f"Expected even number of <|prog_token|> tokens, got {num_tokens}")
                mid = num_tokens // 2
                hidden_states_A.append(tokens[:mid])
                hidden_states_B.append(tokens[mid:])

            progress_logits["A"], success_logits["A"] = self._apply_heads_to_hidden_states(hidden_states_A)
            progress_logits["B"], success_logits["B"] = self._apply_heads_to_hidden_states(hidden_states_B)

            token_hidden = self._extract_hidden_state_from_token(hidden_state, input_ids, "<|pref_token|>")
            pref_or_sim_logits = self.preference_head(token_hidden)

        return progress_logits, success_logits, pref_or_sim_logits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        sample_type=None,  # "preference", "progress"
        second_per_grid_ts=None,
        timing_raw=None,
        # Molmo2-specific parameters
        image_grids=None,
        image_token_pooling=None,
        image_num_crops=None,
        video_grids=None,
        video_token_pooling=None,
        **kwargs,
    ):
        """
        Forward pass for the RBM (Robometer).

        Dispatches to model-specific forward methods:
        - SmolVLM: _forward_smolvlm
        - Qwen2.5/Qwen3: _forward_qwen
        - Molmo2: _forward_molmo

        Returns:
            tuple: (ModelOutput, timing_raw dict)
        """
        if timing_raw is None:
            timing_raw = {}

        # Dispatch to model-specific forward
        if "SmolVLM" in self.base_model_id:
            return self._forward_smolvlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                sample_type=sample_type,
                timing_raw=timing_raw,
                **kwargs,
            )
        elif "Molmo" in self.base_model_id:
            return self._forward_molmo(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                sample_type=sample_type,
                timing_raw=timing_raw,
                image_grids=image_grids,
                image_token_pooling=image_token_pooling,
                image_num_crops=image_num_crops,
                **kwargs,
            )
        else:
            # Qwen2.5 / Qwen3
            return self._forward_qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                sample_type=sample_type,
                timing_raw=timing_raw,
                second_per_grid_ts=second_per_grid_ts,
                **kwargs,
            )
