#!/usr/bin/env python3
"""
ReWiND Transformer implementation.
Contains the ReWINDTransformer class with three prediction heads for different objectives.

Note: make sure that the forward pass uses all of the
heads or there will be some problems with FSDP sharding.
"""

import einops
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers import PretrainedConfig
from robometer.models.utils import ModelOutput
from robometer.models.heads import PredictionHeadsMixin


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ReWINDTransformerConfig(PretrainedConfig):
    model_type = "rewind_transformer"

    def __init__(
        self,
        video_feature_dim: int = 768,
        text_feature_dim: int = 384,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 16,
        causal_mask: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.video_feature_dim = video_feature_dim
        self.text_feature_dim = text_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.max_len = max_len
        self.causal_mask = causal_mask


class ReWiNDTransformer(PredictionHeadsMixin, PreTrainedModel):
    """ReWiND Transformer with three prediction heads for different objectives."""

    config_class = ReWINDTransformerConfig

    def __init__(self, config, processor=None, tokenizer=None, image_encoder=None, text_encoder=None):
        rewind_config = config.rewind

        video_feature_dim = rewind_config.video_feature_dim
        text_feature_dim = rewind_config.text_feature_dim

        if image_encoder is not None:
            video_feature_dim = image_encoder.config.hidden_size
        if text_encoder is not None:
            text_feature_dim = text_encoder.config.hidden_size

        super().__init__(
            config=config.rewind,
            model_config=config,
            hidden_dim=rewind_config.hidden_dim,
            dropout=rewind_config.dropout,
        )

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.processor = processor

        self.video_proj = nn.Linear(video_feature_dim, rewind_config.hidden_dim)
        self.text_proj = nn.Linear(text_feature_dim, rewind_config.hidden_dim)

        self.first_embedding_A = nn.Parameter(torch.randn(1, 1, rewind_config.hidden_dim))
        self.first_embedding_B = nn.Parameter(torch.randn(1, 1, rewind_config.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=rewind_config.hidden_dim,
            nhead=rewind_config.num_attention_heads,
            dim_feedforward=rewind_config.hidden_dim * 4,
            dropout=rewind_config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=rewind_config.num_layers)

        if rewind_config.use_per_frame_progress_token:
            self.text_position_embedding = nn.Parameter(torch.randn(1, 1, rewind_config.hidden_dim))
            self.prog_token_A = nn.Parameter(torch.randn(1, rewind_config.max_len, rewind_config.hidden_dim))
            self.prog_token_B = nn.Parameter(torch.randn(1, rewind_config.max_len, rewind_config.hidden_dim))

        # Prediction token for preference
        self.preference_token = nn.Parameter(torch.randn(1, 1, rewind_config.hidden_dim))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values_videos=None,
        video_embeddings=None,
        text_embeddings=None,
        sample_type=None,  # "preference", "progress"
        timing_raw=None,
        **kwargs,
    ):
        """Forward pass for ReWiND Transformer."""
        if timing_raw is None:
            timing_raw = {}

        use_precomputed = video_embeddings is not None and text_embeddings is not None

        if use_precomputed:
            B, T, D_video = video_embeddings.shape
            D_text = text_embeddings.shape[1]

            # Project embeddings to hidden dimension
            video_embeddings = self.video_proj(video_embeddings.view(-1, D_video)).view(B, T, -1)  # [B, T, hidden_dim]
            text_embeddings = self.text_proj(text_embeddings)  # [B, hidden_dim]
        else:
            # Use raw inputs with encoders
            if pixel_values_videos is None:
                raise ValueError("pixel_values_videos is required when not using precomputed embeddings")

            B, T, C, H, W = pixel_values_videos.shape

            # processing text inputs
            with torch.no_grad():
                text_embeddings = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_embeddings = mean_pooling(text_embeddings, attention_mask)  # [B, text_hidden_dim]
                text_embeddings = self.text_proj(text_embeddings)  # [B, D]

            # processing video inputs
            # T should contain both chosen and rejected trajectories concatenated together
            pixel_values_videos = pixel_values_videos.view(B * T, C, H, W)
            video_embeddings = self.image_encoder(
                pixel_values=pixel_values_videos
            ).pooler_output  # [B, vision_hidden_dim]
            video_embeddings = self.video_proj(video_embeddings)  # [B * T, D]
            video_embeddings = video_embeddings.view(B, T, -1)  # [B, T, D]

        output = ModelOutput()

        if sample_type == "preference":
            half = T // 2
            video_embeddings_A = video_embeddings[:, :half].clone()
            video_embeddings_B = video_embeddings[:, half:].clone()

            # Add the first embedding to the beginning of each sequence
            first_frame_emb_A = einops.repeat(self.first_embedding_A, "1 1 d -> b 1 d", b=B)  # [B, 1, D]
            first_frame_emb_B = einops.repeat(self.first_embedding_B, "1 1 d -> b 1 d", b=B)  # [B, 1, D]
            video_embeddings_A[:, 0:1] += first_frame_emb_A
            video_embeddings_B[:, 0:1] += first_frame_emb_B

            if self.config.use_per_frame_progress_token:
                # Add position embedding to text
                text_pos = einops.repeat(self.text_position_embedding, "1 1 d -> b 1 d", b=B)
                text_emb = text_embeddings.unsqueeze(1) + text_pos

                # Get progress tokens - slice to match sequence length
                prog_token_A = einops.repeat(self.prog_token_A[:, :half, :], "1 t d -> b t d", b=B)  # [B, half, D]
                prog_token_B = einops.repeat(self.prog_token_B[:, :half, :], "1 t d -> b t d", b=B)  # [B, half, D]

                # Interleave frames with progress tokens: [frame, prog_token, frame, prog_token, ...]
                sequence_A = []
                for i in range(half):
                    sequence_A.append(video_embeddings_A[:, i : i + 1, :])  # [B, 1, D]
                    sequence_A.append(prog_token_A[:, i : i + 1, :])  # [B, 1, D]

                sequence_B = []
                for i in range(half):
                    sequence_B.append(video_embeddings_B[:, i : i + 1, :])  # [B, 1, D]
                    sequence_B.append(prog_token_B[:, i : i + 1, :])  # [B, 1, D]

                # Prediction token
                pred_token = einops.repeat(self.preference_token, "1 1 d -> b 1 d", b=B)

                # Concatenate: [text, frame1_A, prog_A, frame2_A, prog_A, ..., frame1_B, prog_B, frame2_B, prog_B, ..., pred]
                full_sequence = torch.cat([text_emb] + sequence_A + sequence_B + [pred_token], dim=1)  # [B, L, D]

                # Build causal mask if enabled
                mask = None
                if self.config.causal_mask:
                    L = full_sequence.shape[1]
                    mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=full_sequence.device), diagonal=1)

                full_embeddings = self.transformer(full_sequence, src_key_padding_mask=None, mask=mask)

                # Extract progress token embeddings
                D = full_embeddings.shape[-1]
                prog_embeddings_A = []
                prog_embeddings_B = []

                idx_A_start = 1
                idx_B_start = 1 + 2 * half
                idx_pred = 1 + 4 * half

                for i in range(half):
                    prog_idx_A = idx_A_start + 2 * i + 1
                    prog_embeddings_A.append(full_embeddings[:, prog_idx_A, :])  # [B, D]

                    prog_idx_B = idx_B_start + 2 * i + 1
                    prog_embeddings_B.append(full_embeddings[:, prog_idx_B, :])  # [B, D]

                prog_embeddings_A = torch.stack(prog_embeddings_A, dim=1)  # [B, half, D]
                prog_embeddings_B = torch.stack(prog_embeddings_B, dim=1)  # [B, half, D]

                # Apply heads
                progress_A_logits = self.progress_head(prog_embeddings_A.reshape(-1, D))
                if self.use_discrete_progress:
                    # Discrete: [b*t, num_bins] -> [b, t, num_bins]
                    progress_A_logits = einops.rearrange(progress_A_logits, "(b t) ... -> b t ...", b=B, t=half)
                else:
                    # Continuous: [b*t, 1] -> [b, t]
                    progress_A_logits = einops.rearrange(progress_A_logits, "(b t) 1 -> b t", b=B, t=half)

                progress_B_logits = self.progress_head(prog_embeddings_B.reshape(-1, D))
                if self.use_discrete_progress:
                    # Discrete: [b*t, num_bins] -> [b, t, num_bins]
                    progress_B_logits = einops.rearrange(progress_B_logits, "(b t) ... -> b t ...", b=B, t=half)
                else:
                    # Continuous: [b*t, 1] -> [b, t]
                    progress_B_logits = einops.rearrange(progress_B_logits, "(b t) 1 -> b t", b=B, t=half)

                progress_logits = {"A": progress_A_logits, "B": progress_B_logits}
                output.progress_logits = progress_logits

                success_A_logits = self.success_head(prog_embeddings_A.reshape(-1, D))
                success_A_logits = einops.rearrange(success_A_logits, "(b t) 1 -> b t", b=B, t=half)

                success_B_logits = self.success_head(prog_embeddings_B.reshape(-1, D))
                success_B_logits = einops.rearrange(success_B_logits, "(b t) 1 -> b t", b=B, t=half)

                success_logits = {"A": success_A_logits, "B": success_B_logits}
                output.success_logits = success_logits

                pred_class_token = full_embeddings[:, idx_pred, :]  # [B, D]
            else:
                # Regular mode: use frame embeddings directly
                pred_token = einops.repeat(self.preference_token, "1 1 d -> b 1 d", b=B)  # [B, 1, D]

                token_sequence = torch.cat(
                    [text_embeddings.unsqueeze(1), video_embeddings_A, video_embeddings_B, pred_token], dim=1
                )  # shape: [B, 2*T + 1, D]

                # Build causal mask if enabled
                mask = None
                if self.config.causal_mask:
                    L = token_sequence.shape[1]
                    mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=token_sequence.device), diagonal=1)

                token_embeddings = self.transformer(token_sequence, src_key_padding_mask=None, mask=mask)
                D = token_embeddings.shape[-1]

                final_embeddings_A = token_embeddings[:, 1 : 1 + half, :]  # avoid the text embedding
                final_embeddings_B = token_embeddings[:, 1 + half : -1, :]  # avoid the text embedding

                progress_A_logits = self.progress_head(final_embeddings_A.reshape(-1, D))
                if self.use_discrete_progress:
                    # Discrete: [b*t, num_bins] -> [b, t, num_bins]
                    progress_A_logits = einops.rearrange(progress_A_logits, "(b t) ... -> b t ...", b=B, t=half)
                else:
                    # Continuous: [b*t, 1] -> [b, t]
                    progress_A_logits = einops.rearrange(progress_A_logits, "(b t) 1 -> b t", b=B, t=half)

                progress_B_logits = self.progress_head(final_embeddings_B.reshape(-1, D))
                if self.use_discrete_progress:
                    # Discrete: [b*t, num_bins] -> [b, t, num_bins]
                    progress_B_logits = einops.rearrange(progress_B_logits, "(b t) ... -> b t ...", b=B, t=half)
                else:
                    # Continuous: [b*t, 1] -> [b, t]
                    progress_B_logits = einops.rearrange(progress_B_logits, "(b t) 1 -> b t", b=B, t=half)

                progress_logits = {"A": progress_A_logits, "B": progress_B_logits}
                output.progress_logits = progress_logits

                # Predict success for all frames
                success_A_logits = self.success_head(final_embeddings_A.reshape(-1, D))
                success_A_logits = einops.rearrange(success_A_logits, "(b t) 1 -> b t", b=B, t=half)

                success_B_logits = self.success_head(final_embeddings_B.reshape(-1, D))
                success_B_logits = einops.rearrange(success_B_logits, "(b t) 1 -> b t", b=B, t=half)

                success_logits = {"A": success_A_logits, "B": success_B_logits}
                output.success_logits = success_logits

                pred_class_token = token_embeddings[:, -1, :]  # [B, D]

            if sample_type == "preference":
                output.pref_logits = self.preference_head(pred_class_token)

        elif sample_type == "progress":
            if self.config.use_per_frame_progress_token:
                # Add position embedding to text
                text_pos_emb = einops.repeat(self.text_position_embedding, "1 1 d -> b 1 d", b=B)  # [B, 1, D]
                text_emb = text_embeddings.unsqueeze(1).clone()
                text_emb += text_pos_emb

                # Add the first embedding to the beginning of the sequence
                first_frame_emb = einops.repeat(self.first_embedding_A, "1 1 d -> b 1 d", b=B)  # [B, 1, D]
                video_embeddings = video_embeddings.clone()
                video_embeddings[:, 0:1] += first_frame_emb

                # Get progress tokens - slice to match sequence length
                prog_token_A = einops.repeat(self.prog_token_A[:, :T, :], "1 t d -> b t d", b=B)  # [B, T, D]

                # Interleave frames with progress tokens: [frame, prog_token, frame, prog_token, ...]
                sequence = []
                for i in range(T):
                    sequence.append(video_embeddings[:, i : i + 1, :])  # [B, 1, D]
                    sequence.append(prog_token_A[:, i : i + 1, :])  # [B, 1, D]

                # Concatenate: [text, frame1, prog_A, frame2, prog_A, ...]
                token_sequence = torch.cat([text_emb] + sequence, dim=1)  # [B, 2*T + 1, D]

                # Build causal mask if enabled
                mask = None
                if self.config.causal_mask:
                    L = token_sequence.shape[1]
                    mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=token_sequence.device), diagonal=1)

                token_embeddings = self.transformer(token_sequence, src_key_padding_mask=None, mask=mask)
                D = token_embeddings.shape[-1]

                # Extract progress token embeddings
                prog_embeddings = []
                for i in range(T):
                    prog_idx = 1 + 2 * i + 1
                    prog_embeddings.append(token_embeddings[:, prog_idx, :])  # [B, D]

                prog_embeddings = torch.stack(prog_embeddings, dim=1)  # [B, T, D]

                # Progress prediction for all frames
                progress_logits = self.progress_head(prog_embeddings.reshape(-1, D))
                if self.use_discrete_progress:
                    # Discrete: [b*t, num_bins] -> [b, t, num_bins]
                    progress_logits = einops.rearrange(progress_logits, "(b t) ... -> b t ...", b=B, t=T)
                else:
                    # Continuous: [b*t, 1] -> [b, t]
                    progress_logits = einops.rearrange(progress_logits, "(b t) 1 -> b t", b=B, t=T)

                # Predict success for all frames
                success_logits = self.success_head(prog_embeddings.reshape(-1, D))
                success_logits = einops.rearrange(success_logits, "(b t) 1 -> b t", b=B, t=T)
            else:
                # Regular mode: use frame embeddings directly
                first_frame_emb = einops.repeat(self.first_embedding_A, "1 1 d -> b 1 d", b=B)  # [B, 1, D]

                # [B, T, D]
                video_embeddings = video_embeddings.clone()
                video_embeddings[:, 0:1] += first_frame_emb

                token_sequence = torch.cat(
                    [text_embeddings.unsqueeze(1), video_embeddings], dim=1
                )  # shape: [B, T + 1, D]

                # Build causal mask if enabled
                mask = None
                if self.config.causal_mask:
                    L = token_sequence.shape[1]
                    mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=token_sequence.device), diagonal=1)

                token_embeddings = self.transformer(token_sequence, src_key_padding_mask=None, mask=mask)
                D = token_embeddings.shape[-1]
                final_embeddings = token_embeddings[:, 1:, :]  # avoid the text embedding

                # Progress prediction for all frames
                progress_logits = self.progress_head(final_embeddings)
                if self.use_discrete_progress:
                    # Discrete: [b, t, num_bins] - keep as is
                    pass
                else:
                    # Continuous: [b, t, 1] -> [b, t]
                    progress_logits = progress_logits.squeeze(-1)

                # Predict success for all frames
                success_logits = self.success_head(final_embeddings)
                success_logits = success_logits.squeeze(-1)

            progress_logits = {"A": progress_logits, "B": None}
            success_logits = {"A": success_logits, "B": None}
            output.progress_logits = progress_logits
            output.success_logits = success_logits

        return output, timing_raw


# Register the model and configs with transformers
AutoConfig.register("rewind_transformer", ReWINDTransformerConfig)
AutoModel.register(ReWINDTransformerConfig, ReWiNDTransformer)
