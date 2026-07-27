#!/usr/bin/env python3
"""
Shared prediction heads plugin for RBM and ReWiND models.
Provides a mixin class that models can inherit from to get prediction heads.
"""

import torch
import torch.nn as nn
from typing import Optional
from robometer.utils.logger import get_logger

logger = get_logger()


class PredictionHeadsMixin(nn.Module):
    """
    Mixin class that provides prediction heads for reward models.

    Models should inherit from this mixin and pass hidden_dim and optionally model_config
    via **kwargs to super().__init__().
    """

    def __init__(
        self,
        *args,
        hidden_dim: Optional[int] = None,
        model_config: Optional[object] = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        """
        Initialize prediction heads if parameters are provided.

        Args:
            hidden_dim: Input hidden dimension for all heads (if None, heads won't be initialized)
            model_config: Optional model config object with loss settings (if provided, extracts progress settings from it)
            dropout: Dropout rate for all heads
        """
        super().__init__(*args, **kwargs)

        if hidden_dim is not None:
            # Extract progress settings from model_config if provided
            if model_config is not None:
                progress_output_size = 1  # Default: continuous output
                progress_use_sigmoid = True
                use_discrete_progress = False
                # Check for progress_loss_type in model_config (direct attribute or under loss)
                progress_loss_type = None
                if hasattr(model_config, "progress_loss_type"):
                    progress_loss_type = model_config.progress_loss_type
                elif hasattr(model_config, "loss") and hasattr(model_config.loss, "progress_loss_type"):
                    progress_loss_type = model_config.loss.progress_loss_type

                if progress_loss_type and progress_loss_type.lower() == "discrete":
                    use_discrete_progress = True
                    if hasattr(model_config, "progress_discrete_bins"):
                        progress_output_size = model_config.progress_discrete_bins
                    elif hasattr(model_config, "loss") and hasattr(model_config.loss, "progress_discrete_bins"):
                        progress_output_size = model_config.loss.progress_discrete_bins
                    else:
                        progress_output_size = 10  # Default bins
                    progress_use_sigmoid = False
                # Set use_discrete_progress attribute for models to use
                self.use_discrete_progress = use_discrete_progress
            else:
                # Default values if no model_config
                progress_output_size = 1
                progress_use_sigmoid = True
                self.use_discrete_progress = False

            logger.info(f"PredictionHeadsMixin. __init__: use_discrete_progress: {self.use_discrete_progress}")

            # Progress head
            progress_layers = [
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, progress_output_size),
            ]
            if progress_use_sigmoid:
                progress_layers.append(nn.Sigmoid())
            self.progress_head = nn.Sequential(*progress_layers)

            # Preference head
            self.preference_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

            # Success head
            self.success_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
