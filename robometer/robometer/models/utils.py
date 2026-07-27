import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelOutput:
    pref_logits: torch.Tensor | None = None
    success_logits: torch.Tensor | None = None
    progress_logits: torch.Tensor | None = None

    hidden_states: torch.Tensor | None = None


def convert_bins_to_continuous(bin_logits: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert discrete bins to continuous progress values in [0, 1] via weighted sum of bin centers."""
    num_bins = bin_logits.shape[-1]
    if (bin_logits.sum(dim=-1) == 1).all():
        bin_probs = bin_logits
    else:
        bin_probs = (
            torch.softmax(bin_logits, dim=-1)
            if isinstance(bin_logits, torch.Tensor)
            else np.softmax(bin_logits, axis=-1)
        )
    bin_centers = (
        torch.linspace(0.0, 1.0, num_bins, device=bin_logits.device, dtype=bin_logits.dtype)
        if isinstance(bin_logits, torch.Tensor)
        else np.linspace(0.0, 1.0, num_bins)
    )
    return (
        (bin_probs * bin_centers).sum(dim=-1)
        if isinstance(bin_logits, torch.Tensor)
        else (bin_probs * bin_centers).sum(axis=-1)
    )


def convert_bins_to_continuous_hard(
    bin_logits: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """
    Convert discrete bins to a continuous value in [0, 1]
    by selecting the argmax bin and returning its center.
    """
    num_bins = bin_logits.shape[-1]

    if isinstance(bin_logits, torch.Tensor):
        idx = torch.argmax(bin_logits, dim=-1)
        bin_centers = torch.linspace(
            0.0,
            1.0,
            num_bins,
            device=bin_logits.device,
            dtype=bin_logits.dtype,
        )
        return bin_centers[idx]

    else:
        idx = np.argmax(bin_logits, axis=-1)
        bin_centers = np.linspace(0.0, 1.0, num_bins)
        return bin_centers[idx]


def convert_bin_index_to_continuous(bin_index: torch.Tensor | np.ndarray, num_bins: int) -> torch.Tensor | np.ndarray:
    """Convert discrete bin index to continuous progress value in [0, 1] by selecting the argmax bin and returning its center."""
    bin_centers = torch.linspace(0.0, 1.0, num_bins, device=bin_index.device, dtype=bin_index.dtype)
    return bin_centers[bin_index.long()]


def convert_discrete_target_to_continuous(
    target: torch.Tensor | np.ndarray, num_bins: int
) -> torch.Tensor | np.ndarray:
    """Convert discrete target to continuous progress value in [0, 1] by selecting the argmax bin and returning its center."""
    if len(target.shape) == 2:
        return convert_bin_index_to_continuous(target, num_bins)
    else:
        return convert_bins_to_continuous(target)
