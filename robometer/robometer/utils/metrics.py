#!/usr/bin/env python3
"""
Utility functions for computing evaluation metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_spearman_correlation(
    pred: torch.Tensor, target: torch.Tensor, aggregate: bool = True, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute Spearman correlation between prediction and target tensors.

    Args:
        pred: Prediction tensor
        target: Target tensor
        aggregate: If True and tensors are 2D, returns mean correlation. If False and 2D, returns per-sample correlations.
        mask: Optional mask tensor of shape (N, T) with 1.0 for valid frames, 0.0 for masked frames. Only used for 2D tensors.

    Returns:
        Spearman correlation coefficient (scalar for 1D or aggregate=True, tensor for 2D with aggregate=False)
    """
    from scipy.stats import spearmanr

    # NumPy doesn't support bf16/half; cast to float32 before moving to CPU
    pred_f32 = pred.detach().to(dtype=torch.float32)
    target_f32 = target.detach().to(dtype=torch.float32)

    pred_np = pred_f32.cpu().numpy()
    target_np = target_f32.cpu().numpy()

    # Handle 1D arrays
    if pred_np.ndim == 1 and target_np.ndim == 1:
        correlation, _ = spearmanr(pred_np, target_np)
        if np.isnan(correlation):
            correlation = 0.0
        return torch.tensor(correlation, device=pred.device, dtype=torch.float32)

    # Handle 2D arrays (batch, sequence)
    elif pred_np.ndim == 2 and target_np.ndim == 2:
        if mask is not None:
            # Convert mask to numpy
            mask_np = mask.detach().to(dtype=torch.float32).cpu().numpy()
            # Ensure mask matches shape
            if mask_np.shape != pred_np.shape:
                raise ValueError(f"Mask shape {mask_np.shape} must match pred shape {pred_np.shape}")

        correlations = []
        for i, (p, t) in enumerate(zip(pred_np, target_np, strict=False)):
            # Apply mask if provided
            if mask is not None:
                valid_mask = mask_np[i] > 0.0
                p = p[valid_mask]
                t = t[valid_mask]

            if len(p) > 1 and len(t) > 1:  # Need at least 2 points for correlation
                corr, _ = spearmanr(p, t)
                if not np.isnan(corr):
                    correlations.append(corr)
                else:
                    correlations.append(0.0)
            else:
                correlations.append(0.0)

        if correlations:
            correlations_tensor = torch.tensor(correlations, device=pred.device, dtype=torch.float32)
            if aggregate:
                return correlations_tensor.mean()
            else:
                return correlations_tensor
        else:
            result = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
            return result if aggregate else result.unsqueeze(0).expand(pred.shape[0])

    else:
        raise ValueError(f"Unsupported tensor dimensions: pred={pred_np.ndim}, target={target_np.ndim}")


def compute_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Area Under the Curve (AUC) for binary classification.

    Args:
        scores: Model prediction scores/logits
        labels: Binary labels (0 or 1)

    Returns:
        AUC score
    """
    try:
        from sklearn.metrics import roc_auc_score

        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # Handle edge cases
        if len(np.unique(labels_np)) < 2:
            return 0.5  # Default AUC for single class

        auc = roc_auc_score(labels_np, scores_np)
        return auc

    except ImportError:
        # Fallback implementation if sklearn is not available
        return manual_auc(scores, labels)


def manual_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Manual implementation of AUC as fallback.

    Args:
        scores: Model prediction scores/logits
        labels: Binary labels (0 or 1)

    Returns:
        AUC score
    """
    # Sort by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices]

    # Count positive and negative samples
    n_pos = torch.sum(labels == 1).item()
    n_neg = torch.sum(labels == 0).item()

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Default AUC for single class

    # Calculate true positive rate and false positive rate
    tp = 0
    fp = 0
    prev_score = float("inf")

    area = 0.0

    for i, label in enumerate(sorted_labels):
        if scores[sorted_indices[i]] != prev_score:
            # Calculate area under the curve
            area += trapezoid_area(fp / n_neg, tp / n_pos, fp / n_neg, tp / n_pos)
            prev_score = scores[sorted_indices[i]]

        if label == 1:
            tp += 1
        else:
            fp += 1

    # Add final area
    area += trapezoid_area(fp / n_neg, tp / n_pos, 1.0, 1.0)

    return area


def trapezoid_area(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate area of trapezoid using trapezoid rule.

    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates

    Returns:
        Area of trapezoid
    """
    return (x2 - x1) * (y1 + y2) / 2


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute accuracy for binary classification.

    Args:
        predictions: Binary predictions (0 or 1)
        targets: Binary targets (0 or 1)

    Returns:
        Accuracy score
    """
    return (predictions == targets).float().mean().item()


def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Squared Error.

    Args:
        predictions: Prediction tensor
        targets: Target tensor

    Returns:
        MSE value
    """
    return F.mse_loss(predictions, targets).item()


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        predictions: Prediction tensor
        targets: Target tensor

    Returns:
        MAE value
    """
    return F.l1_loss(predictions, targets).item()
