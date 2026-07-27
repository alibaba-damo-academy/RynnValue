from typing import Any

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import kendalltau
import itertools


def compute_pearson(y_true: list[float], y_pred: list[float]) -> float:
    """Compute Pearson correlation, robust to constant inputs; returns np.nan if undefined."""

    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return np.nan
    # If either vector is constant, pearsonr returns nan; keep that behavior
    try:
        corr, _ = pearsonr(a, b)
    except Exception:
        corr = np.nan
    return corr


def compute_spearman(y_true: list[float], y_pred: list[float]) -> float:
    """Compute Spearman correlation, robust to constant inputs; returns np.nan if undefined."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return np.nan
    try:
        corr, _ = spearmanr(a, b)
    except Exception:
        corr = np.nan
    return corr


def kendall_tau_a(x, y):
    C = D = 0
    n = len(x)
    for i, j in itertools.combinations(range(n), 2):
        dx = np.sign(x[i] - x[j])
        dy = np.sign(y[i] - y[j])
        if dx == 0 or dy == 0:
            continue  # τ-a ignores ties
        if dx == dy:
            C += 1
        else:
            D += 1
    return (C - D) / (n * (n - 1) / 2)


def compute_kendall(y_true: list[float], y_pred: list[float]) -> float:
    """Compute Kendall tau-b correlation, which handles ties better than Spearman.

    Kendall tau-b is specifically adapted to handle ties in the data.
    Returns np.nan if undefined or if inputs are invalid.

    See: https://stackoverflow.com/questions/10711395/spearman-correlation-and-ties
    """
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return np.nan
    # try:
    #     # variant='b' specifies Kendall tau-b, which handles ties
    #     corr, _ = kendalltau(a, b, variant='a')
    # except Exception:
    #     import ipdb; ipdb.set_trace()
    #     corr = np.nan
    corr = kendall_tau_a(a, b)
    return corr


def compute_preference_accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute preference accuracy over a list of result dicts.
    Expects keys 'predicted_preference' and 'preference_label' per sample.
    Returns dict with accuracy, correct, total, and skipped counts.
    """
    correct = 0
    total = 0
    skipped = 0
    for r in results:
        pred = r.get("predicted_preference")
        label = r.get("preference_label")
        if pred is None or label is None:
            skipped += 1
            continue
        if pred == label:
            correct += 1
        total += 1
    acc = (correct / total) if total > 0 else 0.0
    return {
        "preference_accuracy": acc,
        "num_correct": correct,
        "num_total": total,
        "num_skipped": skipped,
    }


def compute_preference_accuracy_from_progress(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute preference accuracy by using the final progress predictions."""
    correct = 0
    total = 0
    skipped = 0
    for r in results:
        progress_chosen = r.get("progress_pred_chosen")[-1]
        progress_rejected = r.get("progress_pred_rejected")[-1]
        label = r.get("preference_label")
        if progress_chosen is None or progress_rejected is None or label is None:
            skipped += 1
            continue
        if progress_chosen > progress_rejected:
            correct += 1
        total += 1
    acc = (correct / total) if total > 0 else None
    return {
        "preference_accuracy": acc,
        "num_correct": correct,
        "num_total": total,
        "num_skipped": skipped,
    }
