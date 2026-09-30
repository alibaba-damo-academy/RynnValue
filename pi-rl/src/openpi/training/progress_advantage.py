# adapted from openpi
"""Progress-derived frame advantages for filtered behavior cloning (filter BC).

Filter BC trains a plain BC policy on the subset of demonstration frames that make
forward progress toward the task. Each frame carries a scalar ``progress`` label in
the lerobot episode parquet files; the per-frame reward is the progress increment
``r_t = progress[t + 1] - progress[t]`` and the filtering score is the
lambda-discounted return over a finite window of ``horizon`` steps,

    A_t = sum_{k < horizon} lambda^k * r_{t+k},   r_{t+k} = 0 past the episode end.

A frame is kept iff ``A_t`` strictly exceeds a per-repo threshold. By default the
threshold is zero (``criterion="zero"`` -- a frame is kept iff the window makes any
progress); ``"mean"``, ``"median"`` and ``"quantile"`` thresholds are also supported.
Thresholds and the resulting frame whitelists are computed offline from the lerobot
parquet files and cached to JSON.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import logging
import pathlib

from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np
import pyarrow.parquet as pq


def _resolve_repo_root(repo_id: str) -> pathlib.Path:
    """Resolve a lerobot repo id to a local directory (absolute path or HF home)."""
    path = pathlib.Path(repo_id)
    if path.exists():
        return path
    return pathlib.Path(HF_LEROBOT_HOME) / repo_id


def window_advantages(progress: np.ndarray, horizon: int, discount: float = 1.0) -> np.ndarray:
    """Per-frame lambda-discounted progress return over a ``horizon`` window.

    Per-frame rewards are progress increments; rewards past the episode end are zero
    (mirroring lerobot's boundary clipping), so tail frames see a discounted partial
    return. ``discount=1.0`` telescopes to the clamped delta
    ``progress[min(t+H, T)] - progress[t]``.
    """
    n = len(progress)
    if discount == 1.0:
        fut = np.minimum(np.arange(horizon, horizon + n), n - 1)
        return progress[fut] - progress
    rewards = np.empty(n, dtype=np.float64)
    rewards[:-1] = np.diff(progress)
    rewards[-1] = 0.0
    b = np.zeros(n + 1, dtype=np.float64)
    for t in range(n - 1, -1, -1):
        b[t] = rewards[t] + discount * b[t + 1]
    idx = np.minimum(np.arange(horizon, horizon + n), n)
    return b[:n] - discount**horizon * b[idx]


def compute_delta_threshold(
    repo_id: str,
    action_horizon: int,
    criterion: str = "zero",
    quantile: float = 0.7,
    max_windows: int = 200_000,
    seed: int = 0,
    discount: float = 1.0,
) -> float:
    """Threshold over window advantages ``A_t = sum_k lambda^k * r_{t+k}`` (lambda =
    ``discount``, per-frame rewards ``r = progress[t+1] - progress[t]``, window
    ``action_horizon``; ``discount=1.0`` reduces to the clamped delta
    ``progress[min(t+H, T)] - progress[t]``).

    Reduces per ``criterion``: ``"zero"`` (threshold fixed at 0 -- a frame is kept iff
    the window makes any progress), or a statistic of the advantage distribution
    computed by scanning every episode parquet: ``"mean"``, ``"median"`` or
    ``"quantile"`` (uses ``quantile``). Distribution-based criteria subsample to
    ``max_windows``.
    """
    if criterion == "zero":
        logging.info(f"[filter-bc] {repo_id}: advantage threshold (zero) = 0.0")
        return 0.0

    root = _resolve_repo_root(repo_id)
    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'data'}")

    deltas: list[np.ndarray] = []
    for file in files:
        col = pq.read_table(file, columns=["progress"]).column("progress")
        progress = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1)
        if len(progress) == 0:
            continue
        deltas.append(window_advantages(progress, action_horizon, discount))

    if not deltas:
        raise ValueError(f"No delta-progress windows in {repo_id}: empty episode parquet files.")

    all_deltas = np.concatenate(deltas)
    if len(all_deltas) > max_windows:
        rng = np.random.default_rng(seed)
        all_deltas = rng.choice(all_deltas, size=max_windows, replace=False)
    if criterion == "mean":
        threshold = float(all_deltas.mean())
    elif criterion == "median":
        threshold = float(np.median(all_deltas))
    elif criterion == "quantile":
        threshold = float(np.quantile(all_deltas, quantile))
    else:
        raise ValueError(f"Unknown advantage criterion: {criterion!r}")
    logging.info(
        f"[filter-bc] {repo_id}: advantage threshold ({criterion}) = {threshold:.6f} "
        f"({len(all_deltas)} windows, mean={float(all_deltas.mean()):.6f})"
    )
    return threshold


def load_or_compute_thresholds(
    repo_ids: Sequence[str],
    cache_path: pathlib.Path | str,
    action_horizon: int,
    criterion: str = "zero",
    quantile: float = 0.7,
    overrides: Mapping[str, float] | None = None,
    discount: float = 1.0,
) -> dict[str, float]:
    """Per-repo advantage thresholds with a JSON cache.

    Resolution order per repo: ``overrides`` > existing cache entry > freshly computed
    (the cache is updated as new entries are computed). Delete the cache file to force
    recomputation after the underlying data changes.
    """
    cache_path = pathlib.Path(cache_path)
    cached: dict[str, float] = {}
    if cache_path.exists():
        cached = {str(k): float(v) for k, v in json.loads(cache_path.read_text()).items()}

    thresholds: dict[str, float] = {}
    for repo_id in repo_ids:
        if overrides is not None and repo_id in overrides:
            thresholds[repo_id] = float(overrides[repo_id])
            continue
        if repo_id in cached:
            thresholds[repo_id] = cached[repo_id]
            continue
        value = compute_delta_threshold(repo_id, action_horizon, criterion, quantile, discount=discount)
        cached[repo_id] = value
        thresholds[repo_id] = value
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached, indent=2))
        logging.info(f"[filter-bc] cached advantage threshold for {repo_id} at {cache_path}")
    return thresholds


def compute_filtered_indices(
    repo_id: str,
    advantage_horizon: int,
    threshold: float,
    discount: float = 1.0,
) -> list[int]:
    """Global lerobot frame ``index`` values kept by the filter-BC criterion
    ``A_t > threshold``.

    Scans every episode parquet (``index`` + ``progress`` columns only), so the
    result aligns row-for-row with the hf dataset built from the same files.
    """
    root = _resolve_repo_root(repo_id)
    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'data'}")

    kept: list[int] = []
    for file in files:
        table = pq.read_table(file, columns=["index", "progress"])
        indices = np.asarray(table.column("index").to_numpy(zero_copy_only=False)).reshape(-1)
        progress = np.asarray(table.column("progress").to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1)
        if len(progress) == 0:
            continue
        advantages = window_advantages(progress, advantage_horizon, discount)
        kept.extend(indices[advantages > threshold].astype(np.int64).tolist())
    logging.info(
        f"[filter-bc] {repo_id}: kept {len(kept)} frames (A_t > {threshold:.6f}, "
        f"W={advantage_horizon}, lambda={discount})"
    )
    return kept


def load_or_compute_filtered_indices(
    repo_ids: Sequence[str],
    cache_path: pathlib.Path | str,
    advantage_horizon: int,
    thresholds: Mapping[str, float],
    discount: float = 1.0,
) -> dict[str, list[int]]:
    """Per-repo filter-BC frame-index whitelists with a JSON cache
    (``{repo_id: [global frame index, ...]}``). Cache entries are reused when
    present; delete the cache file to force recomputation.
    """
    cache_path = pathlib.Path(cache_path)
    cached: dict[str, list[int]] = {}
    if cache_path.exists():
        cached = {str(k): [int(i) for i in v] for k, v in json.loads(cache_path.read_text()).items()}

    result: dict[str, list[int]] = {}
    for repo_id in repo_ids:
        if repo_id in cached:
            result[repo_id] = cached[repo_id]
            continue
        if repo_id not in thresholds:
            raise ValueError(f"No advantage threshold for {repo_id}; cannot compute filter indices.")
        indices = compute_filtered_indices(repo_id, advantage_horizon, thresholds[repo_id], discount)
        cached[repo_id] = indices
        result[repo_id] = indices
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached))
        logging.info(f"[filter-bc] cached filter indices for {repo_id} at {cache_path}")
    return result
