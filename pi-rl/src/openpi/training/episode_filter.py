# adapted from openpi
"""Episode-id whitelisting for MultiLeRobotDataset.

The RynnValue ``RobotwinStorage/filter.json`` lists exactly which episodes from each
lerobot dump should be used for training. The file is large (~24 MB) and slow to parse,
so we cache a compact ``episodes.json`` next to the dataset's ``norm_stats.json`` and
prefer that on subsequent runs.

The mapping consumed by ``MultiLeRobotDataset(episodes=...)`` is
``{absolute_repo_path: [ep_idx, ...]}``. Filter.json stores dataset-*relative* repo paths
(e.g. ``lerobot_robotwin_eef_aug_500/click_alarmclock-aloha-agilex_randomized_500-1000``),
so we resolve via suffix-match against the absolute paths the factory discovered.
"""

from __future__ import annotations

import json
import logging
import pathlib
from collections.abc import Mapping, Sequence

# In-memory format used throughout the rest of the codebase.
EpisodeIdMapping = Mapping[str, tuple[int, ...]]


def _parse_filter_episodes(raw: dict) -> dict[str, list[int]]:
    """Read RynnValue filter.json -> {relative_repo_id: [ep_idx, ...]} (sorted, deduped)."""
    if "episodes" not in raw:
        raise ValueError("filter.json is missing the top-level 'episodes' field.")
    by_repo: dict[str, set[int]] = {}
    for entry in raw["episodes"]:
        meta = entry.get("metadata") or {}
        rel = meta.get("repo_id")
        ep_idx = meta.get("ep_idx")
        if rel is None or ep_idx is None:
            continue
        by_repo.setdefault(rel, set()).add(int(ep_idx))
    return {rel: sorted(eps) for rel, eps in by_repo.items()}


def _suffix_match_to_absolute(
    rel_to_eps: Mapping[str, Sequence[int]],
    abs_repo_ids: Sequence[str],
) -> tuple[dict[str, tuple[int, ...]], list[str], list[str]]:
    """Match each ``rel`` to the unique abs path that ends with ``/<rel>``.

    Returns ``(resolved, unmatched_rel, unmatched_abs)`` -- unmatched lists are the
    relative ids that found no abs path, and the abs ids that no relative entry covers.
    """
    resolved: dict[str, tuple[int, ...]] = {}
    unmatched_rel: list[str] = []
    matched_abs: set[str] = set()
    for rel, eps in rel_to_eps.items():
        # The relative key from filter.json is a path suffix of one (or zero) absolute repo path.
        candidates = [a for a in abs_repo_ids if a == rel or a.endswith("/" + rel)]
        if not candidates:
            unmatched_rel.append(rel)
            continue
        if len(candidates) > 1:
            # If more than one absolute path matches (extremely unlikely), prefer the longest --
            # it's the most specific match.
            candidates.sort(key=len, reverse=True)
        chosen = candidates[0]
        resolved[chosen] = tuple(int(e) for e in eps)
        matched_abs.add(chosen)
    unmatched_abs = [a for a in abs_repo_ids if a not in matched_abs]
    return resolved, unmatched_rel, unmatched_abs


def load_from_filter_json(
    filter_json_path: str | pathlib.Path,
    abs_repo_ids: Sequence[str],
    *,
    strict: bool = False,
) -> EpisodeIdMapping:
    """Parse filter.json and resolve relative repo ids to the given absolute paths.

    Args:
        filter_json_path: Path to the RynnValue filter.json.
        abs_repo_ids: The discovered absolute repo paths (output of the RoboTwin glob).
        strict: If True, raise when any abs_repo_id has no episode entry in the filter.
            If False (default), abs paths without an entry are dropped from the mapping --
            ``MultiLeRobotDataset`` will then load *all* episodes from those sub-datasets.

    Returns ``{absolute_repo_path: (ep_idx, ...)}`` suitable for
    ``MultiLeRobotDataset(episodes=...)``.
    """
    path = pathlib.Path(filter_json_path)
    raw = json.loads(path.read_text())
    rel_to_eps = _parse_filter_episodes(raw)
    resolved, unmatched_rel, unmatched_abs = _suffix_match_to_absolute(rel_to_eps, abs_repo_ids)
    if unmatched_rel:
        logging.info(
            "episode_filter: %d filter entries had no matching abs repo (e.g. %s)",
            len(unmatched_rel),
            unmatched_rel[0],
        )
    if unmatched_abs:
        msg = (
            f"episode_filter: {len(unmatched_abs)} abs repo(s) have no entry in {path.name}: "
            f"first={unmatched_abs[0]}"
        )
        if strict:
            raise ValueError(msg + ". Pass strict=False to ignore.")
        logging.warning(msg + " -- these sub-datasets will use ALL episodes.")
    logging.info(
        "episode_filter: %d sub-datasets filtered, %d total episodes kept",
        len(resolved),
        sum(len(v) for v in resolved.values()),
    )
    return resolved


def save_compact(
    out_path: str | pathlib.Path,
    mapping: EpisodeIdMapping,
) -> None:
    """Dump ``{abs_repo_id: [ep_idx,...]}`` to ``out_path`` as a small JSON file."""
    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {repo_id: list(eps) for repo_id, eps in mapping.items()}
    path.write_text(json.dumps(payload, indent=2))


def load_compact(in_path: str | pathlib.Path) -> EpisodeIdMapping:
    """Read a compact episodes.json written by ``save_compact``."""
    raw = json.loads(pathlib.Path(in_path).read_text())
    return {repo_id: tuple(int(e) for e in eps) for repo_id, eps in raw.items()}


def load_or_resolve(
    *,
    abs_repo_ids: Sequence[str],
    filter_json_path: str | pathlib.Path | None,
    compact_cache_path: str | pathlib.Path | None,
    strict: bool = False,
) -> EpisodeIdMapping | None:
    """Pick the cheapest source of an episode_id mapping.

    Resolution order:
      1. If ``compact_cache_path`` exists -> load it (fast).
      2. Else if ``filter_json_path`` is set AND exists on disk -> parse + resolve.
      3. Else -> return None (no filter; MultiLeRobotDataset uses all episodes).

    Returns None when no source is available. A configured-but-missing
    ``filter_json_path`` warns and falls through (instead of FileNotFoundError),
    so configs that hard-code the central RynnValue path still work on machines
    where the file isn't mounted (e.g. eval-only boxes).
    """
    if compact_cache_path is not None:
        path = pathlib.Path(compact_cache_path)
        if path.exists():
            logging.info(f"episode_filter: loading compact cache from {path}")
            return load_compact(path)
    if filter_json_path is not None:
        fpath = pathlib.Path(filter_json_path)
        if not fpath.exists():
            logging.warning(
                "episode_filter: configured filter_json_path %s does not exist and no "
                "compact cache was found either -- falling back to using ALL episodes. "
                "Run compute_norm_stats_fast on a machine that can see the filter to "
                "populate the cache, or set episode_filter_path=None to silence this.",
                fpath,
            )
            return None
        return load_from_filter_json(fpath, abs_repo_ids, strict=strict)
    return None
