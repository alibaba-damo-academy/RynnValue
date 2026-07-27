# adapted from openpi
"""LeRobot dataset subclasses with the ``episodes=`` filter index bug fixed.

The stock :func:`lerobot.common.datasets.utils.get_episode_data_index` returns
position-indexed tensors of length ``len(whitelist)``, but
:meth:`LeRobotDataset._get_query_indices` looks them up using the *original* episode
index from each row, which goes out of bounds whenever the whitelist drops episodes
preceding the highest retained ``ep_idx``.

The fix keeps a slot at every original ``ep_idx`` -- whitelisted episodes contribute
their length to the running cumulative sum, dropped episodes contribute 0. After the
prefix sum the resulting tensors are still indexable by the original ``ep_idx``, and the
values for whitelisted episodes correctly reference rows in the *filtered* hf_dataset.

We override the constructors instead of monkey-patching:
  * :class:`PatchedLeRobotDataset` reuses the parent ``__init__`` but overrides the single
    line that builds ``episode_data_index`` via :meth:`_init_episode_data_index`.
  * :class:`PatchedMultiLeRobotDataset` reproduces ``MultiLeRobotDataset.__init__`` so the
    inner list comprehension constructs our subclass instead of the stock one.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import accumulate
import logging
from pathlib import Path

import lerobot.common.datasets.lerobot_dataset as _ld
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.compute_stats import aggregate_stats
import torch


def patched_get_episode_data_index(
    episode_dicts: dict[int, dict],
    episodes: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Position-aligned variant of ``lerobot.common.datasets.utils.get_episode_data_index``.

    Episodes outside ``episodes`` keep their slot but contribute length=0; whitelisted
    episodes contribute their actual length. After cumulative-sum the resulting tensors
    are indexable by the original ``ep_idx`` and reference rows in the *filtered* dataset.
    """
    episode_lengths = {
        ep_idx: ep_dict["length"] if episodes is None or ep_idx in episodes else 0
        for ep_idx, ep_dict in episode_dicts.items()
    }
    cumulative_lengths = list(accumulate(episode_lengths.values()))
    return {
        "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
        "to": torch.LongTensor(cumulative_lengths),
    }


# Re-exported as a passthrough so callers can import every LeRobot dataset class from this
# module. ``LeRobotDatasetMetadata`` doesn't carry the ``episode_data_index`` bug (it only
# loads info.json / episodes.jsonl), so no override is needed -- the alias just keeps imports
# consistent with ``PatchedLeRobotDataset`` / ``PatchedMultiLeRobotDataset``.
PatchedLeRobotDatasetMetadata = _ld.LeRobotDatasetMetadata


class PatchedLeRobotDataset(_ld.LeRobotDataset):
    """``LeRobotDataset`` subclass with the position-indexed ``episode_data_index`` fix.

    Subclassing strategy: let the parent ``__init__`` do all of its setup, then replace
    ``self.episode_data_index`` with the position-aligned version. The parent's call to
    ``check_timestamps_sync`` (which expects the stock layout) is bypassed for the
    duration of ``super().__init__`` -- the upstream check is too strict for progress-
    labeled Franka datasets that contain occasional timestamp jumps (e.g. a single pair
    of consecutive frames with ``[33.57, 0.0]``).
    """

    def __init__(self, *args, **kwargs):
        import lerobot.common.datasets.utils as _ld_utils

        _orig_check = getattr(_ld_utils, "check_timestamps_sync", None)

        def _noop_check(*a, **kw):
            return None

        if _orig_check is not None:
            _ld_utils.check_timestamps_sync = _noop_check
            # The LeRobotDataset module captured ``check_timestamps_sync`` at import time
            # via ``from .utils import ...``, so also patch the reference on the dataset
            # module that actually invokes it inside ``__init__``.
            _orig_ld_check = getattr(_ld, "check_timestamps_sync", None)
            _ld.check_timestamps_sync = _noop_check
        else:
            _orig_ld_check = None

        try:
            super().__init__(*args, **kwargs)
        finally:
            if _orig_check is not None:
                _ld_utils.check_timestamps_sync = _orig_check
            if _orig_ld_check is not None:
                _ld.check_timestamps_sync = _orig_ld_check

        # Re-build episode_data_index keyed by *original* ep_idx (bug fix).
        self.episode_data_index = patched_get_episode_data_index(self.meta.episodes, self.episodes)


class PatchedMultiLeRobotDataset(_ld.MultiLeRobotDataset):
    """Reproduces :class:`MultiLeRobotDataset.__init__` so inner sub-datasets are
    :class:`PatchedLeRobotDataset` instead of the stock :class:`LeRobotDataset`."""

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        # Skip MultiLeRobotDataset.__init__ entirely (it would build stock LeRobotDatasets).
        # Hop straight to its parent (object/torch.utils.data.Dataset) for the base setup.
        super(_ld.MultiLeRobotDataset, self).__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 1e-4)
        self._datasets = [
            PatchedLeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]

        # Same feature-intersection logic as the parent (kept verbatim so disabled_features
        # behave identically).
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])
