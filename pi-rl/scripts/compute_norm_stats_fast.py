# adapted from openpi
"""Fast variant of ``compute_norm_stats.py`` for state/action norm stats.

Differences vs. the stock script:

* Exposes ``--num-workers`` and ``--batch-size`` CLI flags (the stock script wires
  ``TrainConfig.num_workers``, which usually defaults to 2 -- way too low for an
  offline scan over hundreds of thousands of frames).
* Adds ``--skip-video`` (default on): patches each underlying ``LeRobotDataset``
  so its ``_query_videos`` becomes a no-op, then injects zero placeholders for
  the camera keys before the transform pipeline. Norm stats only depend on
  ``state``/``actions``, so the MP4/AV1 frame fetch is pure waste here.

Output paths and the meaning of the saved ``norm_stats.json`` are identical to the
stock script -- this is purely a faster way to produce the same artifact.

Examples:

  # Single task
  python scripts/compute_norm_stats_fast.py \\
      --config-name pi05_robotwin \\
      --repo-id /path/to/datasets/adjust_bottle-demo_clean_collect_200-50

  # Multi-task (50 RoboTwin sub-tasks pulled from the registered config)
  python scripts/compute_norm_stats_fast.py --config-name pi05_robotwin_all
"""

import dataclasses
import os

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.episode_filter as _episode_filter
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _empty_query_videos(query_timestamps, ep_idx):  # noqa: ARG001  (signature must match LeRobot)
    """Module-level no-op so it survives multiprocessing pickling (workers use spawn)."""
    return {}


def _disable_videos_inplace(dataset) -> dict[str, tuple[int, ...]]:
    """Patch one or more LeRobotDataset(s) so __getitem__ skips video decode.

    Strategy: replace ``_query_videos`` with a no-op that returns an empty dict. The
    ``len(meta.video_keys) > 0`` gate inside ``LeRobotDataset.__getitem__`` still evaluates
    True so the (cheap) timestamp-query bookkeeping runs, but the actual MP4/AV1 frame
    fetch is skipped. ``meta.video_keys`` itself is a property without a setter, so we
    cannot zero it out directly.

    Returns ``{camera_key -> shape}`` so the caller can fabricate zero placeholders that
    keep downstream transforms (``RepackTransform``, ``ResizeImages``, ...) happy.
    """
    sub_datasets = getattr(dataset, "_datasets", [dataset])
    cam_shapes: dict[str, tuple[int, ...]] = {}
    for sub in sub_datasets:
        meta = getattr(sub, "meta", None)
        if meta is None:
            continue
        for key in list(getattr(meta, "video_keys", []) or []):
            feat = meta.info["features"].get(key, {})
            shape = tuple(feat.get("shape", (3, 480, 640)))
            cam_shapes[key] = shape
        # Module-level function (not lambda) so spawn-mode DataLoader workers can pickle it.
        sub._query_videos = _empty_query_videos
    return cam_shapes


class InjectZeroImages(transforms.DataTransformFn):
    """Insert zero-byte placeholders for camera keys; runs before the repack transform."""

    def __init__(self, cam_shapes: dict[str, tuple[int, ...]]):
        self._cam_shapes = cam_shapes

    def __call__(self, x: dict) -> dict:
        for key, shape in self._cam_shapes.items():
            if key not in x:
                x[key] = np.zeros(shape, dtype=np.uint8)
        return x


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    skip_video: bool = True,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None and not isinstance(data_config, _config.MultiDataConfig):
        raise ValueError("Data config must have a repo_id (or be a MultiDataConfig).")
    # make_torch_dataset dispatches: single repo -> create_torch_dataset; multi -> create_multi_torch_dataset.
    dataset = _data_loader.make_torch_dataset(data_config, action_horizon, model_config)

    pre_transforms: list[transforms.DataTransformFn] = []
    if skip_video:
        # Walk through TransformedDataset wrapper(s) to reach the raw LeRobot dataset(s).
        raw = dataset
        while hasattr(raw, "_dataset"):
            raw = raw._dataset
        cam_shapes = _disable_videos_inplace(raw)
        if cam_shapes:
            pre_transforms.append(InjectZeroImages(cam_shapes))
            print(f"--skip-video: zeroing camera keys {sorted(cam_shapes)} (no MP4 decode)")

    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *pre_transforms,
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    repo_id: str | None = None,
    num_workers: int | None = None,
    batch_size: int | None = None,
    skip_video: bool = True,
    episode_filter_path: str | None = None,
):
    """Compute norm stats fast.

    Args:
        config_name: Registered TrainConfig name.
        max_frames: Optional cap on the number of frames to scan.
        repo_id: Optional override of the dataset repo id (useful when one TrainConfig is
            reused across multiple sub-datasets, e.g. per-task RoboTwin dumps).
        num_workers: PyTorch DataLoader worker processes. Default ``min(16, cpu_count)``.
            The stock script uses ``TrainConfig.num_workers`` which is typically 2.
        batch_size: Loader batch size. Default ``TrainConfig.batch_size``.
        skip_video: Skip MP4/AV1 video decode (default True). Norm stats only depend on
            state/actions; we patch ``_query_videos`` to a no-op and inject zero image
            placeholders. Set False if you ever extend norm stats to depend on pixels.
        episode_filter_path: Override the episode filter path. Set to /dev/null to disable
            filtering (use all episodes).
    """
    config = _config.get_config(config_name)
    if repo_id is not None:
        new_assets = dataclasses.replace(config.data.assets, asset_id=None)
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=repo_id, assets=new_assets))
    if episode_filter_path is not None and hasattr(config.data, "episode_filter_path"):
        config = dataclasses.replace(
            config, data=dataclasses.replace(config.data, episode_filter_path=episode_filter_path)
        )
    data_config = config.data.create(config.assets_dirs, config.model)

    effective_num_workers = num_workers if num_workers is not None else min(16, os.cpu_count() or 4)
    effective_batch_size = batch_size if batch_size is not None else config.batch_size
    print(
        f"compute_norm_stats_fast: num_workers={effective_num_workers} "
        f"batch_size={effective_batch_size} skip_video={skip_video}"
    )

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, effective_batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            effective_batch_size,
            config.model,
            effective_num_workers,
            max_frames,
            skip_video=skip_video,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    # If this is a MultiDataConfig with a resolved episode whitelist, dump a compact
    # episodes.json next to norm_stats.json. The factory prefers this on subsequent runs
    # over re-parsing the (slow) RynnValue filter.json.
    if isinstance(data_config, _config.MultiDataConfig) and data_config.episode_ids:
        episodes_path = output_path / "episodes.json"
        print(f"Writing compact episodes cache to: {episodes_path}")
        _episode_filter.save_compact(episodes_path, data_config.episode_ids)


if __name__ == "__main__":
    tyro.cli(main)
