# adapted from openpi
"""Data loader for offline RL training on top of openpi LeRobot pipelines.

Each batch carries both the pi0 view (Observation / Actions) and the RL view
(next-step pixels + state + reward + mask). The dataset is built directly from
a LeRobot repo — we do **not** route through ``create_torch_dataset`` — so this
module owns the ``LeRobotDataset`` construction end to end and is free to plug
in whatever ``delta_timestamps`` the RL setup needs (current + next-step
state/image/wrist_image, plus a ``progress`` sequence spanning the full action
chunk).

The transform chain itself comes from the configured :class:`RLDataConfig` and
is applied via the shared ``transform_dataset`` helper — no transforms are
defined here. Robot-specific RL transforms live next to their supervised
counterparts (e.g. ``openpi.policies.libero_policy.LiberoRLInputs`` alongside
``LiberoInputs``).

Reward semantics (set up by the data config's transforms): when the LeRobot
dataset has a ``progress`` feature, the trainer sees the full per-step
progress sequence plus per-step increments; the canonical ``reward`` is the
endpoint delta over the action chunk. Without ``progress``, reward falls back
to terminal-only (r=1 when the action chunk runs past the episode end).
``mask=0`` when the next step is padded, 1 elsewhere.
"""

from __future__ import annotations

import logging

import jax

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.lerobot_patched as _lerobot_patched
import openpi.transforms as _transforms


def create_rl_dataset(
    data_config: _config.RLDataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    skip_norm_stats: bool = False,
) -> _data_loader.Dataset:
    """Build an RL-ready Dataset directly from the configured LeRobot repo.

    We own the ``LeRobotDataset`` construction here (rather than going through
    ``create_torch_dataset``) so the RL pipeline can plug in whatever
    ``delta_timestamps`` it needs — typically the H-step action chunk plus
    next-step views of state/image/wrist_image and the full progress sequence.

    The transform chain (repack → data → normalize → model) is applied via the
    shared ``transform_dataset`` helper, reading the transforms from
    ``data_config``.

    Requires an :class:`RLDataConfig` so the type system catches accidental
    misuse — e.g. handing in a supervised-training DataConfig whose transforms
    wouldn't produce the next-step keys the RL trainer expects.
    """
    if not isinstance(data_config, _config.RLDataConfig):
        raise TypeError(
            "create_rl_dataset requires an RLDataConfig (typically produced by an RL "
            f"DataConfigFactory like LeRobotLiberoRLDataConfig); got {type(data_config).__name__}."
        )
    if data_config.reward_source == "progress" and "progress" not in data_config.extra_delta_timestamps:
        raise ValueError(
            "reward_source='progress' but 'progress' is not in extra_delta_timestamps — "
            "the RL factory should have wired this; check LeRobotLiberoRLDataConfig.create()."
        )
    repo_id = data_config.repo_id
    if repo_id is None or repo_id == "fake":
        raise ValueError("RL data loader requires a real LeRobot repo_id (not 'fake').")
    if isinstance(data_config, _config.MultiDataConfig) and data_config.repo_ids:
        raise NotImplementedError("Multi-repo RL data loading is not supported yet.")

    meta = _lerobot_patched.PatchedLeRobotDatasetMetadata(repo_id)
    fps = float(meta.fps)

    # Build the lerobot delta_timestamps: standard action chunk + RL extras.
    delta_timestamps: dict[str, list[float]] = {
        key: [t / fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
    }
    for key, offsets in data_config.extra_delta_timestamps.items():
        delta_timestamps[key] = list(offsets)

    raw = _lerobot_patched.PatchedLeRobotDataset(
        repo_id, delta_timestamps=delta_timestamps, video_backend="pyav",
        episodes=list(data_config.episode_ids) if data_config.episode_ids else None,
    )
    dataset: _data_loader.Dataset = raw
    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(meta.tasks)]
        )

    # Apply the configured transform chain (repack → data → normalize → model)
    # via the shared helper — there are no RL-specific transforms here.
    return _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)


def create_rl_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = True,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> _data_loader.TorchDataLoader:
    """Build the offline RL data loader (TorchDataLoader on top of create_rl_dataset)."""
    if jax.process_count() > 1:
        raise NotImplementedError("Multi-process RL data loading is not supported.")

    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"[rl] data_config: {data_config}")

    dataset = create_rl_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
        skip_norm_stats=skip_norm_stats,
    )

    local_batch_size = config.batch_size // jax.process_count()
    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    # Duck-type the DataLoader protocol so checkpoints.save_state can read norm_stats.
    torch_loader.data_config = lambda: data_config  # type: ignore[attr-defined]
    return torch_loader
