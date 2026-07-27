# adapted from openpi
"""IQL-specific checkpoint manager.

Extends :class:`orbax.checkpoint.CheckpointManager` with an additional
``iql_state`` orbax item so scripts/train_iql.py can persist the IQL
critic / target_critic / value (params + opt_state + batch_stats) atomically
alongside the pi0 ``train_state`` and ``params`` items.

This module is intentionally separate from :mod:`openpi.training.checkpoints`:
the pi0-only callers (``scripts/train.py``) keep using that module untouched.
The save / restore helpers below mirror the originals' shape but always
include ``iql_state`` -- they reuse the private ``_split_params`` /
``_merge_params`` / ``CallbackHandler`` helpers from the base module rather
than duplicating them.
"""
from __future__ import annotations

import logging

from etils import epath
import orbax.checkpoint as ocp

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.checkpoints as _checkpoints
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


class IQLCheckpointManager(ocp.CheckpointManager):
    """``ocp.CheckpointManager`` preconfigured with an ``iql_state`` item.

    Subclassed (rather than instantiated directly) so callers get a typed
    handle for the IQL flow and don't accidentally pass it where a pi0-only
    manager is expected. All super-class behavior (save / restore / step
    tracking / async finalization) is inherited unchanged.
    """

    def __init__(self, checkpoint_dir: epath.Path | str, *, keep_period: int | None):
        super().__init__(
            epath.Path(checkpoint_dir),
            item_handlers={
                "assets": _checkpoints.CallbackHandler(),
                "train_state": ocp.PyTreeCheckpointHandler(),
                "params": ocp.PyTreeCheckpointHandler(),
                # Holds the IQL critic / target_critic_params / value pytree
                # (host-side, unreplicated). Restored atomically with pi0.
                "iql_state": ocp.PyTreeCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(
                max_to_keep=1,
                keep_period=keep_period,
                create=False,
                async_options=ocp.AsyncOptions(timeout_secs=7200),
            ),
        )


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[IQLCheckpointManager, bool]:
    """Same overwrite/resume semantics as :func:`openpi.training.checkpoints.initialize_checkpoint_dir`,
    but returns an :class:`IQLCheckpointManager` with the ``iql_state`` item registered."""
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mngr = IQLCheckpointManager(checkpoint_dir, keep_period=keep_period)

    # Mirror the base module's "no real checkpoint yet -> don't resume" guard.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: IQLCheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    iql_state: at.PyTree,
) -> None:
    """Save pi0 train_state / params + IQL critic/target/value atomically."""

    def save_assets(directory: epath.Path):
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / _checkpoints._asset_subdir_name(data_config.asset_id), norm_stats)

    with at.disable_typechecking():
        train_state, params = _checkpoints._split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
        "iql_state": iql_state,
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: IQLCheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    iql_state: at.PyTree,
    step: int | None = None,
) -> tuple[training_utils.TrainState, at.PyTree]:
    """Restore pi0 train_state + IQL state in a single orbax call.

    ``iql_state`` must be a freshly-initialized template (e.g. from
    ``PiIQLLearner.iql_save_pytree()``); orbax uses its structure to drive
    deserialization and overwrites the leaves in place.
    """
    del data_loader

    with at.disable_typechecking():
        train_state, params = _checkpoints._split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
                "iql_state": iql_state,
            },
        )
    merged = _checkpoints._merge_params(restored["train_state"], restored["params"])
    return merged, restored["iql_state"]
