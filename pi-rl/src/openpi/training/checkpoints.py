# adapted from openpi
from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import pathlib
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def _asset_subdir_name(asset_id: str) -> str:
    # `directory / asset_id` collapses to just `asset_id` when the latter is absolute,
    # which would write norm_stats back into the dataset folder. Strip to the basename
    # so the file lands inside the checkpoint's assets/ tree as intended.
    p = pathlib.PurePosixPath(asset_id)
    return p.name if p.is_absolute() else asset_id


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
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

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / _asset_subdir_name(data_config.asset_id), norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    # Prefer the in-checkpoint location written by ``save_assets`` (basename when
    # asset_id is absolute). Fall back to the legacy raw-asset_id path so older
    # checkpoints that still rely on reading directly from the dataset directory
    # keep working.
    assets_dir = epath.Path(assets_dir)
    subdir_name = _asset_subdir_name(asset_id)
    primary = assets_dir / subdir_name if subdir_name else assets_dir
    try:
        norm_stats = _normalize.load(primary)
        logging.info(f"Loaded norm stats from {primary}")
        return norm_stats
    except FileNotFoundError:
        # When asset_id is empty (e.g. serve-time config has no repo_id), try to
        # discover norm_stats from the single subdirectory inside assets_dir.
        if not subdir_name:
            discovered = _discover_norm_stats_subdir(assets_dir)
            if discovered is not None:
                norm_stats = _normalize.load(discovered)
                logging.info(f"Loaded norm stats from {discovered} (auto-discovered)")
                return norm_stats
            raise FileNotFoundError(
                f"Norm stats file not found at: {primary / 'norm_stats.json'} "
                f"and no subdirectory with norm_stats.json found in {assets_dir}"
            )
        legacy = assets_dir / asset_id
        if str(legacy) == str(primary):
            raise
        norm_stats = _normalize.load(legacy)
        logging.info(f"Loaded norm stats from {legacy}")
        return norm_stats


def _discover_norm_stats_subdir(assets_dir: epath.Path) -> epath.Path | None:
    """Scan assets_dir for a subdirectory containing norm_stats.json."""
    local_path = pathlib.Path(str(assets_dir))
    if not local_path.is_dir():
        return None
    candidates = [
        d for d in local_path.iterdir()
        if d.is_dir() and (d / "norm_stats.json").exists()
    ]
    if len(candidates) == 1:
        return epath.Path(str(candidates[0]))
    if len(candidates) > 1:
        # Multiple subdirs found; pick the first alphabetically and warn.
        candidates.sort()
        logging.warning(
            "Multiple norm_stats.json found in %s: %s. Using %s.",
            assets_dir, [c.name for c in candidates], candidates[0].name,
        )
        return epath.Path(str(candidates[0]))
    return None


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
