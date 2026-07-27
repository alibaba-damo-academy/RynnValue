# adapted from openpi
"""IQL fine-tuning for openpi pi0/pi05 policies.

Per step we:
  1. Update the jaxrl2 PixelIQL critic/value (pmap across local devices).
  2. Compute IQL advantage weights for the current batch.
  3. Update the pi0 flow-matching policy using BC loss weighted by `exp(A_scaling * adv)`.

The data pipeline (see `openpi.training.rl_data_loader`) yields each batch as a
single dict that carries both the pi0 view (state / image / tokenized_prompt /
actions) and the IQL view (next_state / next_image_base / next_image_wrist /
reward / mask). Rewards are synthesized: r=1 and mask=0 at episode end.
"""
import dataclasses
import functools
import logging
import platform
import time
from collections.abc import Sequence
from typing import Any

import etils.epath as epath
from flax.core.frozen_dict import FrozenDict
import flax.nnx as nnx
import flax.traverse_util as traverse_util
from flax.training import common_utils
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.iql_checkpoints as _checkpoints
import openpi.training.rl_data_loader as _rl_data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


# ---------------------------------------------------------------------------
# logging / wandb
# ---------------------------------------------------------------------------


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


# ---------------------------------------------------------------------------
# pi0 train state init (same as train.py)
# ---------------------------------------------------------------------------


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


# ---------------------------------------------------------------------------
# pi0 policy train step (weighted BC)
# ---------------------------------------------------------------------------


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    advs: at.Array,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)  # [B, H]
        per_sample_loss = jnp.mean(chunked_loss, axis=-1)  # [B]
        weighted = jnp.mean(per_sample_loss * advs)
        return weighted, (chunked_loss, per_sample_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, (chunked_loss, per_sample_loss)), grads = nnx.value_and_grad(
        loss_fn, argnums=diff_state, has_aux=True
    )(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "policy/loss": loss,
        "policy/bc_loss_unweighted": jnp.mean(chunked_loss),
        "policy/bc_loss_per_sample_max": jnp.max(per_sample_loss),
        "policy/bc_loss_per_sample_std": jnp.std(per_sample_loss),
        "policy/grad_norm": optax.global_norm(grads),
        "policy/param_norm": optax.global_norm(kernel_params),
        "policy/adv_weight_mean": jnp.mean(advs),
        "policy/adv_weight_max": jnp.max(advs),
    }
    return new_state, info


# ---------------------------------------------------------------------------
# Batch splitting: pi0 view + IQL view
# ---------------------------------------------------------------------------

_PI0_KEYS = (
    "state",
    "image",
    "image_mask",
    "actions",
    "tokenized_prompt",
    "tokenized_prompt_mask",
    "token_ar_mask",
    "token_loss_mask",
)
# The RL view (next_state, next_image_base, next_image_wrist, reward, mask,
# plus optional progress / progress_is_pad in progress mode) is read directly
# by _build_iql_view / _chunk_reward -- no shared key list needed. PBRS
# reward shaping happens here in the trainer (see _chunk_reward), not in the
# transform: transforms expose raw progress and the scalar endpoint-delta
# ``reward``; trainer builds R = Σ γ^h · (γ·Φ(s_{h+1}) - Φ(s_h)) from it.


def _stack_cameras(arrays: list[np.ndarray]) -> np.ndarray:
    """[B, H, W, 3] xN -> [B, H, W, 3*N, 1] uint8.

    Channels in dim 3 (concatenated per camera), single-frame stack in dim 4
    (the jaxrl2 encoders reshape away the trailing 1 dim).
    """
    cat = np.concatenate(arrays, axis=-1).astype(np.uint8)  # [B, H, W, 3*N]
    return cat[..., None]  # [B, H, W, 3*N, 1]


def _resolve(batch: dict, path: str):
    """Dotted lookup into nested batch dicts, e.g. ``"image.base_0_rgb"``."""
    node = batch
    for part in path.split("."):
        node = node[part]
    return node


def _chunk_reward(
    batch: dict,
    step_reward_discount: float | None,
    progress_reward_weight: float = 1.0,
) -> np.ndarray:
    """Discounted sum of per-step shaped rewards over the H-step action chunk.

    Two paths, matched to the IQL Bellman target ``target_q = R + γ^H·mask·V(s')``:

    * **PBRS path** (``progress`` in batch): potential-based shaping with
      ``Φ(s) = progress(s)``. Per-step ``r_h^PBRS = γ·Φ(s_{h+1}) - Φ(s_h)``;
      the chunk reward ``R = w · Σ_{h=0..H-1} γ^h · r_h^PBRS`` telescopes to
      ``w · (γ^H·progress[H] - progress[0])`` where ``w = progress_reward_weight``.
      PBRS preserves optimal-policy invariance under the γ-discounted Bellman
      target (Ng/Harada/Russell 1999). The RL transform's terminal failure
      penalty (``terminal_reward``: ``-terminal_bonus`` on failed terminal
      transitions, 0 otherwise) is added on top.

    * **Terminal step-cost path** (``actions_is_pad`` in batch, ``progress``
      absent): the RL transform emits a *scalar* reward (-1 non-terminal, 0
      terminal) -- a per-step cost. The Bellman target needs the per-step
      discounted sum, not that scalar repeated. We reconstruct the per-step
      reward sequence from ``actions_is_pad``: ``r_h = -1`` while the action
      offset is in-episode, ``r_h = 0`` once it has crossed the boundary.
      The discounted sum ``Σ γ^h · r_h`` is then correct under the same γ
      used for ``γ^H · V(s_{t+H})`` inside PiIQLLearner.

    Final fallback (no ``progress``, no ``actions_is_pad``): return the
    transform's scalar reward unchanged -- only correct when ``H=1`` or
    when the dataset really does only have a single-step reward.
    """
    if step_reward_discount is not None and "progress" in batch:
        gamma = step_reward_discount
        progress = np.asarray(jax.device_get(batch["progress"]), dtype=np.float32)  # [B, H+1]
        # PBRS per-step reward: γ·Φ(s') - Φ(s), shape [B, H].
        pbrs_step = gamma * progress[..., 1:] - progress[..., :-1]
        gammas = (gamma ** np.arange(pbrs_step.shape[-1])).astype(np.float32)  # [H]
        pbrs = (pbrs_step * gammas).sum(axis=-1).astype(np.float32)  # [B]
        # Terminal failure penalty emitted by the RL transform (-terminal_bonus on
        # failed terminal transitions, 0 otherwise). Zero when absent.
        term = batch.get("terminal_reward")
        term = (
            np.zeros_like(pbrs)
            if term is None
            else np.asarray(jax.device_get(term), dtype=np.float32).reshape(-1)
        )
        return (progress_reward_weight * pbrs + term).astype(np.float32)  # [B]

    if step_reward_discount is not None and "actions_is_pad" in batch:
        gamma = step_reward_discount
        is_pad = np.asarray(jax.device_get(batch["actions_is_pad"])).astype(bool)  # [B, H]
        in_episode = (~is_pad).astype(np.float32)  # [B, H]: 1 inside episode, 0 once past terminal
        per_step_reward = -in_episode  # step-cost: r_h = -1 in-episode, 0 after terminal
        gammas = (gamma ** np.arange(per_step_reward.shape[-1])).astype(np.float32)  # [H]
        return (per_step_reward * gammas).sum(axis=-1).astype(np.float32)  # [B]

    return np.asarray(jax.device_get(batch["reward"]), dtype=np.float32)


def _build_iql_view(
    batch: dict,
    *,
    image_keys: Sequence[str],
    next_image_keys: Sequence[str],
    step_reward_discount: float | None = None,
    progress_reward_weight: float = 1.0,
) -> dict:
    """Pull uint8 pixels + transitions out of the (sharded) batch dict onto the host.

    MUST be called *before* Observation.from_dict, which mutates the image dict
    (uint8 -> float32) in place.

    ``image_keys`` / ``next_image_keys`` are dotted batch paths (e.g.
    ``"image.base_0_rgb"`` or ``"next_image_base"``) -- the matching pair
    feeds the IQL CNN encoder's current/next pixel stack.
    """
    cur_arrays = [np.asarray(jax.device_get(_resolve(batch, k))) for k in image_keys]
    nxt_arrays = [np.asarray(jax.device_get(_resolve(batch, k))) for k in next_image_keys]
    return FrozenDict({
        "observations": FrozenDict({"pixels": _stack_cameras(cur_arrays)}),
        "next_observations": FrozenDict({"pixels": _stack_cameras(nxt_arrays)}),
        "actions": np.asarray(jax.device_get(batch["actions"]), dtype=np.float32),
        "rewards": _chunk_reward(batch, step_reward_discount, progress_reward_weight),
        "masks": np.asarray(jax.device_get(batch["mask"]), dtype=np.float32),
    })


def split_batch(
    batch: dict,
    *,
    image_keys: Sequence[str],
    next_image_keys: Sequence[str],
    step_reward_discount: float | None = None,
    progress_reward_weight: float = 1.0,
) -> tuple[_model.Observation, jax.Array, dict]:
    """Build pi0 Observation/Actions and IQL transition dict from one combined batch."""
    # Build IQL host view FIRST — Observation.from_dict mutates the image dict.
    iql_batch = _build_iql_view(
        batch,
        image_keys=image_keys,
        next_image_keys=next_image_keys,
        step_reward_discount=step_reward_discount,
        progress_reward_weight=progress_reward_weight,
    )

    pi0_dict = {k: batch[k] for k in _PI0_KEYS if k in batch}
    pi0_obs = _model.Observation.from_dict(pi0_dict)
    pi0_actions = pi0_dict["actions"]
    return pi0_obs, pi0_actions, iql_batch


def host_sample_for_iql_init(batch: dict, *, image_keys: Sequence[str]) -> tuple[dict, np.ndarray]:
    """Build a single host-side sample (observations, actions) for PiIQLLearner.__init__.

    We slice on host AFTER device_get rather than on the sharded device array.
    Slicing a data-sharded jax.Array (e.g. ``arr[:1]``) compiles to a gather
    op that needs a NCCL collective working buffer on the GPU, which OOMs
    right after pi0 FSDP weights are loaded and the device is nearly full.
    Pulling the full batch to host once during init is cheap (called only
    here) and only the [:1] slice is retained afterwards.
    """
    cur_arrays = [np.asarray(jax.device_get(_resolve(batch, k)))[:1] for k in image_keys]
    actions = np.asarray(jax.device_get(batch["actions"]), dtype=np.float32)[:1]
    return FrozenDict({"pixels": _stack_cameras(cur_arrays)}), actions


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def init_pi_iql(config: _config.TrainConfig, sample_obs: dict, sample_actions: np.ndarray):
    iql_cfg = config.iql
    if iql_cfg is None:
        raise ValueError(f"Config {config.name} has no .iql section; cannot run train_iql.py")
    from jaxrl2.agents.pi_iql import PiIQLLearner
    return PiIQLLearner(
        seed=config.seed,
        observations=sample_obs,
        actions=sample_actions,
        critic_lr=iql_cfg.critic_lr,
        value_lr=iql_cfg.value_lr,
        hidden_dims=tuple(iql_cfg.hidden_dims),
        cnn_features=tuple(iql_cfg.cnn_features),
        cnn_strides=tuple(iql_cfg.cnn_strides),
        cnn_padding=iql_cfg.cnn_padding,
        latent_dim=iql_cfg.latent_dim,
        discount=iql_cfg.discount,
        tau=iql_cfg.tau,
        expectile=iql_cfg.expectile,
        A_scaling=iql_cfg.A_scaling,
        critic_reduction=iql_cfg.critic_reduction,
        encoder_type=iql_cfg.encoder_type,
        encoder_norm=iql_cfg.encoder_norm,
        color_jitter=iql_cfg.color_jitter,
        use_spatial_softmax=iql_cfg.use_spatial_softmax,
        softmax_temperature=iql_cfg.softmax_temperature,
        aug_next=iql_cfg.aug_next,
        use_bottleneck=iql_cfg.use_bottleneck,
        num_qs=iql_cfg.num_qs,
        num_cameras=iql_cfg.num_cameras,
        action_horizon=config.model.action_horizon,
    )


def _safe_histogram(arr: np.ndarray, num_bins: int = 64):
    """Build a wandb.Histogram, guarding against NaN/Inf and empty arrays."""
    a = np.asarray(arr).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return wandb.Histogram(a.astype(np.float32), num_bins=num_bins)


def _build_distribution_logs(d: dict) -> dict:
    """Per-step distribution snapshots for wandb (histograms)."""
    if not d:
        return {}
    out: dict[str, Any] = {}

    qs = d.get("qs")  # [num_qs, B]
    if qs is not None:
        for i in range(qs.shape[0]):
            h = _safe_histogram(qs[i])
            if h is not None:
                out[f"dist/q_head_{i}"] = h
        # Pooled across heads.
        h = _safe_histogram(qs)
        if h is not None:
            out["dist/q_all"] = h
        # Per-sample min over heads -- this is what compute_advs uses with critic_reduction="min".
        h = _safe_histogram(qs.min(axis=0))
        if h is not None:
            out["dist/q_min_over_heads"] = h

    v = d.get("v")
    if v is not None:
        h = _safe_histogram(v)
        if h is not None:
            out["dist/v"] = h

    adv = d.get("adv")
    if adv is not None:
        h = _safe_histogram(adv)
        if h is not None:
            out["dist/adv_raw"] = h

    advs = d.get("advs")
    if advs is not None:
        h = _safe_histogram(advs)
        if h is not None:
            out["dist/adv_weight"] = h
        # log-scale view since weights span exp(A * adv).
        h = _safe_histogram(np.log(advs + 1e-8))
        if h is not None:
            out["dist/adv_weight_log"] = h

    rewards = d.get("rewards")
    if rewards is not None:
        h = _safe_histogram(rewards)
        if h is not None:
            out["dist/reward"] = h

    actions = d.get("actions")  # [B, H, action_dim] (padded -- only first 7 dims are real for Libero)
    if actions is not None:
        real_dim = 7
        first_step = actions[:, 0, :real_dim]  # [B, 7]
        for d_i in range(real_dim):
            h = _safe_histogram(first_step[:, d_i])
            if h is not None:
                out[f"dist/action_dim{d_i}_step0"] = h
        # Full chunk pooled per dim.
        for d_i in range(real_dim):
            h = _safe_histogram(actions[:, :, d_i])
            if h is not None:
                out[f"dist/action_dim{d_i}_chunk"] = h
        # Pooled across all real dims & all chunk steps.
        h = _safe_histogram(actions[:, :, :real_dim])
        if h is not None:
            out["dist/action_all_real_dims"] = h
    return out


def maybe_log_camera_views(batch: dict):
    try:
        images = batch["image"]
        # Pull each camera once into host numpy then index — slicing the
        # sharded device array per-sample (img[i]) would issue a NCCL gather
        # per call and OOM when the GPU is near capacity.
        host_imgs = {k: np.asarray(jax.device_get(v)) for k, v in images.items()}
        first = next(iter(host_imgs.values()))
        n = min(5, int(first.shape[0]))
        wandb_imgs = [
            wandb.Image(np.concatenate([img[i] for img in host_imgs.values()], axis=1))
            for i in range(n)
        ]
        wandb.log({"camera_views": wandb_imgs}, step=0)
    except Exception as e:
        logging.warning(f"Failed to log camera views: {e}")


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.iql is None:
        raise ValueError(
            f"Config '{config.name}' has no IQL section. Use a config like pi05_libero_iql or attach an iql= IQLConfig."
        )
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _rl_data_loader.create_rl_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized RL data loader (combined batch keys: {sorted(batch.keys())})")
    maybe_log_camera_views(batch)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized pi0 train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # Resolve and validate the camera key contract up-front so a config /
    # transform mismatch fails before we touch GPU memory.
    image_keys = tuple(config.iql.image_keys)
    next_image_keys = tuple(config.iql.next_image_keys)
    if len(image_keys) != config.iql.num_cameras or len(next_image_keys) != config.iql.num_cameras:
        raise ValueError(
            f"IQLConfig: len(image_keys)={len(image_keys)} and "
            f"len(next_image_keys)={len(next_image_keys)} must both equal "
            f"num_cameras={config.iql.num_cameras}. Update the config or the RL "
            "transform so the camera counts line up."
        )

    # Build the IQL agent from scratch first so we have a structure template
    # for orbax. When resuming, restore_state below overwrites the random init
    # with the persisted critic / target_critic / value tensors in a single
    # atomic CheckpointManager.restore call alongside pi0 train_state.
    sample_obs, sample_actions = host_sample_for_iql_init(batch, image_keys=image_keys)
    pixel_iql = init_pi_iql(config, sample_obs, sample_actions)

    logging.info(
        f"Initialized PiIQLLearner: encoder={config.iql.encoder_type}, num_cameras={config.iql.num_cameras}, "
        f"pixels_shape={sample_obs['pixels'].shape}, actions_shape={sample_actions.shape}"
    )

    if resuming:
        train_state, restored_iql = _checkpoints.restore_state(
            checkpoint_manager,
            train_state,
            data_loader,
            iql_state=pixel_iql.iql_save_pytree(),
        )
        pixel_iql.iql_load_pytree(restored_iql)
        logging.info("Restored pi0 train_state + IQL critic/target/value from checkpoint")

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos: list[dict] = []
    # Latest per-sample tensors (refreshed every step, sent to wandb as histograms at log_interval).
    latest_dist: dict[str, np.ndarray] = {}
    warmup = int(config.iql.critic_warmup_steps)

    # LR schedule callable (constant if not scheduled)
    try:
        lr_schedule_fn = config.lr_schedule.create()
    except Exception:
        lr_schedule_fn = None

    last_log_time = time.monotonic()
    last_log_step = start_step

    # Per-step gamma used inside _chunk_reward to build the PBRS shaped
    # reward R = Σ γ^h · (γ·Φ(s_{h+1}) - Φ(s_h)). Matches the γ^H next-state
    # discount inside PiIQLLearner so the Bellman target stays consistent.
    step_reward_discount = float(config.iql.discount)
    # Scales the PBRS shaped reward inside _chunk_reward. Lives on the RL
    # DataConfig — same value the RL transform folds into ``reward`` — so the
    # trainer's shaped reward matches the configured weight instead of 1.0.
    progress_reward_weight = float(getattr(config.data, "progress_reward_weight", 1.0))

    for step in pbar:
        step_t0 = time.monotonic()
        pi0_obs, pi0_actions, iql_batch = split_batch(
            batch,
            image_keys=image_keys,
            next_image_keys=next_image_keys,
            step_reward_discount=step_reward_discount,
            progress_reward_weight=progress_reward_weight,
        )
        split_t = time.monotonic() - step_t0

        # ---- 1) update critic / value ----
        critic_t0 = time.monotonic()
        critic_info = pixel_iql.update(iql_batch)
        critic_info = {f"iql/{k}": jnp.asarray(v) for k, v in critic_info.items()}
        critic_t = time.monotonic() - critic_t0

        # ---- 2) compute IQL advantages (always, for Q/V logging; advs unused for policy during warmup) ----
        adv_t0 = time.monotonic()
        advs_dev, advs_computed, raw_adv_info, qs_host, v_host, adv_host = pixel_iql.compute_advs(
            iql_batch, output_sharding=data_sharding
        )
        if step < warmup:
            # During warmup we ignore the learned advantages and use uniform weights.
            # Replace both the device-side input to ptrain_step and the host-side
            # array used for logging.
            advs_computed = np.ones(iql_batch["actions"].shape[0], dtype=np.float32)
            advs_sharded = jax.make_array_from_process_local_data(data_sharding, advs_computed)
            adv_info = {f"iql/{k}": jnp.asarray(v) for k, v in raw_adv_info.items()}
            adv_info["iql/warmup_active"] = jnp.asarray(1.0)
        else:
            advs_sharded = advs_dev  # already lives on device with data_sharding
            adv_info = {f"iql/{k}": jnp.asarray(v) for k, v in raw_adv_info.items()}
            adv_info["iql/warmup_active"] = jnp.asarray(0.0)
        # Host-side adv distribution (quantiles + clip frac)
        adv_info.update(
            {
                "iql/adv_weight_host_p50": float(np.median(advs_computed)),
                "iql/adv_weight_host_p95": float(np.quantile(advs_computed, 0.95)),
                "iql/adv_weight_host_p99": float(np.quantile(advs_computed, 0.99)),
                "iql/adv_weight_host_max": float(advs_computed.max()),
                "iql/adv_weight_host_min": float(advs_computed.min()),
                "iql/adv_weight_host_clip_frac": float((advs_computed >= 100.0).mean()),
            }
        )
        adv_t = time.monotonic() - adv_t0

        # ---- 3) update pi0 policy with weighted BC ----
        policy_t0 = time.monotonic()
        with sharding.set_mesh(mesh):
            train_state, policy_info = ptrain_step(train_rng, train_state, (pi0_obs, pi0_actions), advs_sharded)
        policy_t = time.monotonic() - policy_t0

        # ---- 4) host-side batch + timing stats ----
        rewards_host = np.asarray(iql_batch["rewards"])
        masks_host = np.asarray(iql_batch["masks"])
        actions_host = np.asarray(iql_batch["actions"])
        # Stash per-sample arrays (already host numpy) for histogram logging.
        latest_dist = {
            "qs": qs_host,            # [num_qs, B]
            "v": v_host,              # [B]
            "adv": adv_host,          # [B]
            "advs": advs_computed,    # [B] -- post-clip advantage weights
            "actions": actions_host,  # [B, H, action_dim]
            "rewards": rewards_host,  # [B]
        }
        data_info = {
            "data/reward_mean": float(rewards_host.mean()),
            "data/reward_frac_nonzero": float((rewards_host > 0).mean()),
            "data/mask_mean": float(masks_host.mean()),
            "data/terminal_frac": float((masks_host == 0).mean()),
            "data/action_abs_mean": float(np.abs(actions_host).mean()),
            "data/action_std": float(actions_host.std()),
        }
        timing_info = {
            "timing/split_batch_s": split_t,
            "timing/critic_update_s": critic_t,
            "timing/adv_compute_s": adv_t,
            "timing/policy_update_s": policy_t,
            "timing/total_step_s": time.monotonic() - step_t0,
        }
        if lr_schedule_fn is not None:
            try:
                timing_info["policy/lr"] = float(lr_schedule_fn(int(train_state.step)))
            except Exception:
                pass

        merged = {**policy_info, **critic_info, **adv_info, **data_info, **timing_info}
        merged = {k: (jnp.asarray(v) if not isinstance(v, jnp.ndarray) else v) for k, v in merged.items()}
        infos.append(merged)

        if step % config.log_interval == 0:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            now = time.monotonic()
            elapsed = now - last_log_time
            steps_done = step - last_log_step + 1
            reduced["timing/steps_per_sec"] = steps_done / max(elapsed, 1e-6)
            last_log_time = now
            last_log_step = step + 1
            # Group keys for readable printing.
            groups: dict[str, list[str]] = {}
            for k in sorted(reduced):
                grp = k.split("/", 1)[0]
                groups.setdefault(grp, []).append(k)
            lines = [f"Step {step}:"]
            for grp, keys in groups.items():
                seg = ", ".join(f"{k.split('/', 1)[1] if '/' in k else k}={float(reduced[k]):.4f}" for k in keys)
                lines.append(f"  [{grp}] {seg}")
            pbar.write("\n".join(lines))

            # Histograms from the most recent step.
            hist_log = _build_distribution_logs(latest_dist)
            wandb.log({**reduced, **hist_log}, step=step)
            infos = []

        # Drop refs to this step's batch/derived tensors so the next batch's
        # device buffers can be reused instead of stacking on top of the
        # previous step's allocations.
        del pi0_obs, pi0_actions, iql_batch, batch
        del advs_sharded, advs_dev, advs_computed
        del qs_host, v_host, adv_host, raw_adv_info

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                step,
                iql_state=pixel_iql.iql_save_pytree(),
            )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
