# adapted from jaxrl2
from typing import Dict, Tuple, Optional, Any

import jax
import jax.numpy as jnp
import optax
from jaxrl2.data.dataset import DatasetDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


def _apply_value(state: TrainState, observations, params=None, train: bool = False):
    params = state.params if params is None else params

    if hasattr(state, "batch_stats") and state.batch_stats is not None:
        variables = {"params": params, "batch_stats": state.batch_stats}
        if train:
            out, new_model_state = state.apply_fn(
                variables, observations, mutable=["batch_stats"]
            )
            return out, new_model_state
        else:
            out = state.apply_fn(variables, observations, mutable=False)
            return out, None
    else:
        out = state.apply_fn({"params": params}, observations)
        return out, None


def _apply_critic(state: TrainState, observations, actions, params=None, train: bool = False):
    params = state.params if params is None else params

    if hasattr(state, "batch_stats") and state.batch_stats is not None:
        variables = {"params": params, "batch_stats": state.batch_stats}
        if train:
            out, new_model_state = state.apply_fn(
                variables, observations, actions, mutable=["batch_stats"]
            )
            return out, new_model_state
        else:
            out = state.apply_fn(variables, observations, actions, mutable=False)
            return out, None
    else:
        out = state.apply_fn({"params": params}, observations, actions)
        return out, None


def update_v(
    target_critic: TrainState,
    value: TrainState,
    batch: DatasetDict,
    expectile: float,
    critic_reduction: str,
    axis_name: Optional[str] = None,
) -> Tuple[TrainState, Dict[str, float]]:
    qs, _ = _apply_critic(
        target_critic,
        batch["observations"],
        batch["actions"],
        train=False,
    )

    if critic_reduction == "min":
        q = qs.min(axis=0)
    elif critic_reduction == "mean":
        q = qs.mean(axis=0)
    else:
        raise NotImplementedError(f"Unknown critic_reduction: {critic_reduction}")

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, Tuple[Dict[str, float], Any]]:
        v, new_model_state = _apply_value(
            value,
            batch["observations"],
            params=value_params,
            train=True,
        )
        diff = q - v
        weight = jnp.where(diff > 0, expectile, 1.0 - expectile)
        value_loss = (weight * diff ** 2).mean()
        info = {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_std": v.std(),
            "v_min": v.min(),
            "v_max": v.max(),
            "q_in_v": q.mean(),
            "td_v_mean": diff.mean(),  # E[q - v] -- should be > 0 with expectile > 0.5
            "td_v_abs_mean": jnp.abs(diff).mean(),
            "expectile_weight_mean": weight.mean(),
            "expectile_pos_frac": (diff > 0).astype(jnp.float32).mean(),
        }
        return value_loss, (info, new_model_state)

    (value_loss, (info, new_model_state)), grads = jax.value_and_grad(
        value_loss_fn, has_aux=True
    )(value.params)

    if axis_name is not None:
        grads = jax.lax.pmean(grads, axis_name=axis_name)
        info = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name=axis_name), info)

    info["value_grad_norm"] = optax.global_norm(grads)

    if new_model_state is not None and "batch_stats" in new_model_state:
        new_batch_stats = new_model_state["batch_stats"]
        if axis_name is not None:
            new_batch_stats = jax.lax.pmean(new_batch_stats, axis_name=axis_name)
        new_value = value.apply_gradients(grads=grads, batch_stats=new_batch_stats)
    else:
        new_value = value.apply_gradients(grads=grads)

    return new_value, info


def update_q(
    critic: TrainState,
    value: TrainState,
    batch: DatasetDict,
    discount: float,
    axis_name: Optional[str] = None,
) -> Tuple[TrainState, Dict[str, float]]:
    next_v, _ = _apply_value(
        value,
        batch["next_observations"],
        train=False,
    )

    # Defensive: target_q currently only depends on value / target_critic
    # params (not the critic_params we differentiate against), so a missing
    # stop_gradient is a no-op today. Guard against a future loss change that
    # mixes V into the critic gradient path and would silently bootstrap.
    target_q = jax.lax.stop_gradient(batch["rewards"] + discount * batch["masks"] * next_v)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Tuple[Dict[str, float], Any]]:
        qs, new_model_state = _apply_critic(
            critic,
            batch["observations"],
            batch["actions"],
            params=critic_params,
            train=True,
        )
        td = qs - target_q  # [num_qs, B]
        critic_loss = (td ** 2).mean()
        # Per-ensemble-head stats.
        q_per_head_mean = qs.mean(axis=-1)  # [num_qs]
        info = {
            "critic_loss": critic_loss,
            "q_mean": qs.mean(),
            "q_std": qs.std(),
            "q_min": qs.min(),
            "q_max": qs.max(),
            "q_ensemble_disagreement": qs.std(axis=0).mean(),  # std across ensemble per sample
            "q_head_spread": q_per_head_mean.max() - q_per_head_mean.min(),
            "target_q_mean": target_q.mean(),
            "target_q_std": target_q.std(),
            "next_v_mean": next_v.mean(),
            "next_v_std": next_v.std(),
            "td_q_mean": td.mean(),
            "td_q_abs_mean": jnp.abs(td).mean(),
            "td_q_max_abs": jnp.abs(td).max(),
            "reward_mean": batch["rewards"].mean(),
            "mask_mean": batch["masks"].mean(),
        }
        return critic_loss, (info, new_model_state)

    (critic_loss, (info, new_model_state)), grads = jax.value_and_grad(
        critic_loss_fn, has_aux=True
    )(critic.params)

    if axis_name is not None:
        grads = jax.lax.pmean(grads, axis_name=axis_name)
        info = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name=axis_name), info)

    info["critic_grad_norm"] = optax.global_norm(grads)

    if new_model_state is not None and "batch_stats" in new_model_state:
        new_batch_stats = new_model_state["batch_stats"]
        if axis_name is not None:
            new_batch_stats = jax.lax.pmean(new_batch_stats, axis_name=axis_name)
        new_critic = critic.apply_gradients(grads=grads, batch_stats=new_batch_stats)
    else:
        new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info
