# adapted from jaxrl2
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jaxrl2.data.dataset import DatasetDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def update_v(
    target_critic: TrainState,
    value: TrainState,
    batch: DatasetDict,
    expectile: float,
    critic_reduction: str,
) -> Tuple[TrainState, Dict[str, float]]:
    qs = target_critic.apply_fn(
        {"params": target_critic.params}, batch["observations"], batch["actions"]
    )

    if critic_reduction == "min":
        q = qs.min(axis=0)
    elif critic_reduction == "mean":
        q = qs.mean(axis=0)
    else:
        raise NotImplemented()

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        v = value.apply_fn({"params": value_params}, batch["observations"])
        value_loss = loss(q - v, expectile).mean()
        return value_loss, {"value_loss": value_loss, "v": v.mean()}

    grads, info = jax.grad(value_loss_fn, has_aux=True)(value.params)
    new_value = value.apply_gradients(grads=grads)

    return new_value, info


def update_q(
    critic: TrainState, value: TrainState, batch: DatasetDict, discount: float
) -> Tuple[TrainState, Dict[str, float]]:
    next_v = value.apply_fn({"params": value.params}, batch["next_observations"])

    target_q = batch["rewards"] + discount * batch["masks"] * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn(
            {"params": critic_params}, batch["observations"], batch["actions"]
        )
        critic_loss = ((qs - target_q) ** 2).mean()
        return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info
