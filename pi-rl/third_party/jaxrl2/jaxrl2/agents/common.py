# adapted from jaxrl2
from __future__ import annotations

from functools import partial
from typing import Callable, Tuple, Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import distrax

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey
import flax.linen as nn
from typing import Any, Callable, Dict, Sequence, Union

# Helps to minimize CPU to GPU transfer.
def _unpack(batch):
    # Assuming that if next_observation is missing, it's combined with observation:
    obs_pixels = batch['observations']['pixels'][..., :-1]
    next_obs_pixels = batch['observations']['pixels'][..., 1:]

    obs = batch['observations'].copy(add_or_replace={'pixels': obs_pixels})
    next_obs = batch['next_observations'].copy(
        add_or_replace={'pixels': next_obs_pixels})

    batch = batch.copy(add_or_replace={
        'observations': obs,
        'next_observations': next_obs
    })

    return batch

@partial(jax.jit, static_argnames='actor_apply_fn')
def eval_log_prob_jit(actor_apply_fn: Callable[..., distrax.Distribution],
                      actor_params: Params, actor_batch_stats: Any, batch: DatasetDict) -> float:
    # batch = _unpack(batch)
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, 
                          batch['observations'],
                          training=False,
                          mutable=False)
    log_probs = dist.log_prob(batch['actions'])
    return log_probs.mean()

@partial(jax.jit, static_argnames='actor_apply_fn')
def eval_mse_jit(actor_apply_fn: Callable[..., distrax.Distribution],
                      actor_params: Params, actor_batch_stats: Any, batch: DatasetDict) -> float:
    # batch = _unpack(batch)
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, 
                          batch['observations'],
                          training=False,
                          mutable=False)
    mse = (dist.loc - batch['actions']) ** 2
    return mse.mean()

def eval_reward_function_jit(actor_apply_fn: Callable[..., distrax.Distribution],
                      actor_params: Params, actor_batch_stats: Any, batch: DatasetDict) -> float:
    # batch = _unpack(batch)
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, 
                          batch['observations'],
                          training=False,
                          mutable=False)
    pred = dist.mode().reshape(-1)
    loss = - (batch['rewards'] * jnp.log(1. / (1. + jnp.exp(-pred))) + (1.0 - batch['rewards']) * jnp.log(1. - 1. / (1. + jnp.exp(-pred))))
    return loss.mean()


@partial(jax.jit, static_argnames='actor_apply_fn')
def eval_actions_jit(actor_apply_fn: Callable[..., distrax.Distribution],
                     actor_params: Params,
                     observations: np.ndarray,
                     actor_batch_stats: Any) -> jnp.ndarray:
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, observations, training=False,
                          mutable=False)
    return dist.mode()


@partial(jax.jit, static_argnames='actor_apply_fn')
def sample_actions_jit(
        rng: PRNGKey, actor_apply_fn: Callable[..., distrax.Distribution],
        actor_params: Params,
        observations: np.ndarray,
        actor_batch_stats: Any) -> Tuple[PRNGKey, jnp.ndarray]:
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


@partial(jax.jit, static_argnames="actor_apply_fn")
def sample_actions_seeded_jit(
    seed,
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
    actor_batch_stats: Any,
) -> jnp.ndarray:
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, observations)
    return dist.sample(seed=seed)


@partial(jax.jit, static_argnames='actor_apply_fn')
def sample_actions_with_log_probs_jit(
        rng: PRNGKey, actor_apply_fn: Callable[..., distrax.Distribution],
        actor_params: Params,
        observations: np.ndarray,
        actor_batch_stats: Any) -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray]:
    """Sample actions and compute their log probabilities."""
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, observations)
    rng, key = jax.random.split(rng)
    # Use sample_and_log_prob to correctly handle transformed distributions (e.g., TanhTransformed)
    # This avoids numerical issues with computing log_prob on already-transformed samples
    actions, log_probs = dist.sample_and_log_prob(seed=key)
    return rng, actions, log_probs


class ModuleDict(nn.Module):
    """
    from https://github.com/rail-berkeley/jaxrl_minimal/blob/main/jaxrl_m/common/common.py#L33
    Utility class for wrapping a dictionary of modules. This is useful when you have multiple modules that you want to
    initialize all at once (creating a single `params` dictionary), but you want to be able to call them separately
    later. As a bonus, the modules may have sub-modules nested inside them that share parameters (e.g. an image encoder)
    and Flax will automatically handle this without duplicating the parameters.

    To initialize the modules, call `init` with no `name` kwarg, and then pass the example arguments to each module as
    additional kwargs. To call the modules, pass the name of the module as the `name` kwarg, and then pass the arguments
    to the module as additional args or kwargs.

    Example usage:
    ```
    shared_encoder = Encoder()
    actor = Actor(encoder=shared_encoder)
    critic = Critic(encoder=shared_encoder)

    model_def = ModuleDict({"actor": actor, "critic": critic})
    params = model_def.init(rng_key, actor=example_obs, critic=(example_obs, example_action))

    actor_output = model_def.apply({"params": params}, example_obs, name="actor")
    critic_output = model_def.apply({"params": params}, example_obs, action=example_action, name="critic")
    ```
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f"When `name` is not specified, kwargs must contain the arguments for each module. "
                    f"Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}"
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)