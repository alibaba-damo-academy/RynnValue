# adapted from jaxrl2
from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
from jaxrl2.data.dataset import DatasetDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params, PRNGKey


def update_actor(
    key: PRNGKey,
    actor: TrainState,
    target_critic: TrainState,
    value: TrainState,
    batch: DatasetDict,
    A_scaling: float,
    critic_reduction : str = 'min',
    cross_norm : bool = False
) -> Tuple[TrainState, Dict[str, float]]:
    
    key, key_act = jax.random.split(key, num=2)
    
    if hasattr(value, 'batch_stats') and value.batch_stats is not None:
        v, _ = value.apply_fn({'params': value.params, 'batch_stats': value.batch_stats}, batch['observations'], mutable=['batch_stats'])
    else:    
        v = value.apply_fn({'params': value.params}, batch['observations'])  

    if hasattr(target_critic, 'batch_stats') and target_critic.batch_stats is not None:
        qs, _ = target_critic.apply_fn({'params': target_critic.params, 'batch_stats': target_critic.batch_stats}, batch['observations'],
                        batch['actions'], mutable=['batch_stats'])
    else:    
        qs = target_critic.apply_fn({'params': target_critic.params}, batch['observations'], batch['actions'])

    if critic_reduction == 'min':
        q = qs.min(axis=0)
    elif critic_reduction == 'mean':
        q = qs.mean(axis=0)
    else:
        raise ValueError(f"Invalid critic reduction: {critic_reduction}")

    exp_a = jnp.exp((q - v) * A_scaling)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[Dict[str, float], Any]]:
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['observations'], mutable=['batch_stats'])
            if cross_norm:
                next_dist = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['next_observations'], mutable=['batch_stats'])
            else:
                next_dist = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['next_observations'])
            if type(next_dist) == tuple:
                next_dist, new_model_state = next_dist
        else:
            dist = actor.apply_fn({'params': actor_params}, batch['observations'])
            next_dist = actor.apply_fn({'params': actor_params}, batch['next_observations'])
            new_model_state = {}
        
        # For logging only
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        mean_dist_norm = jnp.linalg.norm(mean_dist, axis=-1)
        std_dist_norm = jnp.linalg.norm(std_diag_dist, axis=-1)

        
        _, log_probs = dist.sample_and_log_prob(seed=key_act)
        actor_loss = -(exp_a * log_probs).mean()

        things_to_log = {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'q_pi_in_actor': q.mean(),
            'mean_pi_norm': mean_dist_norm.mean(),
            'std_pi_norm': std_dist_norm.mean(),
            'mean_pi_avg': mean_dist.mean(),
            'mean_pi_max': mean_dist.max(),
            'mean_pi_min': mean_dist.min(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_max': std_diag_dist.max(),
            'std_pi_min': std_diag_dist.min(),
            'adv': q - v,
        }

        return actor_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info