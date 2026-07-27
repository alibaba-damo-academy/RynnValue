# adapted from jaxrl2
"""Noise critic updater for DSRL-NA."""
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def update_action_critic(
        key: PRNGKey, critic: TrainState,
        target_critic: TrainState, temp: TrainState, batch: DatasetDict,
        discount: float, backup_entropy: bool = False,
        critic_reduction: str = 'min') -> Tuple[TrainState, Dict[str, float]]:
    """TD update for action-space critic using executed actions."""
    next_actions = batch['next_executed_actions']
    next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     batch['next_observations'], next_actions)
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q

    if backup_entropy:
        ent_coef = temp.apply_fn({'params': temp.params})
        entropy_term = ent_coef * batch['next_log_probs']
        masked_entropy_term = batch["discount"] * batch['masks'] * entropy_term
        target_q -= masked_entropy_term

    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn({'params': critic_params}, batch['observations'],
                             batch['executed_actions'])
        critic_loss = ((qs - target_q)**2).mean()
        return critic_loss, {
            'action_critic_loss': critic_loss,
            'action_critic_q': qs.mean(),
            'action_critic_q_min': qs.min(),
            'action_critic_q_max': qs.max(),
            'action_critic_target_q': target_q.mean(),
            'action_critic_next_q': next_q.mean(),
            'action_critic_rewards': batch['rewards'].mean(),
            'action_critic_discount': batch['discount'].mean(),
            'action_critic_masks': batch['masks'].mean(),
            'action_critic_ent_coef': ent_coef.mean() if backup_entropy else 0.0,
            'action_critic_next_log_probs': batch['next_log_probs'].mean() if backup_entropy else 0.0,
            'action_critic_entropy_term': (entropy_term.mean() if backup_entropy else 0.0),
            'action_critic_masked_entropy_term': (masked_entropy_term.mean() if backup_entropy else 0.0),
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    # Log gradient norm before optimizer applies any transforms
    # grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads)))
    # info['action_critic_grad_norm'] = grad_norm
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info


def update_noise_critic(
        key: PRNGKey, critic: TrainState,
        action_critic: TrainState, batch: DatasetDict, 
        critic_reduction: str = 'min') -> Tuple[TrainState, Dict[str, float]]:
    """Update for noise critic: distillation from action critic.
    
    Args:
        key: Random key
        critic: Noise critic train state
        action_critic: Action critic train state (for distillation)
        batch: Batch of data
        critic_reduction: How to reduce Q values ('min' or 'mean')
    
    Returns:
        Updated critic and info dict
    """
    distill_qs = action_critic.apply_fn({'params': action_critic.params},
                                        batch['observations'], batch['distill_actions'])
    if critic_reduction == 'min':
        distill_target = distill_qs.min(axis=0)
    else:
        distill_target = distill_qs.mean(axis=0)

    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        info = {}
        distill_qs = critic.apply_fn({'params': critic_params}, batch['observations'], 
                                        batch['distill_noise'])
        distill_loss = ((distill_qs - distill_target)**2).mean()
        total_loss = distill_loss
        info['noise_critic_distill_loss'] = distill_loss
        info['noise_critic_distill_target'] = distill_target.mean()
        info['noise_critic_distill_target_min'] = distill_target.min()
        info['noise_critic_distill_target_max'] = distill_target.max()
        info['noise_critic_qs'] = distill_qs.mean()
        info['noise_critic_qs_min'] = distill_qs.min()
        info['noise_critic_qs_max'] = distill_qs.max()
        
        info['noise_critic_total_loss'] = total_loss
        return total_loss, info

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    # Log gradient norm before optimizer applies any transforms
    # grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads)))
    # info['noise_critic_grad_norm'] = grad_norm
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info
