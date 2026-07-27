# adapted from jaxrl2
from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.mlp import MLP


class StateValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(observations,
                                                   training=training)
        return jnp.squeeze(critic, -1)


class StateValueEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_vs: int = 2

    @nn.compact
    def __call__(self, observations, training: bool = False):
        VmapCritic = nn.vmap(StateValue,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_vs)
        qs = VmapCritic(self.hidden_dims,
                        activations=self.activations)(observations,
                                                      training)
        return qs
