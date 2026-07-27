# adapted from jaxrl2
from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

import jax

from jaxrl2.networks.mlp import MLP
from jaxrl2.networks.mlp import MLPActionSep
from jaxrl2.networks.constants import default_init

from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any 
Array = Any
PrecisionLike = Union[None, str, jax.lax.Precision, Tuple[str, str],
                      Tuple[jax.lax.Precision, jax.lax.Precision]]

default_kernel_init = nn.initializers.lecun_normal()

class StateActionValue(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_action_sep: bool = False

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 training: bool = False):
        inputs = {'states': observations, 'actions': actions}
        if self.use_action_sep:
            critic = MLPActionSep(
                (*self.hidden_dims, 1),
                activations=self.activations,
                use_layer_norm=True)(inputs, training=training)
        else:
            critic = MLP((*self.hidden_dims, 1),
                        activations=self.activations,
                        use_layer_norm=True)(inputs, training=training)
        return jnp.squeeze(critic, -1)
