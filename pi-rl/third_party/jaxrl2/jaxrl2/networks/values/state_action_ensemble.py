# adapted from jaxrl2
from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks.values.state_action_value import StateActionValue


class StateActionEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    use_action_sep: bool = False

    @nn.compact
    def __call__(self, states, actions, training: bool = False):

        # print ('Use action sep in state action ensemble: ', self.use_action_sep)
        VmapCritic = nn.vmap(StateActionValue,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        activations=self.activations,
                        use_action_sep=self.use_action_sep)(states, actions,
                                                      training)
        return qs
