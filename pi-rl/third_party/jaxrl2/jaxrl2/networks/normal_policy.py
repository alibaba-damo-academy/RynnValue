# adapted from jaxrl2
from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init, xavier_init


class NormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    std: Optional[float] = 1.
    init_scale: Optional[float] = 1.
    output_scale: Optional[float] = 1.
    init_method: str = 'xavier'

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate,
                      init_scale = self.init_scale
                      )(observations, training=training)

        if self.init_method == 'xavier':
            # print('fc layer {}x{}'.format(outputs.shape, self.action_dim))
            means = nn.Dense(self.action_dim, kernel_init=xavier_init())(outputs)
        else:
            means = nn.Dense(self.action_dim, kernel_init=default_init(self.init_scale))(outputs)

        means *= self.output_scale

        return distrax.MultivariateNormalDiag(loc=means,
                                              scale_diag=jnp.ones_like(means)*self.std)
