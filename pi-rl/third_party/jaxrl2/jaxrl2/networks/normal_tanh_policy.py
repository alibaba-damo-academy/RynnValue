# adapted from jaxrl2
from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init, xavier_init


class TanhMultivariateNormalDiag(distrax.Transformed):

    def __init__(self,
                 loc: jnp.ndarray,
                 scale_diag: jnp.ndarray,
                 low: Optional[jnp.ndarray] = None,
                 high: Optional[jnp.ndarray] = None):
        distribution = distrax.MultivariateNormalDiag(loc=loc,
                                                      scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1))

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    low: Optional[jnp.ndarray] = None
    high: Optional[jnp.ndarray] = None
    mlp_init_scale: float = 1.0
    init_method: str = 'default'

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate,
                      init_scale=self.mlp_init_scale)(observations,
                                                      training=training)

        if self.init_method == 'xavier':
            means = nn.Dense(self.action_dim, kernel_init=xavier_init())(outputs)
            log_stds = nn.Dense(self.action_dim, kernel_init=xavier_init())(outputs)
        else:
            means = nn.Dense(self.action_dim, kernel_init=default_init(self.mlp_init_scale))(outputs)
            log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        return TanhMultivariateNormalDiag(loc=means,
                                          scale_diag=jnp.exp(log_stds) * self.mlp_init_scale,
                                          low=self.low,
                                          high=self.high)
