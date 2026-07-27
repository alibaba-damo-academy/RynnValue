# adapted from jaxrl2
from typing import Dict, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init, xavier_init, kaiming_init

from functools import partial
from typing import Any, Callable, Sequence, Tuple

ModuleDef = Any

class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(features,
                        kernel_size=(3, 3),
                        strides=(stride, stride),
                        kernel_init=default_init(),
                        padding=self.padding)(x)
            x = nn.relu(x)

        return x.reshape((*x.shape[:-3], -1))
    

class PixelMultiplexer(nn.Module):
    encoder: Union[nn.Module, list]
    network: nn.Module
    latent_dim: int
    use_bottleneck: bool=True

    @nn.compact
    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 actions: Optional[jnp.ndarray] = None,
                 training: bool = False):
        observations = FrozenDict(observations)

        x = self.encoder(observations['pixels'], training)
        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        x = observations.copy(add_or_replace={'pixels': x})

        # print('fully connected keys', x.keys())
        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)
