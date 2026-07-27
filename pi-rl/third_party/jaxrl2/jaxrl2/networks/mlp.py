# adapted from jaxrl2
from typing import Callable, Optional, Sequence, Union
from flax.core import frozen_dict

import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init


def _flatten_dict(x: Union[FrozenDict, jnp.ndarray]):
    if hasattr(x, 'values'):
        obs = []
        for k, v in sorted(x.items()):
            # if k == "actions":
            #     v = v[:, 0:1, ...]
            if k == 'state': # flatten action chunk to 1D
                obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
                # v = jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])])
            elif k == 'prev_action' or k == 'actions':
                if v.ndim > 2:
                    # deal with action chunk
                    obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
                else:
                    obs.append(v)
            else:
                obs.append(_flatten_dict(v))
        return jnp.concatenate(obs, -1)
    else:
        return x

def _flatten_dict_special(x):
    if hasattr(x, 'values'):
        obs = []
        action = None
        for k, v in sorted(x.items()):
            if k == 'state' or k == 'prev_action':
                obs.append(jnp.reshape(v, [*v.shape[:-2], np.prod(v.shape[-2:])]))
            elif k == 'actions':
                print ('action shape: ', v.shape)
                action = v
            else:
                obs.append(_flatten_dict(v))
        return jnp.concatenate(obs, -1), action
    else:
        return x
        

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    init_scale: Optional[float] = 1.
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = _flatten_dict(x)
        # print('mlp post flatten', x.shape)

        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init(self.init_scale))(x)
            # print('post fc size', x.shape)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class MLPActionSep(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    init_scale: Optional[float] = 1.
    use_layer_norm: bool = False
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False):
        x, action = _flatten_dict_special(x)
        print ('mlp action sep state post flatten', x.shape)
        print ('mlp action sep action post flatten', action.shape)

        for i, size in enumerate(self.hidden_dims):
            x_used = jnp.concatenate([x, action], axis=-1)
            x = nn.Dense(size, kernel_init=default_init())(x_used)
            print ('FF layers: ', x_used.shape, x.shape)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x