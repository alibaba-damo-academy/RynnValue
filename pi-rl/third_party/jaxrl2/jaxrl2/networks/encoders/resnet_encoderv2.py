# adapted from IDQL (IDQL-main/jaxrl5/networks/encoders/ln_resnet_encoder.py: lines 1-18, 21-28, 31-51)
# Based on:
# https://github.com/google/flax/blob/main/examples/imagenet/models.py
# and
# https://github.com/google-research/big_transfer/blob/master/bit_jax/models.py
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any


class ResNetV2Block(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.norm()(x)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides)(residual)

        return residual + y


class MyGroupNorm(nn.GroupNorm):

    def __call__(self, x):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)


class ResNetV2Encoder(nn.Module):
    """ResNetV2."""
    stage_sizes: Sequence[int]
    num_filters: int = 16
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    norm: str = 'batch'

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        if self.norm == 'batch':
            norm = partial(nn.BatchNorm,
                           use_running_average=not train,
                           momentum=0.9,
                           epsilon=1e-5,
                           dtype=self.dtype)
        elif self.norm == 'groupnorm':
            norm = partial(MyGroupNorm,
                           num_groups=4,
                           epsilon=1e-5,
                           dtype=self.dtype)
        else:
            raise ValueError('norm not found')

        x = x.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        x = conv(self.num_filters, (3, 3))(x)
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = ResNetV2Block(self.num_filters * 2**i,
                                  strides=strides,
                                  conv=conv,
                                  norm=norm,
                                  act=self.act)(x)

        x = norm()(x)
        x = self.act(x)
        return x.reshape((*x.shape[:-3], -1))
