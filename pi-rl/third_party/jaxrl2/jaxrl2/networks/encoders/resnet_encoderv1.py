# adapted from jaxrl2
import flax.linen as nn
import jax.numpy as jnp

from functools import partial
from typing import Any, Callable, Sequence, Tuple
from jaxrl2.networks.constants import default_init, xavier_init, kaiming_init
from jaxrl2.networks.encoders.spatial_softmax import SpatialSoftmax
from jaxrl2.networks.encoders.cross_norm import CrossNorm

ModuleDef = Any


class MyGroupNorm(nn.GroupNorm):

    def __call__(self, x):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)

class ResNetBlock(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x, ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm()(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class ResNetEncoder(nn.Module):
    """ResNetV1."""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv
    norm: str = 'batch'
    use_spatial_softmax: bool = True
    softmax_temperature: float = 1.0

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        
        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        conv = partial(self.conv, use_bias=False, dtype=self.dtype, kernel_init=kaiming_init())
        if self.norm == 'batch':
            norm = partial(nn.BatchNorm,
                           use_running_average=not train,
                           momentum=0.9,
                           epsilon=1e-5,
                           dtype=self.dtype)
        elif self.norm == 'group':
            norm = partial(MyGroupNorm,
                           num_groups=4,
                           epsilon=1e-5,
                           dtype=self.dtype)
        elif self.norm == 'cross':
            norm = partial(CrossNorm,
                           use_running_average=not train,
                           momentum=0.9,
                           epsilon=1e-5,
                           dtype=self.dtype)
        elif self.norm == 'layer':
            norm = partial(nn.LayerNorm, 
                epsilon=1e-5,
                dtype=self.dtype,
            )
        else:
            raise ValueError('norm not found')

        # print('input ', x.shape)
        strides = (2, 2, 2, 1, 1)
        x = conv(self.num_filters, (7, 7), (strides[0], strides[0]),
                 padding=[(3, 3), (3, 3)],
                 name='conv_init')(x)
        # print('post conv1', x.shape)

        x = norm(name='bn_init')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(strides[1], strides[1]), padding='SAME')
        # print('post maxpool1', x.shape)
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                stride = (strides[i + 1], strides[i + 1]) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2 ** i,
                                   strides=stride,
                                   conv=conv,
                                   norm=norm,
                                   act=self.act)(x)
            #     print('post block layer ', x.shape)
            # print('post block ', x.shape)

        if self.use_spatial_softmax:
            height, width, channel = x.shape[len(x.shape) - 3:]
            pos_x, pos_y = jnp.meshgrid(
                jnp.linspace(-1., 1., height),
                jnp.linspace(-1., 1., width)
            )
            pos_x = pos_x.reshape(height * width)
            pos_y = pos_y.reshape(height * width)
            # print('pre spatial softmax', x.shape)
            x = SpatialSoftmax(height, width, channel, pos_x, pos_y, self.softmax_temperature)(x)
            # print('post spatial softmax', x.shape)
        else:
            x = jnp.mean(x, axis=(len(x.shape) - 3,len(x.shape) - 2))
            # print('post flatten', x.shape)
        return x

ResNetSmall = partial(ResNetEncoder, stage_sizes=(1, 1, 1, 1),
                   block_cls=ResNetBlock)
ResNet18 = partial(ResNetEncoder, stage_sizes=(2, 2, 2, 2),
                   block_cls=ResNetBlock)
ResNet34 = partial(ResNetEncoder, stage_sizes=(3, 4, 6, 3),
                   block_cls=ResNetBlock)
