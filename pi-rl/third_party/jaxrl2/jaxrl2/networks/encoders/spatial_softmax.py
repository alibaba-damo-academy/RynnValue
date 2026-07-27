# adapted from jaxrl2
from typing import Dict, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks.constants import default_init, xavier_init, kaiming_init

from functools import partial
from typing import Any, Callable, Sequence, Tuple

ModuleDef = Any

class SpatialSoftmax(nn.Module):
    height: int
    width: int
    channel: int
    pos_x: jnp.ndarray
    pos_y: jnp.ndarray
    temperature: None
    log_heatmap: bool = False

    @nn.compact
    def __call__(self, feature):
        if self.temperature == -1:
            from jax.nn import initializers
            # print("Trainable temperature parameter")
            temperature = self.param('softmax_temperature', initializers.ones, (1), jnp.float32)
        else:
            temperature = 1.

        # print(temperature)
        assert len(feature.shape) == 4
        batch_size, num_featuremaps = feature.shape[0], feature.shape[3]
        feature = feature.transpose(0, 3, 1, 2).reshape(batch_size, num_featuremaps, self.height * self.width)

        softmax_attention = nn.softmax(feature / temperature)
        expected_x = jnp.sum(self.pos_x * softmax_attention, axis=2, keepdims=True).reshape(batch_size, num_featuremaps)
        expected_y = jnp.sum(self.pos_y * softmax_attention, axis=2, keepdims=True).reshape(batch_size, num_featuremaps)
        expected_xy = jnp.concatenate([expected_x, expected_y], axis=1)

        expected_xy = jnp.reshape(expected_xy, [batch_size, 2*num_featuremaps])
        return expected_xy

