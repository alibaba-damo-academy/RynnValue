# adapted from jaxrl2
"""Normalization modules for Flax."""

from typing import (Any, Callable, Optional, Tuple, Iterable, Union)

from jax import lax
from jax.nn import initializers
import jax.numpy as jnp

from flax.linen.module import Module, compact, merge_param


PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?

Axes = Union[int, Iterable[int]]

import flax.linen as nn
from flax.linen.module import Module, compact, merge_param

def _canonicalize_axes(rank: int, axes: Axes) -> Tuple[int, ...]:
  """Returns a tuple of deduplicated, sorted, and positive axes."""
  if not isinstance(axes, Iterable):
    axes = (axes,)
  return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))


def _abs_sq(x):
  """Computes the elementwise square of the absolute value |x|^2."""
  if jnp.iscomplexobj(x):
    return lax.square(lax.real(x)) + lax.square(lax.imag(x))
  else:
    return lax.square(x)


def _compute_stats(x: Array, axes: Axes,
                   axis_name: Optional[str] = None,
                   axis_index_groups: Any = None,
                   alpha: float = 0.5):
  """Computes mean and variance statistics.
  This implementation takes care of a few important details:
  - Computes in float32 precision for half precision inputs
  -  mean and variance is computable in a single XLA fusion,
    by using Var = E[|x|^2] - |E[x]|^2 instead of Var = E[|x - E[x]|^2]).
  - Clips negative variances to zero which can happen due to
    roundoff errors. This avoids downstream NaNs.
  - Supports averaging across a parallel axis and subgroups of a parallel axis
    with a single `lax.pmean` call to avoid latency.
  Arguments:
    x: Input array.
    axes: The axes in ``x`` to compute mean and variance statistics for.
    axis_name: Optional name for the pmapped axis to compute mean over.
    axis_index_groups: Optional axis indices.
  Returns:
    A pair ``(mean, var)``.
  """
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
  split_1, split_2 = jnp.split(x, 2, axis=0) # split x into two parts
  mean_s1 = jnp.mean(split_1, axes)
  mean_s2 = jnp.mean(split_2, axes)
  
  mean2_s1 = jnp.mean(_abs_sq(split_1), axes)
  mean2_s2 = jnp.mean(_abs_sq(split_2), axes)

  mean = alpha * mean_s1 + (1 - alpha) * mean_s2

  if axis_name is not None:
    concatenated_mean = jnp.concatenate([mean, mean2])
    mean, mean2 = jnp.split(
        lax.pmean(
            concatenated_mean,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups), 2)
  # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
  # to floating point round-off errors.
  var_s1 = mean2_s1 - _abs_sq(mean_s1)
  var_s2 = mean2_s2 - _abs_sq(mean_s2)
  var = alpha * var_s1 + (1 - alpha) * var_s2

  var = jnp.maximum(0., var)
  return mean, var


def _normalize(mdl: Module, x: Array, mean: Array, var: Array,
               reduction_axes: Axes, feature_axes: Axes,
               dtype: Dtype, param_dtype: Dtype,
               epsilon: float,
               use_bias: bool, use_scale: bool,
               bias_init: Callable[[PRNGKey, Shape, Dtype], Array],
               scale_init: Callable[[PRNGKey, Shape, Dtype], Array]):
  """"Normalizes the input of a normalization layer and optionally applies a learned scale and bias.
  Arguments:
    mdl: Module to apply the normalization in (normalization params will reside
      in this module).
    x: The input.
    mean: Mean to use for normalization.
    var: Variance to use for normalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
      for each specified feature.
    dtype: Dtype of the returned result.
    param_dtype: Dtype of the parameters.
    epsilon: Normalization epsilon.
    use_bias: If true, add a bias term to the output.
    use_scale: If true, scale the output.
    bias_init: Initialization function for the bias term.
    scale_init: Initialization function for the scaling function.
  Returns:
    The normalized input.
  """
  reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(x.ndim, feature_axes)
  stats_shape = list(x.shape)
  for axis in reduction_axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  feature_shape = [1] * x.ndim
  reduced_feature_shape = []
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
    reduced_feature_shape.append(x.shape[ax])
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  if use_scale:
    scale = mdl.param('scale', scale_init, reduced_feature_shape,
                      param_dtype).reshape(feature_shape)
    mul *= scale
  y *= mul
  if use_bias:
    bias = mdl.param('bias', bias_init, reduced_feature_shape,
                     param_dtype).reshape(feature_shape)
    y += bias
  return jnp.asarray(y, dtype)

class CrossNorm(Module):
  """CrossNorm Module.
  Usage Note:
  If we define a model with CrossNorm, for example::
    BN = nn.CrossNorm(use_running_average=False, momentum=0.9, epsilon=1e-5,
                      dtype=jnp.float32)
  The initialized variables dict will contain in addition to a 'params'
  collection a separate 'batch_stats' collection that will contain all the
  running statistics for all the BatchNorm layers in a model::
    vars_initialized = BN.init(key, x)  # {'params': ..., 'batch_stats': ...}
  We then update the batch_stats during training by specifying that the
  `batch_stats` collection is mutable in the `apply` method for our module.::
    vars_in = {'params': params, 'batch_stats': old_batch_stats}
    y, mutated_vars = BN.apply(vars_in, x, mutable=['batch_stats'])
    new_batch_stats = mutated_vars['batch_stats']
  During eval we would define BN with `use_running_average=True` and use the
  batch_stats collection from training to set the statistics.  In this case
  we are not mutating the batch statistics collection, and needn't mark it
  mutable::
    vars_in = {'params': params, 'batch_stats': training_batch_stats}
    y = BN.apply(vars_in, x)
  Attributes:
    use_running_average: if True, the statistics stored in batch_stats
      will be used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of
      the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma).
      When the next layer is linear (also e.g. nn.relu), this can be disabled
      since the scaling will be done by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over
      the examples on the first two and last two devices. See `jax.lax.psum`
      for more details.

      Note modified original BatchNorm module
  """
  use_running_average: Optional[bool] = None
  axis: int = -1
  momentum: float = 0.99
  epsilon: float = 1e-5
  dtype: Dtype = jnp.float32
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
  axis_name: Optional[str] = None
  axis_index_groups: Any = None
  alpha: float = 0.5

  @compact
  def __call__(self, x, use_running_average: Optional[bool] = None):
    """Normalizes the input using batch statistics.
    NOTE:
    During initialization (when parameters are mutable) the running average
    of the batch statistics will not be updated. Therefore, the inputs
    fed during initialization don't need to match that of the actual input
    distribution and the reduction axis (set with `axis_name`) does not have
    to exist.
    Args:
      x: the input to be normalized.
      use_running_average: if true, the statistics stored in batch_stats
        will be used instead of computing the batch statistics on the input.
    Returns:
      Normalized inputs (the same shape as inputs).
    """

    use_running_average = merge_param(
        'use_running_average', self.use_running_average, use_running_average)
    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
    feature_shape = [x.shape[ax] for ax in feature_axes]

    # see NOTE above on initialization behavior
    initializing = self.is_mutable_collection('params')

    ra_mean = self.variable('batch_stats', 'mean',
                            lambda s: jnp.zeros(s, jnp.float32),
                            feature_shape)
    ra_var = self.variable('batch_stats', 'var',
                           lambda s: jnp.ones(s, jnp.float32),
                           feature_shape)

    if use_running_average:
      mean, var = ra_mean.value, ra_var.value
    else:
      mean, var = _compute_stats(
          x, reduction_axes,
          axis_name=self.axis_name if not initializing else None,
          axis_index_groups=self.axis_index_groups, alpha=self.alpha)

      if not initializing:
        ra_mean.value = self.momentum * ra_mean.value + (1 -
                                                         self.momentum) * mean
        ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

    return _normalize(
        self, x, mean, var, reduction_axes, feature_axes,
        self.dtype, self.param_dtype, self.epsilon,
        self.use_bias, self.use_scale,
        self.bias_init, self.scale_init)