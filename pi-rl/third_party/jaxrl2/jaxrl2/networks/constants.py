# adapted from jaxrl2
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

def xavier_init():
    return nn.initializers.xavier_normal()

def kaiming_init():
    return nn.initializers.kaiming_normal()