# adapted from jaxrl2
from jaxrl2.networks.mlp import MLP

# Heavier policy classes (NormalPolicy, NormalTanhPolicy, LearnedStdNormalPolicy)
# pull in distrax + tensorflow_probability at import time, which initializes the
# JAX backend and preallocates GPU memory. Import them directly from their
# submodules (e.g. `from jaxrl2.networks.normal_policy import NormalPolicy`)
# when needed, instead of re-exporting them here.
