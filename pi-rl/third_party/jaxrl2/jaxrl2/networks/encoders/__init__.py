# adapted from jaxrl2
# Intentionally empty — encoder modules are imported via their full paths
# (e.g. `from jaxrl2.networks.encoders.networks import PixelMultiplexer`).
# Re-exporting policy classes here used to drag distrax + tensorflow_probability
# into every encoder import, which initializes the JAX backend and preallocates
# GPU memory.
