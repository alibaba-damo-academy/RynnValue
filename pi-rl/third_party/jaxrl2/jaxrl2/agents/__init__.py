# adapted from jaxrl2
# Learners are imported via their full paths (e.g.
# `from jaxrl2.agents.pi_iql import PiIQLLearner`). Eagerly loading PixelSAC /
# PixelIQL here pulled in distrax + tensorflow_probability via their policy
# networks, which initializes the JAX backend and preallocates GPU memory just
# from importing the package.
