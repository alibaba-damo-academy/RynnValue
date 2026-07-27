# adapted from jaxrl2
from typing import Dict, Tuple

import jax
from flax.training.train_state import TrainState


def update_temperature(
        temp: TrainState, entropy: float,
        target_entropy: float) -> Tuple[TrainState, Dict[str, float]]:

    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {
            'temperature': temperature,
            'temperature_loss': temp_loss
        }

    grads, info = jax.grad(temperature_loss_fn, has_aux=True)(temp.params)
    new_temp = temp.apply_gradients(grads=grads)

    return new_temp, info