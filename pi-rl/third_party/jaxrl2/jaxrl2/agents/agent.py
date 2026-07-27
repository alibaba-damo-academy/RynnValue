# adapted from jaxrl2
from typing import Tuple

import numpy as np
import os
from flax.training import checkpoints
import pathlib
from flax.training.train_state import TrainState

from jaxrl2.agents.common import (eval_actions_jit, eval_log_prob_jit, eval_mse_jit, eval_reward_function_jit,
                                  sample_actions_jit, sample_actions_with_log_probs_jit)
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import PRNGKey


def get_batch_stats(actor):
    if hasattr(actor, 'batch_stats'):
        return actor.batch_stats
    else:
        return None

class Agent(object):
    _actor: TrainState
    _critic: TrainState
    _rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(self._actor.apply_fn, self._actor.params,
                                   observations, get_batch_stats(self._actor))
        return np.asarray(actions)

    def eval_log_probs(self, batch: DatasetDict) -> float:
        return eval_log_prob_jit(self._actor.apply_fn, self._actor.params, get_batch_stats(self._actor),
                                 batch)

    def eval_mse(self, batch: DatasetDict) -> float:
        return eval_mse_jit(self._actor.apply_fn, self._actor.params, get_batch_stats(self._actor),
                                 batch)

    def eval_reward_function(self, batch: DatasetDict) -> float:
        return eval_reward_function_jit(self._actor.apply_fn, self._actor.params, self._actor.batch_stats,
                                 batch)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, actions = sample_actions_jit(self._rng, self._actor.apply_fn,
                                          self._actor.params, observations, get_batch_stats(self._actor))

        self._rng = rng
        return np.asarray(actions)

    def sample_actions_with_log_probs(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample actions and return their log probabilities."""
        rng, actions, log_probs = sample_actions_with_log_probs_jit(
            self._rng, self._actor.apply_fn,
            self._actor.params, observations, get_batch_stats(self._actor))
        self._rng = rng
        return np.asarray(actions), np.asarray(log_probs)

    @property
    def _save_dict(self):
        return None

    def save_checkpoint(self, dir, step, keep_every_n_steps, keep=None):
        checkpoints.save_checkpoint(dir, self._save_dict, step, prefix='checkpoint', overwrite=True, keep=keep, keep_every_n_steps=keep_every_n_steps)

    def restore_checkpoint(self, dir):
        raise NotImplementedError


