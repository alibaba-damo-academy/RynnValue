# adapted from IDQL (IDQL-main/jaxrl5/data/dataset.py: lines 17-28, 31-37, 45-80, 87-105)
from typing import Dict, Iterable, Optional, Tuple, Union
import collections
import jax
import numpy as np
from gym.utils import seeding
import jax.numpy as jnp
from jaxrl2.types import DataType

DatasetDict = Dict[str, DataType]
from flax.core import frozen_dict

def concat_recursive(batches):
    new_batch = {}
    for k, v in batches[0].items():
        if isinstance(v, frozen_dict.FrozenDict):
            new_batch[k] = concat_recursive([batches[0][k], batches[1][k]])
        else:
            new_batch[k] = np.concatenate([b[k] for b in batches], 0)
    return new_batch

def _check_lengths(dataset_dict: DatasetDict,
                   dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, 'Inconsistent item lengths in the dataset.'
        else:
            raise TypeError('Unsupported type.')
    return dataset_len


def _split(dataset_dict: DatasetDict,
           index: int) -> Tuple[DatasetDict, DatasetDict]:
    train_dataset_dict, test_dataset_dict = {}, {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            train_v, test_v = _split(v, index)
        elif isinstance(v, np.ndarray):
            train_v, test_v = v[:index], v[index:]
        else:
            raise TypeError('Unsupported type.')
        train_dataset_dict[k] = train_v
        test_dataset_dict[k] = test_v
    return train_dataset_dict, test_dataset_dict


def _sample(dataset_dict: Union[np.ndarray, DatasetDict],
            indx: np.ndarray) -> DatasetDict:
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch


class Dataset(object):

    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        # Seeding similar to OpenAI Gym:
        # https://github.com/openai/gym/blob/master/gym/spaces/space.py#L46
        self._np_random = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.RandomState:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def __len__(self) -> int:
        return self.dataset_len

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        if indx is None:
            if hasattr(self.np_random, 'integers'):
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        return frozen_dict.freeze(batch)

    def split(self, ratio: float) -> Tuple['Dataset', 'Dataset']:
        assert 0 < ratio and ratio < 1
        index = int(self.dataset_len * ratio)
        train_dataset_dict, test_dataset_dict = _split(self.dataset_dict,
                                                       index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)
