# adapted from jaxrl2
"""Replay buffer for DSRL-NA that stores executed actions and original observations."""
import collections

import jax
import numpy as np
from flax.core import frozen_dict

from jaxrl2.data.replay_buffer import ReplayBuffer


class CacheStorage:
    def __init__(self, capacity):
        self.capacity = capacity
        # Storing as list because cache structure can be complex (list of arrays, etc.)
        self.k_cache = [None for _ in range(capacity)]
        self.v_cache = [None for _ in range(capacity)]

    def insert(self, k, v, index):
        if index >= self.capacity:
            raise RuntimeError(
                "CacheStorage capacity exceeded; round-robin buffers do not support expansion."
            )
            self.expand()
        self.k_cache[index] = k
        self.v_cache[index] = v

    def expand(self):
        new_capacity = self.capacity * 2
        new_k = [None for _ in range(new_capacity)]
        new_v = [None for _ in range(new_capacity)]
        new_k[:self.capacity] = self.k_cache
        new_v[:self.capacity] = self.v_cache
        self.k_cache = new_k
        self.v_cache = new_v
        self.capacity = new_capacity

    def get(self, indices, key_type='k'):
        """Retrieve cache for given indices."""
        # indices can be numpy array
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        
        target = self.k_cache if key_type == 'k' else self.v_cache
        return [target[i] for i in indices]

class ReplayBufferNA(ReplayBuffer):
    """Extended replay buffer for DSRL-NA that stores executed actions and original obs."""

    def __init__(self, observation_space, action_space, executed_action_dim, capacity):
        """
        State shape should be a single int.
        """
        # Initialize base class (sets up observations, actions, rewards, masks, discount, etc.)
        super().__init__(observation_space, action_space, capacity)
        
        self.executed_action_dim = executed_action_dim

        print(f"Extending to DSRL-NA replay buffer")
        print(f"  - Noise action shape: {self.action_space.shape}")
        print(f"  - Executed action dim: {self.executed_action_dim}")

        # Add NA-specific fields to self.data
        self.data['executed_actions'] = np.empty((self.capacity, self.executed_action_dim), dtype=np.float32)

        # Original observations for distillation queries
        self.data['original_observations'] = [None for _ in range(self.capacity)]

        # Cache storage for selective fetching
        self.cache_storage = CacheStorage(self.capacity)

    def _expand_capacity(self):
        raise RuntimeError(
            "ReplayBufferNA._expand_capacity is deprecated; only round-robin buffers are supported."
        )

    def insert(self, data_dict):
        # Handle cache separately
        k_cache = data_dict.pop('original_k_cache', None)
        v_cache = data_dict.pop('original_v_cache', None)
        
        # Capture write index before super().insert() increments it
        current_idx = self.insert_index
        # Handle original observations separately (list of dicts)
        original_obs = data_dict.pop('original_observations')
        self.data['original_observations'][current_idx] = original_obs
        super().insert(data_dict)

        if k_cache is not None and v_cache is not None:
            self.cache_storage.insert(k_cache, v_cache, current_idx)

    def get_cache(self, indices, key_type='k'):
        return self.cache_storage.get(indices, key_type)

    def sample(self, batch_size, keys=None, indx=None):
        # Use base class sampling which handles round-robin boundary logic
        # and returns 'indices' and 'next_indices'
        data_dict = dict(super().sample(batch_size, keys, indx))
        
        indices = data_dict['indices']
        next_indices = data_dict['next_indices']
        
        # Derive original_next_* fields for distillation queries
        data_dict['original_next_observations'] = self._index_field(self.data['original_observations'], next_indices)
        
        return frozen_dict.freeze(data_dict)

    def get_iterator(self, batch_size, keys=None, indx=None, queue_size=2):
        keys_to_remove = ['original_observations', 'original_next_observations']
        queue = collections.deque()
        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                # Separate jittable and non-jittable data
                jit_data = {k: v for k, v in data.items() if k not in keys_to_remove}
                jit_data = jax.device_put(jit_data)
                non_jit_data = {k: v for k, v in data.items() if k in keys_to_remove}
                data = {**jit_data, **non_jit_data}
                queue.append(data)
        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def restore(self, filename):
        """Restore buffer from file, with backward compatibility for old snapshots."""
        super().restore(filename)
        # Parent already handles next_observations, next_actions removal
        if 'original_observations' not in self.data:
            raise RuntimeError(
                "Missing original_observations in replay buffer snapshot; older formats are not supported."
            )
        if 'executed_actions' not in self.data:
            raise RuntimeError(
                "Missing executed_actions in replay buffer snapshot; older formats are not supported."
            )
