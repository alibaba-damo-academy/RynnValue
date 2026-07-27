# adapted from jaxrl2
from typing import Union
from typing import Iterable, Optional
import jax 
import gym
import gym.spaces
import numpy as np
import pickle

import copy

from jaxrl2.data.dataset import Dataset, DatasetDict
import collections
from flax.core import frozen_dict

def _init_replay_dict(obs_space: gym.Space,
                      capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, capacity: int, ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.capacity = capacity

        print("making replay buffer of capacity ", self.capacity)

        observations = _init_replay_dict(self.observation_space, self.capacity)
        actions = np.empty((self.capacity, *self.action_space.shape), dtype=self.action_space.dtype)
        rewards = np.empty((self.capacity, ), dtype=np.float32)
        masks = np.empty((self.capacity, ), dtype=np.float32)
        discount = np.empty((self.capacity, ), dtype=np.float32)

        self.data = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'masks': masks,
            'discount': discount,
        }

        self.size = 0
        self.insert_index = 0
        self._traj_counter = 0
        self._start = 0
        self.traj_start_index = 0
        self.traj_bounds = dict()
        self.streaming_buffer_size = None # this is for streaming the online data
        self._traj_end_mask = np.zeros((self.capacity,), dtype=bool)

    def __len__(self) -> int:
        return self.size

    def length(self) -> int:
        return self.size

    def increment_traj_counter(self):
        if self.size > 0:
            # We use (insert_index - 1) % capacity because insert_index points to the *next* slot
            prev_idx = (self.insert_index - 1 + self.capacity) % self.capacity
            self._traj_end_mask[prev_idx] = True
        
        self.traj_bounds[self._traj_counter] = (self.traj_start_index, self.insert_index) # [start, end)
        self.traj_start_index = self.insert_index
        self._start = self.size # Keep track of total size for legacy or logging if needed, though _start seems unused now for bounds
        self._traj_counter += 1

    def get_random_trajs(self, num_trajs: int):
        self.which_trajs = np.random.randint(0, self._traj_counter, num_trajs)
        observations_list = []
        next_observations_list = []
        actions_list = []
        rewards_list = []
        terminals_list = []
        masks_list = []
        discount_list = []

        for i in self.which_trajs:
            start, end = self.traj_bounds[i]
            # Resample if the trajectory is wrapped around (invalid for straightforward slicing)
            while start > end:
                i = np.random.randint(0, self._traj_counter)
                start, end = self.traj_bounds[i]

            # handle this as a dictionary
            obs_dict_curr_traj = dict()
            for k in self.data['observations']:
                obs_dict_curr_traj[k] = self.data['observations'][k][start:end]
            observations_list.append(obs_dict_curr_traj)
            
            # Derive next_observations by shifting indices (last step uses same obs)
            next_obs_dict_curr_traj = dict()
            for k in self.data['observations']:
                next_obs = self.data['observations'][k][start:end].copy()
                if end - start > 1:
                    next_obs[:-1] = self.data['observations'][k][start+1:end]
                next_obs_dict_curr_traj[k] = next_obs
            next_observations_list.append(next_obs_dict_curr_traj)
            
            actions_list.append(self.data['actions'][start:end])
            rewards_list.append(self.data['rewards'][start:end])
            terminals_list.append(1-self.data['masks'][start:end])
            masks_list.append(self.data['masks'][start:end])


        
        batch = {
            'observations': observations_list,
            'next_observations': next_observations_list,
            'actions': actions_list,
            'rewards': rewards_list,
            'terminals': terminals_list,
            'masks': masks_list,
            
            
        }
        return batch
        
    def insert(self, data_dict: DatasetDict):
        # Round robin: no expansion
        # if self.size == self.capacity:
        #     self._expand_capacity()
        
        idx = self.insert_index
        self._traj_end_mask[idx] = False

        for x in data_dict:
            if x in self.data:
                if isinstance(data_dict[x], dict):
                    for y in data_dict[x]:
                        self.data[x][y][idx] = data_dict[x][y]
                else:                        
                    self.data[x][idx] = data_dict[x]
        
        self.insert_index = (self.insert_index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def _expand_capacity(self):
        raise RuntimeError(
            "ReplayBuffer._expand_capacity is deprecated; only round-robin buffers are supported."
        )
        """Double the buffer capacity."""
        new_capacity = self.capacity * 2
        
        new_data = {
            'observations': _init_replay_dict(self.observation_space, new_capacity),
            'actions': np.empty((new_capacity, *self.action_space.shape), dtype=self.action_space.dtype),
            'rewards': np.empty((new_capacity,), dtype=np.float32),
            'masks': np.empty((new_capacity,), dtype=np.float32),
            'discount': np.empty((new_capacity,), dtype=np.float32),
        }

        for x in self.data:
            if isinstance(self.data[x], np.ndarray):
                new_data[x][:self.capacity] = self.data[x]
            elif isinstance(self.data[x], dict):
                for y in self.data[x]:
                    new_data[x][y][:self.capacity] = self.data[x][y]
            else:
                raise TypeError()
                
        self.data = new_data
        new_traj_end_mask = np.zeros((new_capacity,), dtype=bool)
        new_traj_end_mask[:self.capacity] = self._traj_end_mask
        self._traj_end_mask = new_traj_end_mask
        self.capacity = new_capacity
        print(f"Expanded buffer capacity to {new_capacity}")
    
    def compute_action_stats(self):
        actions = self.data['actions']
        return {'mean': actions.mean(axis=0), 'std': actions.std(axis=0)}

    def normalize_actions(self, action_stats):
        # do not normalize gripper dimension (last dimension)
        copy.deepcopy(action_stats)
        action_stats['mean'][-1] = 0
        action_stats['std'][-1] = 1
        self.data['actions'] = (self.data['actions'] - action_stats['mean']) / action_stats['std']

    def _index_field(self, field, indices):
        """Index into a field (array, dict of arrays, or list) with given indices."""
        if isinstance(field, np.ndarray):
            return field[indices]
        if isinstance(field, dict):
            return {k: self._index_field(v, indices) for k, v in field.items()}
        if isinstance(field, list):
            return [field[i] for i in indices]
        raise TypeError()

    def sample(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        if self.streaming_buffer_size:
            max_size = self.streaming_buffer_size
        else:
            max_size = self.size
        if max_size <= 0:
            raise ValueError("ReplayBuffer.sample: buffer is empty.")

        def invalid_indices(sel_indices):
            masks = self.data['masks'][sel_indices]
            invalid = (self._traj_end_mask[sel_indices]) & (masks == 1)

            if self.size < self.capacity:
                latest_idx = self.size - 1
                invalid |= (masks == 1) & (sel_indices >= latest_idx)
            else:
                latest_idx = (self.insert_index - 1 + self.capacity) % self.capacity
                invalid |= (masks == 1) & (sel_indices == latest_idx)

            return invalid

        indices = np.random.randint(0, max_size, batch_size)
        invalid = invalid_indices(indices)
        if invalid.any():
            all_indices = np.arange(max_size)
            if invalid_indices(all_indices).all():
                raise ValueError(
                    "ReplayBuffer.sample: no valid indices (all candidates are trajectory ends with mask=1)."
                )
            while invalid.any():
                indices[invalid] = np.random.randint(0, max_size, invalid.sum())
                invalid = invalid_indices(indices)
        
        # Compute next indices, respecting episode boundaries
        if self.size < self.capacity:
            # Linear filling phase
            max_index = self.size - 1
            boundary = indices >= max_index
            boundary |= self._traj_end_mask[indices]
            boundary |= self.data['masks'][indices] == 0
            next_indices = np.where(boundary, indices, indices + 1)
        else:
            # Full circular buffer
            # The logical end is the element just before insert_index
            latest_idx = (self.insert_index - 1 + self.capacity) % self.capacity
            boundary = indices == latest_idx
            
            boundary |= self._traj_end_mask[indices]
            boundary |= self.data['masks'][indices] == 0
            next_indices = np.where(boundary, indices, (indices + 1) % self.capacity)
        
        # Index current data
        data_dict = {}
        for x in self.data:
            data_dict[x] = self._index_field(self.data[x], indices)
        
        # Derive next_* fields at sample time
        data_dict['next_observations'] = self._index_field(self.data['observations'], next_indices)
        data_dict['next_actions'] = self._index_field(self.data['actions'], next_indices)
        
        data_dict['indices'] = indices
        data_dict['next_indices'] = next_indices

        return frozen_dict.freeze(data_dict)

    def get_iterator(self, batch_size: int, keys: Optional[Iterable[str]] = None, indx: Optional[np.ndarray] = None, queue_size: int = 2):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)


    def save(self, filename):
        save_dict = dict(
            data=self.data,
            size=self.size,
            capacity=self.capacity,
            _traj_counter=self._traj_counter,
            _start=self._start,
            traj_bounds=self.traj_bounds,
            traj_end_mask=self._traj_end_mask,
        )
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f, protocol=4)


    def restore(self, filename):
        with open(filename, 'rb') as f:
            save_dict = pickle.load(f)
        self.data = save_dict['data']
        # Remove fields that are now derived at sample time (for backward compatibility)
        self.data.pop('next_observations', None)
        self.data.pop('next_actions', None)
        self.size = save_dict['size']
        self.capacity = save_dict.get('capacity', self.data['actions'].shape[0])
        self._traj_counter = save_dict['_traj_counter']
        self._start = save_dict['_start']
        self.traj_bounds = save_dict['traj_bounds']
        
        # Restore or init insert_index
        # If restoring from old buffer, assume linear fill up to size
        self.insert_index = save_dict.get('insert_index', self.size % self.capacity)
        self.traj_start_index = save_dict.get('traj_start_index', 0) # Default might be wrong for old buffers if not 0, but usually 0 start

        self._traj_end_mask = save_dict.get('traj_end_mask', np.zeros((self.capacity,), dtype=bool))
        if self._traj_end_mask.shape[0] != self.capacity:
            new_mask = np.zeros((self.capacity,), dtype=bool)
            new_mask[:self._traj_end_mask.shape[0]] = self._traj_end_mask
            self._traj_end_mask = new_mask
