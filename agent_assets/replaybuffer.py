import numpy as np
import random

class ReplayBuffer():
    def __init__(self, buffer_size, observation_space:dict):
        """
        Stores observation space in uint8 dtype
        """
        self.size = buffer_size
        self.obs_buffer = {}
        for name, space in observation_space.spaces.items():
            shape = space.shape
            self.obs_buffer[name] = np.zeros([buffer_size]+list(shape),
                                            dtype=np.uint8)
        self.action_buffer = np.zeros(buffer_size, dtype=np.int32)
        self.reward_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.done_buffer = np.zeros(buffer_size, dtype=np.bool)
        self.next_idx = 0
        self.num_in_buffer = 0

    def store_obs(self, observation:dict) :
        """
        Give the original observation in uint8
        """
        for name, obs in observation.items() :
            self.obs_buffer[name][self.next_idx] = obs
        saved_idx = self.next_idx
        self.next_idx = (self.next_idx +1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer +1)
        return saved_idx

    def store_effect(self, idx, action, reward, done) :
        self.action_buffer[idx] = action
        self.reward_buffer[idx] = reward
        self.done_buffer[idx] = done

    def sample(self, batch_size):
        indices = np.array(random.sample(range(self.num_in_buffer-1),batch_size))
        next_indices = indices + 1
        obs_sample = {}
        next_obs_sample = {}
        for name, buf in self.obs_buffer.items():
            obs_sample[name] = buf[indices]
            next_obs_sample[name] = buf[next_indices]
        action_sample = self.action_buffer[indices]
        reward_sample = self.reward_buffer[indices]
        done_sample = self.done_buffer[indices]
        return (obs_sample, 
                action_sample, 
                reward_sample, 
                done_sample,
                next_obs_sample
                )