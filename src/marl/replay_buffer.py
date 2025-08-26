import numpy as np
import random
from collections import deque

class MultiAgentReplayBuffer:
    """
    Stores joint experiences for multi-agent training.
    Each entry packs:
      obs        : list of per-agent observations (tuples, arrays, or dicts you encode)
      actions    : list of per-agent discrete actions (ints)
      rewards    : list of per-agent rewards (floats)
      next_obs   : list of per-agent next observations
      done       : bool (episode termination)
    """
    def __init__(self, capacity: int = 100_000, seed: int = 42):
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)
        np.random.seed(seed)

    def push(self, obs, actions, rewards, next_obs, done: bool):
        # We store raw python objects to keep it flexible.
        self.buffer.append((obs, actions, rewards, next_obs, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs_b, act_b, rew_b, next_obs_b, done_b = zip(*batch)
        return list(obs_b), list(act_b), list(rew_b), list(next_obs_b), np.array(done_b, dtype=np.float32)
