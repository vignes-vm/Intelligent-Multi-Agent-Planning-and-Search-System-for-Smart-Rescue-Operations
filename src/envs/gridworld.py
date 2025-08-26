# File: src/envs/gridworld.py

import numpy as np
import random

class GridWorld:
    def __init__(self, width=20, height=20, n_coop=2, n_adv=0, n_targets=2, max_steps=500):
        self.width = width
        self.height = height
        self.n_coop = n_coop
        self.n_adv = n_adv
        self.n_targets = n_targets
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        """Reset environment at start of an episode"""
        self.steps = 0
        # Place cooperative agents randomly
        self.coop_agents = [self._random_position() for _ in range(self.n_coop)]
        # Place adversarial agents randomly
        self.adv_agents = [self._random_position() for _ in range(self.n_adv)]
        # Place targets randomly
        self.targets = [self._random_position() for _ in range(self.n_targets)]
        return self._get_obs()

    def _random_position(self):
        return (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def step(self, coop_actions, adv_actions=None):
        """Take a step in environment"""
        if adv_actions is None:
            adv_actions = [random.choice([0,1,2,3]) for _ in range(self.n_adv)]  # random moves

        self.steps += 1
        self.coop_agents = [self._move(pos, act) for pos, act in zip(self.coop_agents, coop_actions)]
        self.adv_agents = [self._move(pos, act) for pos, act in zip(self.adv_agents, adv_actions)]

        # Check if targets found
        rewards = [0] * self.n_coop
        done = False
        for i, pos in enumerate(self.coop_agents):
            if pos in self.targets:
                rewards[i] += 10
                self.targets.remove(pos)

        if not self.targets or self.steps >= self.max_steps:
            done = True

        return self._get_obs(), rewards, done

    def _move(self, pos, action):
        x, y = pos
        if action == 0:   # up
            y = max(0, y - 1)
        elif action == 1: # down
            y = min(self.height - 1, y + 1)
        elif action == 2: # left
            x = max(0, x - 1)
        elif action == 3: # right
            x = min(self.width - 1, x + 1)
        return (x, y)

    def _get_obs(self):
        return {
            "coop_agents": self.coop_agents,
            "adv_agents": self.adv_agents,
            "targets": self.targets,
            "steps": self.steps
        }

    def render(self):
        grid = np.full((self.height, self.width), ".")
        for tx, ty in self.targets:
            grid[ty, tx] = "T"
        for ax, ay in self.adv_agents:
            grid[ay, ax] = "A"
        for i, (cx, cy) in enumerate(self.coop_agents):
            grid[cy, cx] = str(i)
        print("\n".join(" ".join(row) for row in grid))
        print("-" * 40)
