# File: src/agents/cooperative.py

import random

class CooperativeAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def act(self, observation):
        """
        Decide next move.
        For now: random move (0=up,1=down,2=left,3=right).
        Later: this will be replaced by learned RL policy.
        """
        return random.choice([0, 1, 2, 3])
