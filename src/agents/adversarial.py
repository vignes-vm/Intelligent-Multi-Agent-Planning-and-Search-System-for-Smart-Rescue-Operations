# File: src/agents/adversarial.py

import random

class AdversarialAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def act(self, observation):
        """
        Decide next move.
        For now: random move (same as cooperative).
        Later: will include adversarial strategies.
        """
        return random.choice([0, 1, 2, 3])
