from base_agent import BaseAgent
import numpy as np
import random

class RandomAgent(BaseAgent):
    """
    Create an agent that returns random actions for every observation
    """

    def __init__(self, actions, state_dim):
        """
        Initialize the agent
        """
        super(RandomAgent, self).__init__(actions, state_dim)

    def get_action(self, state):
        """
        Return randomly sampled action
        """
        return random.choice(self.actions)
