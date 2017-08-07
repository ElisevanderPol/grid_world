from base_agent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    """
    Create an agent that returns random actions for every observation
    """

    def __init__(self, actions, state_dim):
        """
        Initialize the agent
        """
        super(RandomAgent, self).__init__(actions, state_dim)
