import numpy as np

class BaseAgent(object):
    """
    Create an agent that returns random actions for every observation
    """

    def __init__(self, actions, state_dim, gamma=0.9):
        """
        Initialize the agent
        """
        self.actions = actions
        self.state_dim = state_dim
        self.gamma = gamma

    def get_action(self, state):
        """
        Empty function
        Select an action using the agent's policy
        """
        pass

    def update(self, state, action, reward, next_state, is_done=False):
        """
        Empty function
        Use the (state, action, reward, next_state) tuple sampled from the
        environment to update the agent in some way
        """
        pass

    def plot(self):
        """
        Empty function
        Plot the value functions found by the agent, if applicable
        """
        pass
