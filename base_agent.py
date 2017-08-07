import numpy as np

class BaseAgent(object):
    """
    Create an agent that returns random actions for every observation
    """

    def __init__(self, actions, state_dim, gamma=0.9, min_reward=-1,
                 max_reward=1):
        """
        Initialize the agent
        """
        self.actions = actions
        self.state_dim = state_dim
        self.min_r = min_reward
        self.max_r = max_reward
        self.gamma = gamma

    def get_action(self, state):
        """
        If no agent is specified, select action randomly
        """
        return self.actions[np.random.choice(range(len(self.actions)))]

    def update(self, state, action, reward, next_state, is_done=False):
        """
        Empty function
        """
        pass

    def pretty_print(self):
        """
        Empty function
        """
        pass

    def episode_end(self):
        """
        Empty function
        """
        return True

    def plot(self):
        """
        Empty function
        """
        pass
