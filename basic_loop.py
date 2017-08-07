import numpy as np
import matplotlib.pyplot as plt

import os


class Experiment(object):
    """
    Wrapper to run experiments with
    """
    def __init__(self, algorithm, problem, max_episodes=100, max_t=100):
        """
        Initialize experiment with the right algorithm, problem
        """
        self.setup_problem(problem)
        self.setup_algorithm(algorithm)
        self.max_episodes = max_episodes
        self.max_t = max_t

    def setup_problem(self, problem):
        """
        Initialize MDP
        """
        self.problem = problem
        if problem == "treasure_hunt":
            from treasure_hunt import TreasureHunt
            self.mdp = TreasureHunt(n_states=9, grid_shape=(3, 3))
        self.mdp.configure()
        self.actions = self.mdp.actions
        self.grid_shape = self.mdp.grid_shape

    def setup_algorithm(self, algorithm):
        """
        Initialize agent that uses the given algorithm
        """
        if algorithm == "random":
            from random_agent import RandomAgent
            self.agent = RandomAgent(self.actions, self.grid_shape)
        # Add agent here
        else:
            raise ValueError("No algorithm implemented for '"
                             "{}'".format(algorithm))

    def main_loop(self):
        """
        Run the main loop interacting between the agent and the environment.
        """
        total_r = 0
        r_list = []
        for ep in xrange(self.max_episodes):
            new_state = self.mdp.reset()
            for t in xrange(self.max_t):
                state = new_state
                action = self.agent.get_action(state)
                new_state, reward, done, info = self.mdp.step(action)
                self.agent.update(state, action, reward, new_state,
                                  is_done=done)
                total_r += reward
                self.mdp.render()
                if done:
                    break
            r_list.append(total_r)
            total_r = 0
        self.agent.plot()
        return r_list

def smooth(values, step):
    """
    Smooth the list of values by averaging over every interval of step values.
    """
    smoothed_list = []
    n_values = len(values)
    for i in range(n_values/step):
        r = sum(values[i:i+step])/step
        smoothed_list.append(r)
    return smoothed_list

if __name__ == "__main__":
    # Setup experiment
    random_exp = Experiment("random", "treasure_hunt", max_episodes=1000,
                            max_t=1000)
    # Run simulations
    random_reward_list = random_exp.main_loop()
    # Smooth reward
    smooth_step = random_exp.max_t / 100
    random_reward_list = smooth(random_reward_list, smooth_step)

    # Plot rewards over time
    plt.plot(range(len(random_reward_list)), random_reward_list,
             label="Random Agent")

    plt.legend()
    plt.title("Reward over {} episodes, smoothed over {}-size "
              "intervals".format(random_exp.max_episodes, smooth_step))

    # Save plot
    plot_folder = "plots"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    plt.savefig(os.path.join(plot_folder, "reward_over_time.png"))
