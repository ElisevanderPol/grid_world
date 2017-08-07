import numpy as np

class TreasureHunt(object):
    """
    Simple toy problem with the goal to find the rewarding state, while
    avoiding the states that give punishment
    """

    def __init__(self, n_states=4, grid_shape=(2, 2), slip=0.1):
        """
        Initialize the toy problem by specifiying the number of states, the
        corresponding shape of the grid world, and the probability that the
        agent will slip and go in a random direction after taking an action
        """
        self.n_states = n_states
        self.grid_shape = grid_shape
        self.state_size = len(grid_shape)
        self.slip = slip
        self.done = False
        self.max_reward = 0.

    def configure(self):
        """
        Build the world and environmental dynamics
        """
        self.actions = [(-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                        (0, 0)]

        self.transitions = self.deterministic_transitions

        # For now, the reward depends only on the next state. However, in the
        # general case it depends on state, action and next state
        self.xend = self.grid_shape[0]-1
        self.yend = self.grid_shape[1]-1
        self.reward_func = {(self.xend, i): -1. for i in range(self.yend-1)}
        self.reward_func[(self.xend, self.yend)] = 1.
        self.reward = self.terminal_reward

        # This is hardcoded for now
        self.starting_states = [(0, 0)]
        self.terminating_states = [(self.xend, self.yend)]

    def reset(self):
        """
        Reset the world to the starting position
        """
        state_index = np.random.choice(range(len(self.starting_states)))
        self.current_state = self.starting_states[state_index]
        self.done = False
        return self.current_state

    def render(self):
        """
        Print the environment as a grid
        """
        grid = np.zeros(self.grid_shape)
        grid[self.current_state] = 1
        print(" ".join(["-" for i in range(2*self.n_states)]))
        for row in grid:
            print "|",
            for cell in row:
                if cell == 0:
                    print "_ |",
                else:
                    print "X |",
            print ""
        print(" ".join(["-" for i in range(2*self.n_states)]))

    def step(self, action):
        """
        Take an action in the environment, update the environment, and return
        the new state, the reward, a boolean indicating whether or not the
        episode has ended, and optionally a string with more information
        """
        new_state, reward = self.take_action(action)
        return new_state, reward, self.done, ""

    def deterministic_transitions(self, state, action):
        """
        Simply compute the next state given the state and action
        """
        if state == (self.xend, self.yend):
            return state
        new_state = (min(max(state[0] + action[0], 0), self.xend),
                     min(max(state[1] + action[1], 0), self.yend))
        return new_state

    def transition(self, state, action):
        """
        Compute the next state that this action should give in the
        deterministic case
        """
        if state == (self.xend, self.yend):
            return state
        new_state = (min(max(state[0] + action[0], 0), self.xend),
                     min(max(state[1] + action[1], 0), self.yend))
        return new_state

    def transition_function(self, state, action):
        """
        Return the probability distribution over next states, given a state and
        action
        """
        # Get actual next state
        next_state = self.transition(state, action)
        # Get all the neighbours
        state_neighbours = self.get_neighbours(state)
        state_neighbours.append(next_state)
        state_neighbours = list(set(state_neighbours))
        division = self.slip / max(len(state_neighbours)-1., 1.)
        transition_list = []
        end_state = (self.xend, self.yend)
        if state == end_state:
            old_slip = self.slip
            self.slip = 0.0
        for nb in state_neighbours:
            if nb == next_state:
                transition_list.append(1.-self.slip)
            else:
                transition_list.append(division)
        if state == end_state:
            self.slip = old_slip
        return transition_list, state_neighbours

    def get_neighbours(self, state):
        """
        Get the state's neighbouring states (including itself, possibly)
        """
        state_list = []
        for action in self.actions:
            state_list.append(self.transition(state, action))
        return list(set(state_list))

    def state_dependent_reward(self, state, action, next_state=None):
        """
        Return the reward
        """
        if state in self.terminating_states:
            return 1 + self.max_reward
        try:
            if next_state is not None:
                reward = self.reward_func[next_state] + self.max_reward
            else:
                reward = self.reward_func[state] + self.max_reward
        # If there is no key for this state, the reward is 0
        except KeyError:
            reward = 0 + self.max_reward
        return reward


    def reward_for_state(self, state):
        """
        Return the reward for just being in a state, regardless of transitions
        """
        if state in self.terminating_states:
            return 1.
        try:
            reward = self.reward_func[state]
        except KeyError:
            reward = 0
        return reward


    def take_action(self, action):
        """
        Sample a new state according to the transition model, and a reward
        according to the reward function
        """
        new_state = self.sample_state(self.current_state, action)
        reward = self.sample_reward(self.current_state, action, new_state)
        self.current_state = new_state
        if new_state in self.terminating_states:
            self.done = True
        return new_state, reward

    def sample_state(self, state, action):
        """
        Sample a new state according to the transition model
        """
        p, next_states = self.transition_function(state, action)
        index = np.random.choice(range(len(next_states)), p=p)
        return next_states[index]

    def sample_reward(self, state, action, next_state=None):
        """
        Sample a reward from the reward function. For now, it is deterministic
        """
        reward = self.reward(state, action, next_state)
        return reward

    def terminal_reward(self, state, action, next_state=None):
        """
        Give a reward upon reaching the final state, and then never again
        """
        end_state = (self.xend, self.yend)
        if state == end_state and next_state == end_state:
            return 0.
        elif next_state == end_state:
            return 1.
        elif next_state is not None:
            try:
                return self.reward_func[next_state]
            except KeyError:
                return 0.
        else:
            try:
                return self.reward_func[state]
            except KeyError:
                return 0.



if __name__ == "__main__":
    th = TreasureHunt(n_states=16, grid_shape=(4, 4))
    th.configure()
    th.reset()
    time = 100
    total_r = 0.
    for i in range(time):
        action = th.actions[np.random.choice(range(len(th.actions)))]
        s, r, d, info = th.step(action)
        total_r += r
        th.render()
    print("Total reward: {}".format(total_r/time))
