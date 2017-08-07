This is a simple grid world toy problem

Some of the states give a negative reward, and the terminating goal state
gives a positive reward. The agent's goal is to get to the final state (i.e.
finding the treasure) without hitting the traps on the way.

An example agent is random_agent.py (which inherits from base_agent.py) and the
problem itself (including transition function and reward function) is specified
in treasure_hunt.py
