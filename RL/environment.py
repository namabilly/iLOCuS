import json
import os
import random
import numpy as np
from driver_func_test import DriverSim

class Environment(object):
    def __init__(self,
                env_size=(15, 15),
                objective=None,
                driver_sim):
        self.env_size = env_size
        self.objective = objective
        self.driver_sim = driver_sim
    '''
    Main function step
    '''
    #action is a pricing table, representing the proposed price for each cell
    #output is next_state, reward
    #          next_state: the new distribution of the taxis
    #          reward: reward for the current pricing table

    def step(self, action):
        new_state, is_terminal = self.driver_sim.react(self.state, action)
        reward = self._compute_reward(new_state, self.objective)
        self.state = new_state

        return np.copy(new_state), reward, is_terminal

    '''
    Calculate reward
    '''
    # compute the reward given the current distribution of taxis and desired distribution.
    def _compute_reward(self, state, objective):
        # KL divergence
        return np.sum(np.where(state != 0, state * np.log(state / objective), 0))
        