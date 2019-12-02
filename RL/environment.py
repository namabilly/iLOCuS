import json
import os
import random
import numpy as np

class Environment(object):
    def __init__(self,
                driver_sim,
                env_size=(15, 15),
                objective=None):
        self.env_size = env_size
        if not objective:
            self.objective = np.ones((15,15))
        else:
            self.objective = objective
        self.objective /= np.sum(self.objective)
        self.driver_sim = driver_sim

    def reset(self):
        return self.driver_sim.reset()

    '''
    Main function step
    '''
    #action is a pricing table, representing the proposed price for each cell
    #output is next_state, reward
    #          next_state: the new distribution of the taxis
    #          reward: reward for the current pricing table

    def step(self, action):
        new_state, is_terminal = self.driver_sim.step(action)
        # new_state shape: (4, 15, 15)
        reward = self._compute_reward(new_state[1,:,:], self.objective)
        # print(reward)
        return np.copy(new_state), reward, is_terminal

    '''
    Calculate reward
    '''
    # compute the reward given the current distribution of taxis and desired distribution.
    def _compute_reward(self, state, objective):
        state = np.copy(state) + 1e-7

        # normalize
        state /= np.sum(state)

        # KL divergence
        return -np.sum(np.where(state != 0, state * np.log(state / objective), 0))
        