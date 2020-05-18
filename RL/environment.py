import json
import os
import random
import numpy as np
import copy
SIZE_R = 5
SIZE_C = 5

class Environment(object):
    def __init__(self,
                driver_sim,
                env_size=(SIZE_R, SIZE_C),
                objective=None):
        self.env_size = env_size
        if not objective:
            self.objective = np.ones((SIZE_R, SIZE_C))
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
        new_state = self.driver_sim.step(action)
        # new_state shape: (4, 15, 15)
        reward = self._compute_reward(new_state[1,:,:], self.objective)
        # print(np.min(reward))
        # print(np.max(reward))
        if np.max(reward) < 2.27:
            is_terminal = True
        else:
            is_terminal = False
        return np.copy(new_state), reward, is_terminal


    '''
    Calculate reward
    '''
    # compute the reward given the current distribution of taxis and desired distribution.
    def _compute_reward(self, state, objective):
        state = np.copy(state) + 1e-7

        # normalize
        state /= np.sum(state)
        tmp = 1e-2 + np.abs((np.where(state != 0, np.log(state / objective), 1e+7)))
        # KL divergence
        return np.minimum(1000/(1+tmp) ,0.1*np.exp(1/tmp))
        
