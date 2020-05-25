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
        if np.max(reward) < 10 and np.min(reward)< 0:
            is_terminal = True
        else:
            is_terminal = False
        return np.copy(new_state), reward, is_terminal


    '''
    Calculate reward
    '''
    # compute the reward given the current distribution of taxis and desired distribution.
    def _compute_reward(self, state, objective):
        # state = np.copy(state)
        state = state.astype(float)
        # normalize
        # print(state)
        state /= np.sum(state)
        state = state + 1e-5
        # print(state)
        # print(objective)
        # tmp_reward = -10*(state/np.exp(1))*(np.log(state/(np.exp(1)*objective)))
        # tmp_reward = 10*np.where(state < objective*np.exp(1), -100*(state/np.exp(1))*(np.log(state/(np.exp(1)*objective))), 500*(np.exp(1)*objective - state))
        # tmp_reward = np.where(state < objective*np.exp(1), -1000*(state/np.exp(1))*(np.log(state/(np.exp(1)*objective))), -50000*(state/np.exp(1))*(np.log(state/(np.exp(1)*objective))))
        tmp_reward = np.where(state < objective * np.exp(1),
                              -1000 * (state / np.exp(1)) * (np.log(state / (np.exp(1) * objective))),
                              -50000 * (state / np.exp(1)) * (np.log(state / (np.exp(1) * objective))))
        # print(tmp_reward + 0.1*np.mean(tmp_reward))
        # _tmp__ = np.reshape(tmp_reward + 0.5*np.mean(tmp_reward), SIZE_R*SIZE_C)
        # # print(_tmp__)
        # _tmp___ = np.reshape(_tmp__, [SIZE_R, SIZE_C])
        # print(_tmp___)
        # tmp = 1e-2 + np.abs((np.where(state != 0, np.log(state / objective), np.log(1e-5))))
        # tmp_reward = np.minimum(10/(1+tmp) -1 ,np.exp(1/tmp))
        # # KL divergence
        # return np.reshape(tmp_reward + 0.5*np.mean(tmp_reward) - 5 , SIZE_R*SIZE_C)
        return np.reshape(tmp_reward + 0.1*np.mean(tmp_reward), SIZE_R*SIZE_C)
        
