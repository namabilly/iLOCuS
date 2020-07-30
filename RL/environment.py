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
        self.objective = 2*np.ones((SIZE_R, SIZE_C))

        self.objective[1:4,1:4] = 6
        self.objective[2, 2] = 20
        self.objective /= np.sum(self.objective)

        self.driver_sim = driver_sim

    def reset(self, seed=None):
        return self.driver_sim.reset(seed_=seed)

    '''
    Main function step
    '''
    #action is a pricing table, representing the proposed price for each cell
    #output is next_state, reward
    #          next_state: the new distribution of the taxis
    #          reward: reward for the current pricing table

    def step(self, action):
        old_state = self.driver_sim.state()
        # old_reward = self._compute_reward(old_state[1, :, :], self.objective)
        new_state = self.driver_sim.step(action)
        # new_state shape: (4, 15, 15)
        # new_reward = self._compute_reward(new_state[1,:,:], self.objective)
        # reward = new_reward - old_reward
        reward = self._compute_reward(new_state[1, :, :], self.objective)
        # print(np.max(reward))
        # print(np.min(reward))
        # print(np.mean(reward))
        # if np.max(reward) < 12 and np.min(reward)< 0:
        if np.mean(reward) < 0:
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
        # tmp_reward = np.multiply(state, self.objective)
        # return np.reshape(np.mean(tmp_reward)*np.ones((SIZE_R,SIZE_C)), SIZE_R*SIZE_C)
        # normalize
        # print(state)
        state /= np.sum(state)
        state = state + 1e-5
        tmp_reward = -100 * objective * (np.log(objective / state))
        return np.reshape((np.sum(tmp_reward) + 400)* np.ones((SIZE_R, SIZE_C)), SIZE_R * SIZE_C)
        # print(state)
        # print(objective)
        # tmp_reward = -10*(state/np.exp(1))*(np.log(state/(np.exp(1)*objective)))
        # tmp_reward = 10*np.where(state < objective*np.exp(1), -100*(state/np.exp(1))*(np.log(state/(np.exp(1)*objective))), 500*(np.exp(1)*objective - state))
        # tmp_reward_1 = np.where(state < objective*np.exp(1), -1000*(state/np.exp(1))*(np.log(state/(np.exp(1)*objective))), -5000*(state/np.exp(1))*(np.log(state/(np.exp(1)*objective))))
        # tmp_reward = -1000 * state * (np.log(state / objective))
        # tmp_reward = -100 * objective * (np.log(objective/state))
        # return np.reshape((np.sum(tmp_reward) +400)* np.ones((SIZE_R, SIZE_C)), SIZE_R * SIZE_C)
        # print(tmp_reward)
        # print(np.mean(tmp_reward_2))
        # print(tmp_reward + 0.1*np.mean(tmp_reward))
        # _tmp__ = np.reshape(tmp_reward + 0.5*np.mean(tmp_reward), SIZE_R*SIZE_C)
        # # print(_tmp__)
        # _tmp___ = np.reshape(_tmp__, [SIZE_R, SIZE_C])
        # print(_tmp___)
        # tmp = 1e-2 + np.abs((np.where(state != 0, np.log(state / objective), np.log(1e-5))))
        # tmp_reward = np.minimum(10/(1+tmp) -1 ,np.exp(1/tmp))
        # # KL divergence
        # return np.reshape(tmp_reward + 0.5*np.mean(tmp_reward) - 5 , SIZE_R*SIZE_C)
        # return np.reshape(tmp_reward+np.mean(tmp_reward), SIZE_R*SIZE_C)
        
