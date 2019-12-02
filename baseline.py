
from Reaction.drivers import Drivers
from RL.environment import Environment
import numpy as np

if __name__ == '__main__':
    drivers = Drivers()
    env = Environment(driver_sim=drivers)
    env.reset()
    num_episodes = 10
    max_episode_length = 50
    num_actions = 4
    mean_reward = 0
    for _ in range(num_episodes):
        total_reward = 0
        env.reset()
        for t in range(1, max_episode_length+1):

            action_map = np.random.randint(num_actions, size=(15, 15))
            # print(action_map.shape)
            # Take 1 action
            next_state, reward, is_terminal = env.step(action_map)

            total_reward += reward
        
        mean_reward += total_reward
    print( mean_reward / num_episodes / max_episode_length)
