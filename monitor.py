''' Using Moniter to evaluate model
              and
    to generate video csapture '''

import argparse
import os

import numpy as np
from keras.models import load_model
from RL.environment import Environment
from reaction.drivers_false import Drivers

from RL.policy import LinearDecayGreedyEpsilonPolicy
from RL.objectives import mean_huber_loss, mean_huber_loss_duel
from RL.core import ReplayMemory


SIZE_R = 5
SIZE_C = 5

def get_output_folder(parent_dir, env_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-evaluate')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-evaluate{}'.format(experiment_id)
    return parent_dir

def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Evaluate model using Monitor')
    parser.add_argument('--env', default='driversim', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='deepQ', help='Directory to save data to')
    parser.add_argument("--num_actions", default=10, type=int, help="level of pricing")

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    print(args.output)
    os.makedirs(args.output, exist_ok=True)

    driver_sim = Drivers()
    env = Environment(driver_sim=driver_sim)

    policy = LinearDecayGreedyEpsilonPolicy(start_value=1, end_value=0.1,
                                                 num_steps=100, num_actions=args.num_actions)

    print('load trained model...')
    q_net = load_model('ilocus-v0/driverSim-v0-run157/qnet-1of5.h5',
                       custom_objects={'mean_huber_loss': mean_huber_loss})

    num_episodes = 50
    rewards = []

    mean_cost=[]
    eval_memory = ReplayMemory(10000, 1)
    tmp_rewards = []
    for episode in range(num_episodes):
        eval_memory.clear()
        prev_state = env.reset()
        eval_memory.append_state(prev_state)
        print("starting the simulator....")
        # print(_compute_reward(prev_state[1, :, :]))
        total_reward = 0
        for t in range(20):
            # env.render()
            fwd_states = eval_memory.gen_forward_state()
            fwd_res = q_net.predict_on_batch(np.asarray(fwd_states))
            # print(fwd_res)
            action_map = policy.select_action(fwd_res, False)
            # action_map = np.reshape(_action, (SIZE_R, SIZE_C))
            mean_cost.append(np.sum(action_map))
            # Take 1 action
            next_state, reward, is_terminal = env.step(action_map)
            # if t %10 == 1:
            #     tmp_reward = _compute_reward(next_state[1, :, :])
            #     print(tmp_reward)
            # if reward != 0:
            # print(total_reward)
            # print(next_state[1,:,:])
            print(action_map)
            total_reward += np.mean(reward)
            if is_terminal:
                print("Episode finished after {} timesteps".format(t + 1))
                break
            eval_memory.append_other(action_map, reward, t, is_terminal)
            prev_state = np.copy(next_state)
            eval_memory.append_state(prev_state)

        tmp_reward = _compute_reward(next_state[1, :, :])
        print(next_state[1, :, :])
        tmp_rewards.append(tmp_reward)
        # print(total_reward)
        rewards.append(total_reward)
    rewards = np.asarray(rewards)
    print(np.mean(rewards))
    print(np.std(rewards))
    print(np.mean(tmp_rewards))
    print(np.std(tmp_rewards))

def _compute_reward(state):
    objective = np.ones((SIZE_R, SIZE_C))
    objective /= np.sum(objective)
    state = np.copy(state) + 1e-7

    # normalize
    state /= np.sum(state)

    # KL divergence
    # return np.sum((np.where(state != 0, state*np.log(state / objective), 0)))
    return np.sum(np.sum(state * np.log(state / objective)))

if __name__ == '__main__':
    main()