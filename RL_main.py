#!/usr/bin/env python
"""Run iLOCuS with DQN."""
import sys

import argparse
import os
import shutil

import tensorflow as tf
import numpy as np

from RL.environment import Environment
from RL.policy import LinearDecayGreedyEpsilonPolicy
from RL.agent import DQNAgent
from RL.core import ReplayMemory
from reaction.drivers import Drivers
from RL.model import create_model
from RL.objectives import mean_huber_loss
from keras.optimizers import Adam, SGD

def main():  # noqa: D103
    parser = argparse.ArgumentParser(description="Run DQN on iLOCuS")
    parser.add_argument("--network_name", default="deep_q_network", type=str, help="Type of model to use")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--map_shape", default=(15, 15), type=tuple, help="map size")
    parser.add_argument("--num_actions", default=10, type=int, help="level of pricing")

    parser.add_argument("--gamma", default=0.8, type=float, help="Discount factor")
    parser.add_argument("--alpha", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration probability for epsilon-greedy")
    parser.add_argument("--target_update_freq", default=10000, type=int,
                        help="Frequency for copying weights to target network")
    parser.add_argument("--num_iterations", default=5000000, type=int,
                        help="Number of overal interactions to the environment")

    parser.add_argument("--save_interval", default=2000, type=int, help="Interval of saving model")
    parser.add_argument("--max_episode_length", default=400, type=int, help="Terminate earlier for one episode")
    parser.add_argument("--train_freq", default=8, type=int, help="Frequency for training")
    parser.add_argument("--train_interv", default=8, type=int, help="interval for training")
    parser.add_argument("--num-burn-in", default=1000, type=int, help="number of memory before train")

    parser.add_argument("-o", "--output", default="ilocus-v0", type=str, help="Directory to save data to")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--train", default=True, type=bool, help="Train/Evaluate, set True if train the model")
    parser.add_argument("--model_path", default="atari-v0", type=str, help="specify model path to evaluation")
    parser.add_argument("--max_grad", default=1.0, type=float, help="Parameter for huber loss")
    parser.add_argument("--log_dir", default="log", type=str, help="specify log folder to save evaluate result")
    parser.add_argument("--eval_num", default=100, type=int, help="number of evaluation to run")
    parser.add_argument("--save_freq", default=100000, type=int, help="model save frequency")

    # memory related args
    parser.add_argument("--buffer_size", default=40000, type=int, help="reply memory buffer size")
    parser.add_argument("--look_back_steps", default=1, type=int, help="how many previous pricing tables will be fed into RL")
    
    args = parser.parse_args()
    # print("\nParameters:")
    # for arg in vars(args):
    #     print(arg, getattr(args, arg))

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.mkdir(args.output)
    
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)
    os.mkdir(args.log_dir)
    # Initiating policy for both tasks (training and evaluating)
    policy = LinearDecayGreedyEpsilonPolicy(start_value=1, end_value=0.1, 
                                            num_steps=50000, num_actions=args.num_actions)

    if not args.train:
        '''Evaluate the model'''
        # check model path
        if args.model_path is '':
            print("Model path must be set when evaluate")
            exit(1)

        # specific log file to save result
        log_file = os.path.join(args.log_dir, args.network_name, str(args.model_num))
        model_dir = os.path.join(args.model_path, args.network_name, str(args.model_num))

        with tf.Session() as sess:
            with tf.device('/cpu:0'):
                print('Evaluationg model')
            # load model
            q_network = create_model(args.look_back_steps, args.map_shape, args.num_actions)
            sess.run(tf.global_variables_initializer())
            #
            # load weights into model
            q_network.load_weights(args.output + '/qnet.h5')

            driver_sim = Drivers()
            env = Environment(driver_sim=driver_sim)
            eval_driver_sim = Drivers()
            eval_env = Environment(driver_sim=eval_driver_sim)
            eval_memory = ReplayMemory(args.buffer_size, args.look_back_steps)
            eval_policy = LinearDecayGreedyEpsilonPolicy(start_value=1, end_value=0.1, 
                                            num_steps=1000000, num_actions=args.num_actions)


            q_network = create_model(args.look_back_steps, args.map_shape, args.num_actions)
            dqn_agent = DQNAgent(q_network=q_network, memory=None, policy=policy, gamma=args.gamma, 
                                target_update_freq=args.target_update_freq, num_burn_in=args.num_burn_in, 
                                train_freq=args.train_freq, batch_size=args.batch_size, train_interv=args.train_interv,
                                log_dir=args.log_dir)
            
            dqn_agent.evaluate(eval_env, eval_memory, eval_policy, 10, 50)
        exit(0)

    '''Train the model'''

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            print("created model")

            driver_sim = Drivers()
            env = Environment(driver_sim=driver_sim)
            print("set up environment")

            # # create output dir, meant to pop up error when dir exist to avoid over written
            # os.mkdir(args.output + "/" + args.network_name)

            memory = ReplayMemory(args.buffer_size, args.look_back_steps)
            q_network = create_model(args.look_back_steps, args.map_shape, args.num_actions)
            dqn_agent = DQNAgent(q_network=q_network, memory=memory, policy=policy, gamma=args.gamma, 
                                target_update_freq=args.target_update_freq, num_burn_in=args.num_burn_in, 
                                train_freq=args.train_freq, batch_size=args.batch_size, train_interv=args.train_interv,
                                log_dir=args.log_dir)
            print( "defined dqn agent")

            optimizer = Adam(learning_rate=args.alpha)
            q_network.compile(optimizer, mean_huber_loss)

            sess.run(tf.global_variables_initializer())

            print ("initializing environment")
            env.reset()

            print ("in fit")

            eval_driver_sim = Drivers()
            eval_env = Environment(driver_sim=eval_driver_sim)
            eval_memory = ReplayMemory(args.buffer_size, args.look_back_steps)
            eval_policy = LinearDecayGreedyEpsilonPolicy(start_value=1, end_value=0.1, 
                                            num_steps=1000000, num_actions=args.num_actions)
            dqn_agent.fit(env=env, num_iterations=args.num_iterations,
                          eval_env=eval_env, eval_policy=eval_policy, eval_memory=eval_memory,
                          output_dir=os.path.join(args.output),
                          max_episode_length=args.max_episode_length, save_interval=args.save_interval)

if __name__ == '__main__':
    main()