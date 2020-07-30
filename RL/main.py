#!/usr/bin/env python
"""Run iLOCuS with DQN."""
import sys

import argparse
import os
import shutil

import tensorflow as tf
import numpy as np

from environment import Environment
from policy import LinearDecayGreedyEpsilonPolicy
from agent import DQNAgent
from core import ReplayMemory
from driver_func_test import DriverSim
from model import create_model
from objectives import mean_huber_loss
from keras.optimizers import Adam, SGD

def main():  # noqa: D103
    parser = argparse.ArgumentParser(description="Run DQN on iLOCuS")
    parser.add_argument("--network_name", default="deep_q_network", type=str, help="Type of model to use")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--map_shape", default=(15, 15), type=tuple, help="map size")
    parser.add_argument("--num_actions", default=4, type=int, help="level of pricing")

    parser.add_argument("--gamma", default=0.8, type=float, help="Discount factor")
    parser.add_argument("--alpha", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration probability for epsilon-greedy")
    parser.add_argument("--target_update_freq", default=10000, type=int,
                        help="Frequency for copying weights to target network")
    parser.add_argument("--num_iterations", default=5000000, type=int,
                        help="Number of overal interactions to the environment")
    parser.add_argument("--max_episode_length", default=200000, type=int, help="Terminate earlier for one episode")
    parser.add_argument("--train_freq", default=4, type=int, help="Frequency for training")
    parser.add_argument("--num-burn-in", default=10000, type=int, help="number of memory before train")

    parser.add_argument("-o", "--output", default="ilocus-v0", type=str, help="Directory to save data to")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--train", default=True, action='store_false', help="Train/Evaluate, set True if train the model")
    parser.add_argument("--model_path", default="atari-v0", type=str, help="specify model path to evaluation")
    parser.add_argument("--max_grad", default=1.0, type=float, help="Parameter for huber loss")
    parser.add_argument("--log_dir", default="log", type=str, help="specify log folder to save evaluate result")
    parser.add_argument("--flip_coin", default=False, type=str,
                        help="specify whether or not choosing double q learning")
    parser.add_argument("--eval_num", default=100, type=int, help="number of evaluation to run")
    parser.add_argument("--save_freq", default=100000, type=int, help="model save frequency")

    # memory related args
    parser.add_argument("--buffer_size", default=100000, type=int, help="reply memory buffer size")
    parser.add_argument("--look_back_steps", default=4, type=int, help="how many previous pricing tables will be fed into RL")
    
    args = parser.parse_args()
    print("\nParameters:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Initiating policy for both tasks (training and evaluating)
    policy = LinearDecayGreedyEpsilonPolicy(args.epsilon, 0.1, 1000000, args.num_actions)

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
            # load model
            # with open(model_dir + ".json", 'r') as json_file:
            #     loaded_model_json = json_file.read()
            #     q_network_online = model_from_json(loaded_model_json)
            #     q_network_target = model_from_json(loaded_model_json)
            #
            # sess.run(tf.global_variables_initializer())
            #
            # # load weights into model
            # q_network_online.load_weights(model_dir + ".h5")
            # q_network_target.load_weights(model_dir + ".h5")

            driver_sim = DriverSim()
            env = Environment(driver_sim=driver_sim)

            memory = ReplayMemory(args.buffer_size, args.look_back_steps)
            q_network = create_model(args.look_back_steps, args.map_shape, args.num_actions)
            dqn_agent = DQNAgent(q_network=q_network, memory=memory, policy=policy, gamma=args.gamma, 
                                target_update_freq=args.target_update_freq, num_burn_in=args.num_burn_in, 
                                train_freq=args.train_freq, batch_size=args.batch_size)
        exit(0)

    '''Train the model'''

    with tf.Session() as sess:
        # with tf.device('/cpu:0'):
        print("created model")

        driver_sim = DriverSim()
        env = Environment(driver_sim=driver_sim)
        print("set up environment")

        # # create output dir, meant to pop up error when dir exist to avoid over written
        # os.mkdir(args.output + "/" + args.network_name)

        memory = ReplayMemory(args.buffer_size, args.look_back_steps)
        q_network = create_model(args.look_back_steps, args.map_shape, args.num_actions)
        dqn_agent = DQNAgent(q_network=q_network, memory=memory, policy=policy, gamma=args.gamma, 
                            target_update_freq=args.target_update_freq, num_burn_in=args.num_burn_in, 
                            train_freq=args.train_freq, batch_size=args.batch_size)
        print( "defined dqn agent")

        optimizer = Adam(learning_rate=args.alpha)
        q_network.compile(optimizer, mean_huber_loss)

        sess.run(tf.global_variables_initializer())

        print ("initializing environment")
        env.reset()

        print ("in fit")
        if os.path.exists(args.output):
            shutil.rmtree(args.output)
        os.mkdir(args.output)
        dqn_agent.fit(env=env, num_iterations=args.num_iterations,
                        output_dir=os.path.join(args.output),
                        max_episode_length=args.max_episode_length)

if __name__ == '__main__':
    main()