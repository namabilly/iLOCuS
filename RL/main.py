#!/usr/bin/env python
"""Run iLOCuS with DQN."""
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import argparse
import os

import tensorflow as tf

from environment import Environment
from policy import *

def main():  # noqa: D103
    parser = argparse.ArgumentParser(description="Run DQN on iLOCuS")
    parser.add_argument("--network_name", default="deep_q_network_duel", type=str, help="Type of model to use")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--n_samples", default=32, type=int, help="The number of sentences to be sampled")

    parser.add_argument("--gamma", default=0.8, type=float, help="Discount factor")
    parser.add_argument("--alpha", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration probability for epsilon-greedy")
    parser.add_argument("--target_update_freq", default=1, type=int,
                        help="Frequency for copying weights to target network")
    parser.add_argument("--num_iterations", default=5000000, type=int,
                        help="Number of overal interactions to the environment")
    parser.add_argument("--max_episode_length", default=200000, type=int, help="Terminate earlier for one episode")
    parser.add_argument("--train_freq", default=4, type=int, help="Frequency for training")

    parser.add_argument("-o", "--output", default="ilocus-v0", type=str, help="Directory to save data to")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--train", default=True, type=bool, help="Train/Evaluate, set True if train the model")
    parser.add_argument("--model_path", default="atari-v0", type=str, help="specify model path to evaluation")
    parser.add_argument("--max_grad", default=1.0, type=float, help="Parameter for huber loss")
    parser.add_argument("--model_num", default=5000000, type=int, help="specify saved model number during train")
    parser.add_argument("--log_dir", default="log", type=str, help="specify log folder to save evaluate result")
    parser.add_argument("--flip_coin", default=False, type=str,
                        help="specify whether or not choosing double q learning")
    parser.add_argument("--eval_num", default=100, type=int, help="number of evaluation to run")
    parser.add_argument("--save_freq", default=100000, type=int, help="model save frequency")

    parser.add_argument("--lstm_units", default=64, type=int, help="number of units in LSTM model")
    parser.add_argument("--data_path", default="data", type=str, help="number of units in LSTM model")
    parser.add_argument("--h_units_state", default=64, type=int, help="dqn hidden unit")
    parser.add_argument("--h_units_classifier", default=64, type=int, help="classifier hidden unit")
    parser.add_argument("--num_epoch", default=2, type=int, help="number of epochs to train the classifier")

    args = parser.parse_args()
    print("\nParameters:")
    for arg in vars(args):
        print arg, getattr(args, arg)
    print("")

    # define model object

    # dict = Dictionary(entity_path="data",
    #                   embed_path="/media/hongbao/Study/Sailing Lab/Data/google_word2vec_vocab",
    #                   word2vec_model_path="/media/hongbao/Study/Sailing Lab/Data/GoogleNews-vectors-negative300.bin.gz")

    # Initiating policy for both tasks (training and evaluating)
    policy = LinearDecayGreedyEpsilonPolicy(args.epsilon, 0, 1000000)

    if not args.train:
        '''Evaluate the model'''
        # check model path
        if args.model_path is '':
            print "Model path must be set when evaluate"
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
            model = Model(args=args, num_actions=4)

            env = Environment(sess=sess,
                              # dict=dict,
                              acc_threshold=0.4,
                              add_threshold=0.8,
                              model=model,
                              nlp_server_args=args)

            dqn_agent = DQNAgent(q_network=model, memory=None, policy=policy, num_actions=4,
                                 gamma=args.gamma, target_update_freq=args.target_update_freq,
                                 network_name=args.network_name, max_grad=args.max_grad, sess=sess)

            dqn_agent.evaluate(env, log_file, args.eval_num)
        exit(0)

    '''Train the model'''

    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            model = Model(args=args, num_actions=4)
            print "created model"

            env = Environment(sess=sess,
                              model=model,
                              objective=np.ones((args.state_size,1))/args.state_size,
                              beta=args.gamma)
            print "set up environment"

            # # create output dir, meant to pop up error when dir exist to avoid over written
            # os.mkdir(args.output + "/" + args.network_name)

            dqn_agent = DQNAgent(q_network=model, memory=None, policy=policy, num_actions=4,
                                 gamma=args.gamma, target_update_freq=args.target_update_freq,
                                 network_name=args.network_name, max_grad=args.max_grad, sess=sess)
            print "defined dqn agent"

            optimizer = tf.train.AdamOptimizer(learning_rate=args.alpha)
            dqn_agent.compile(optimizer, mean_huber_loss)

            sess.run(tf.global_variables_initializer())

            # pre-train classifier with gold sentences
            print "initializing environment"
            env.initialize_environment(args.num_epoch)
            print "in fit"
            dqn_agent.fit(env=env, num_iterations=args.num_iterations,
                          output_folder=os.path.join(args.output, args.network_name, str(args.save_freq)),
                          max_episode_length=args.max_episode_length)

if __name__ == '__main__':
    main()