"""Main DQN agent."""

import keras
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
import pickle
import gc
import copy

PRINT_INTERV = 100
SIZE_R = 5
SIZE_C = 5


class DDPGAgent:
    def __init__(self,
                 actor,
                 critic,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 train_interv,
                 batch_size,
                 log_dir,
                 sess,
                 a_input,
                 c_input,
                 c_action,
                 lr):
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.train_interv = train_interv
        self.batch_size = batch_size
        self.reward_log = open(log_dir + 'reward', 'w')
        self.loss_log = open(log_dir + 'loss', 'w')
        self.a_input = a_input
        self.c_input = c_input
        self.sess = sess
        self.c_action = c_action
        self.lr = lr

        self.tau = 0.01  # soft replace target weight

        self.action_grads = tf.gradients(self.critic.output, self.c_action)
        self.action_gradient = tf.placeholder(tf.float32, [None, 25])
        self.params_grad = tf.gradients(self.actor.output, self.actor.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.actor.weights)
        self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def calc_q_values(self, state, q_net):
        q_value = q_net.predict_on_batch(state)
        return q_value

    def update_critic(self, target_critic, target_actor):
        x, x_next, other_infos, actions, state_actions = self.memory.sample(self.batch_size)
        actions = actions.reshape([actions.shape[0], SIZE_R * SIZE_C])
        y = self.calc_q_values([x, actions], self.critic)
        a_next = self.calc_q_values(x_next, target_actor)  # next action by actor
        # a_next = a_next.reshape([a_next.shape[0], SIZE_R, SIZE_C])
        # next_state_actions = np.zeros((128, 5, SIZE_R, SIZE_C))
        # next_state_actions[:, 0:4, :, :] = x_next
        # next_state_actions[:, 4, :, :] = a_next
        y_next = self.calc_q_values([x_next, a_next], target_critic)  # next q value by critic
        y_max = np.reshape(y_next, [self.batch_size, SIZE_R, SIZE_C])
        # Q learning update
        for _sample_index, (action, is_terminal, reward) in enumerate(other_infos):

            for rew_index, _reward in enumerate(reward):
                tmp_rew_index = np.unravel_index(rew_index, (SIZE_R, SIZE_C))

                if is_terminal:
                    y[_sample_index, rew_index] = _reward
                else:
                    y[_sample_index, rew_index] = _reward + self.gamma * y_max[
                        _sample_index, tmp_rew_index[0], tmp_rew_index[1]]
        train_loss = self.critic.train_on_batch([x, actions], y)
        return train_loss

    def update_actor(self, target_actor, target_critic, lr):
        x, x_next, other_infos, actions, state_actions = self.memory.sample(self.batch_size)

        action = self.calc_q_values(x, self.actor)  # y: action by actor
        # action = action.reshape([action.shape[0], SIZE_R, SIZE_C])
        # action_grads = tf.gradients(self.critic.output, self.c_action)
        a_grads = self.sess.run(self.action_grads, feed_dict={
            self.c_input: x,
            self.c_action: action
        })[0]

        # action_gradient = tf.placeholder(tf.float32, [None, 25])
        # a_grads = np.array(a_grads).reshape([len(a_grads[0]), SIZE_R*SIZE_C])
        # params_grad = tf.gradients(self.actor.output, self.actor.weights, self.action_gradient)
        # grads = zip(params_grad, self.actor.weights)
        # optimize = tf.train.AdamOptimizer(lr).apply_gradients(grads)
        self.sess.run(self.optimize, feed_dict={
            self.a_input: x,
            self.action_gradient: a_grads
        })

    def _update_actor_target(self, target_actor):  # soft replace
        actor_model_weights = self.actor.get_weights()
        actor_target_weights = target_actor.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i] * self.tau + actor_target_weights[i] * (1 - self.tau)
        target_actor.set_weights(actor_target_weights)
        return target_actor

    def _update_critic_target(self, target_critic):
        critic_model_weights = self.critic.get_weights()
        critic_target_weights = target_critic.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i] * self.tau + critic_target_weights[i] * (1 - self.tau)
        target_critic.set_weights(critic_target_weights)
        return target_critic

    def fit(self, env, output_add, eval_env, eval_memory, eval_policy, num_iterations, max_episode_length, lr):
        # Alaogrithm 1 from the reference paper
        # Initialize a target Q network as same as the online Q network
        a_config = self.actor.get_config()
        target_actor = Model.from_config(a_config)
        a_weights = self.actor.get_weights()
        target_actor.set_weights(a_weights)

        c_config = self.critic.get_config()
        target_critic = Model.from_config(c_config)
        c_weights = self.critic.get_weights()
        target_critic.set_weights(c_weights)

        # INITIALIZE counters and containers
        loss = []
        score = []
        Q_update_counter = 0
        targetQ_update_counter = 0
        evalQ_update_counter = 0
        episode_counter = 0
        episode_reward = []
        episode_len = []
        best_reward = 0
        kl_reward_list = []
        while True:
            if Q_update_counter > num_iterations:
                break
            gc.collect()
            # For every new episode, reset the environment and the preprocessor
            episode_counter += 1
            print("********  0 Begin the training episode: ", episode_counter, ", currently ", Q_update_counter,
                  " step  *******************")
            prev_state = env.reset()
            count_terminal = 0
            self.memory.append_state(prev_state)
            for t in range(max_episode_length):
                # Generate samples according to different policy
                if self.memory.current_size > self.num_burn_in:
                    fwd_states = self.memory.gen_forward_state()
                    fwd_res = self.calc_q_values(np.asarray(fwd_states), self.actor)
                    # action_map = self.policy.select_action(fwd_res, True)
                    action_map = fwd_res.reshape(5, 5)
                    # print(action_map.shape)
                else:
                    action_map = np.random.randint(self.policy.num_actions, size=(SIZE_R, SIZE_C))
                    # print(action_map.shape)

                # action_map = np.reshape(_action, (SIZE_R, SIZE_C))
                # Take 1 action
                # print(action_map)
                next_state, reward, is_terminal = env.step(action_map)
                # print("***training reward is...",np.sum(reward))
                if is_terminal == True:
                    count_terminal += 1
                    reward -= 50
                episode_reward.append(reward)
                # append other infor to replay memory (action, reward, t, is_terminal)
                self.memory.append_other(action_map, reward, t, is_terminal)
                Q_update_counter += 1
                if Q_update_counter == 1:
                    self.actor.save(output_add + '/anet-0of5.h5')
                    self.critic.save(output_add + '/cnet-0of5.h5')
                elif Q_update_counter == num_iterations // 5:
                    self.actor.save(output_add + '/anet-1of5.h5')
                    self.critic.save(output_add + '/cnet-0of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-1of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-1of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-1of5.p", "wb"))
                    pickle.dump(kl_reward_list, open(output_add + "/reward-best.p", "wb"))
                elif Q_update_counter == num_iterations // 5 * 2:
                    self.actor.save(output_add + '/anet-2of5.h5')
                    self.critic.save(output_add + '/cnet-0of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-2of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-2of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-2of5.p", "wb"))
                    pickle.dump(kl_reward_list, open(output_add + "/reward-best.p", "wb"))
                elif Q_update_counter == num_iterations // 5 * 3:
                    self.actor.save(output_add + '/anet-3of5.h5')
                    self.critic.save(output_add + '/cnet-0of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-3of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-3of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-3of5.p", "wb"))
                    pickle.dump(kl_reward_list, open(output_add + "/reward-best.p", "wb"))
                elif Q_update_counter == num_iterations // 5 * 4:
                    self.actor.save(output_add + '/anet-4of5.h5')
                    self.critic.save(output_add + '/cnet-0of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-4of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-4of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-4of5.p", "wb"))
                    pickle.dump(kl_reward_list, open(output_add + "/reward-best.p", "wb"))
                elif Q_update_counter == num_iterations:
                    self.actor.save(output_add + '/anet-5of5.h5')
                    self.critic.save(output_add + '/cnet-0of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-5of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-5of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-5of5.p", "wb"))
                    pickle.dump(kl_reward_list, open(output_add + "/reward-best.p", "wb"))

                # Update the Q net using minibatch from replay memory and update the target Q net
                if self.memory.current_size > self.num_burn_in:

                    # Update network

                    #  ####### update #####  #
                    if Q_update_counter % self.train_freq == 0:
                        # train actor and critic
                        tmp_value = []
                        tmp_actor_value = []
                        for _ in range(100):
                            evalQ_update_counter += 1
                            if (Q_update_counter % 50000 == 0) and (Q_update_counter < 200000):
                                K.set_value(self.critic.optimizer.lr, lr / 10)
                                K.set_value(self.actor.optimizer.lr, lr / 10)
                            tmp_value = [evalQ_update_counter, self.update_critic(target_critic, target_actor)]
                            self.update_actor(target_actor, target_critic, lr / 10)

                            # soft replace target tau=0.01
                            target_actor = self._update_actor_target(target_actor)
                            target_critic = self._update_critic_target(target_critic)
                        #  ####### update #####  #

                        if evalQ_update_counter % 500 == 0:
                            loss.append(tmp_value)
                            _eval_reward = self.evaluate(str(episode_counter), eval_env, eval_memory, eval_policy, 10,
                                                         50)
                            score.append([Q_update_counter, _eval_reward[0]])
                            kl_reward_list.append([Q_update_counter, _eval_reward[-1]])
                            print("1 The average total score for 10 episodes after ", evalQ_update_counter,
                                  " updates is ", score[-1])
                            print("2 The loss after ", evalQ_update_counter, " updates is: ", loss[-1])
                            print("3 The KL divergence after ", evalQ_update_counter, " updates is: ", _eval_reward[-1])
                            print(action_map)
                            if _eval_reward[-1] > best_reward:
                                self.actor.save(output_add + '/anet-5of5.h5')
                                self.critic.save(output_add + '/cnet-0of5.h5')

                                # Save the episode_len, loss, score into files
                                pickle.dump(episode_len, open(output_add + "/episode_best.p", "wb"))
                                pickle.dump(loss, open(output_add + "/loss-best.p", "wb"))
                                pickle.dump(score, open(output_add + "/score-best.p", "wb"))
                                pickle.dump(kl_reward_list, open(output_add + "/reward-best.p", "wb"))
                                best_reward = float(_eval_reward[-1])

                    targetQ_update_counter += 1

                    # Uate the target actor and critic every self.target_update_freq stepspd
                    # if targetQ_update_counter == self.target_update_freq:
                    #     targetQ_update_counter = 0
                    #     weights = self.actor.get_weights()
                    #     target_actor.set_weights(weights)
                    #
                    #     weights = self.critic.get_weights()
                    #     target_critic.set_weights(weights)

                if is_terminal:
                    break

                if t < max_episode_length - 1:
                    prev_state = copy.deepcopy(next_state)
                    self.memory.append_state(prev_state)

            episode_len.append(t)

    def _tmp_compute_reward(self, state):
        objective = np.ones((SIZE_R, SIZE_C))
        objective /= np.sum(objective)
        state = np.copy(state) + 1e-7

        # normalize
        state /= np.sum(state)

        # KL divergence
        # return np.sum((np.where(state != 0, state*np.log(state / objective), 0)))
        return np.sum(np.sum(state * np.log(state / objective)))

    def evaluate(self, env_name, eval_env, eval_memory, eval_policy, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.

        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.
        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.
        You can also call the render function here if you want to
        visually inspect your policy.
        """
        print(' ********** Evaluating *******')
        mean_reward = 0
        mean_cost = 0
        kl_reward = 0
        pricing_fp = open('log/pricing_' + env_name + '.txt', 'w')
        for _ in range(num_episodes):
            total_reward = 0
            eval_memory.clear()
            prev_state = eval_env.reset()
            eval_memory.append_state(prev_state)
            for t in range(max_episode_length):
                # Generate samples according to different policy
                fwd_states = eval_memory.gen_forward_state()
                fwd_res = self.calc_q_values(np.asarray(fwd_states), self.actor)
                # action_map = eval_policy.select_action(fwd_res, False)
                action_map = fwd_res.reshape(5, 5)
                # action_map = np.reshape(_action, (SIZE_R, SIZE_C))
                mean_cost += np.sum(action_map)

                # Take 1 action
                next_state, reward, is_terminal = eval_env.step(action_map)

                pricing_fp.write('state\n' + np.array2string(prev_state) + '\n')
                pricing_fp.write('price\n' + np.array2string(action_map) + '\n')

                total_reward += np.sum(reward)
                eval_memory.append_other(action_map, reward, t, is_terminal)
                prev_state = copy.deepcopy(next_state)
                eval_memory.append_state(prev_state)
            kl_reward += self._tmp_compute_reward(next_state[1, :, :])

            mean_reward += total_reward
        print("evaluating action map is ", action_map)
        print("evaluating state is ", next_state[1, :, :])
        pricing_fp.close()
        print(' ********** average cost', mean_cost / num_episodes / max_episode_length, "*********")
        return mean_reward / num_episodes / max_episode_length, kl_reward / num_episodes
