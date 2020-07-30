"""Main DQN agent."""

import keras
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


class DQNAgent:
    """Class implementing DQN.
    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgent. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.
    Feel free to change the functions and function parameters that the
    class provides.
    We have provided docstrings to go along with our suggested API.
    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """

    def __init__(self,
                 q_network,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 train_interv,
                 batch_size,
                 log_dir):
        self.q_network = q_network
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.train_interv = train_interv
        self.batch_size = batch_size
        self.reward_log = open(log_dir + '/reward','w')
        self.loss_log = open(log_dir + '/loss','w')

    def calc_q_values(self, state, q_net):
        """Given a state (or batch of states) calculate the Q-values.
        Basically run your network on these states.
        Return
        ------
        Q-values for the state(s)
        """
        q_value = q_net.predict_on_batch(state)
        return q_value

    def update_policy(self, target_q):
        """Update your policy.
        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.
        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.
        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """

        # Get a mini batch of samples (only index)
        x, x_next, other_infos = self.memory.sample(self.batch_size)

        y = self.calc_q_values(x, self.q_network)  # reserve the order in mini_batch
        # print(y)
        # print(x)
        y_next = self.calc_q_values(x_next, target_q)  # reserve the order in mini_batch
        # print(x_next)
        tmp_y_next = np.reshape(y_next, [self.batch_size, 10, SIZE_R, SIZE_C])
        y_max = np.reshape(np.amax(tmp_y_next, axis=1), [self.batch_size, SIZE_R, SIZE_C])
        # Q learning update
        for _sample_index, (action, is_terminal, reward) in enumerate(other_infos):
            for rew_index, _reward in enumerate(reward):
                tmp_rew_index = np.unravel_index(rew_index, (SIZE_R, SIZE_C))
                tmp_index = np.ravel_multi_index(((action[tmp_rew_index[0], tmp_rew_index[1]],)+ tmp_rew_index), (10, SIZE_R, SIZE_C))
                if is_terminal:
                    y[_sample_index, tmp_index] = _reward
                else:
                    # print(y_max[_sample_index, tmp_rew_index[0], tmp_rew_index[1]])
                    # print(_reward)
                    y[_sample_index, tmp_index] = _reward + self.gamma * y_max[_sample_index, tmp_rew_index[0], tmp_rew_index[1]]
        # print(x[:,-1,0,0])
        train_loss = self.q_network.train_on_batch(x, y)
        return train_loss

    def fit(self, env, output_add, eval_env, eval_memory, eval_policy, num_iterations, max_episode_length, lr):
        """Fit your model to the provided environment.
        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.
        You should probably also periodically save your network
        weights and any other useful info.
        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.
        """
        # Alaogrithm 1 from the reference paper
        # Initialize a target Q network as same as the online Q network
        config = self.q_network.get_config()
        target_q = Model.from_config(config)
        weights = self.q_network.get_weights()
        target_q.set_weights(weights)

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
            # learning rate decay
            # lr = self.q_network.optimizer.lr
            # if episode_counter == 2 or episode_counter == 40000 or episode_counter == 60000:
            #     self.q_network.optimizer.lr.set_value(lr * 0.1)
            #     print("@@@@@@@@@@@@@@@@")
            #     print(self.q_network.optimizer.lr)
            #     lrs = LearningRateScheduler(my_learning_rate)
            print("********  0 Begin the training episode: ", episode_counter, ", currently ", Q_update_counter,
            " step  *******************")
            prev_state = env.reset()
            # print(prev_state[1,:,:])
            count_terminal = 0
            self.memory.append_state(prev_state)
            for t in range(max_episode_length):
                # Generate samples according to different policy
                if self.memory.current_size > self.num_burn_in:
                    fwd_states = self.memory.gen_forward_state()
                    fwd_res = self.calc_q_values(np.asarray(fwd_states), self.q_network)
                    action_map = self.policy.select_action(fwd_res, True)
                    # print(action_map.shape)
                else:
                    action_map = np.random.randint(self.policy.num_actions, size=(SIZE_R,  SIZE_C))
                    # print(action_map.shape)

                # action_map = np.reshape(_action, (SIZE_R, SIZE_C))
                # Take 1 action
                # print(action_map)
                next_state, reward, is_terminal = env.step(action_map)
                # print("***training reward is...",reward)
                # print(prev_state[1,:,:])
                # print(next_state[1,:,:])
                if is_terminal == True:
                    count_terminal += 1
                    reward = -50*np.ones(SIZE_R*SIZE_C)
                episode_reward.append(reward)
                # append other infor to replay memory (action, reward, t, is_terminal)
                self.memory.append_other(action_map, reward, t, is_terminal)
                Q_update_counter += 1
                if Q_update_counter == 1:
                    self.q_network.save(output_add + '/qnet-0of5.h5')
                elif Q_update_counter == num_iterations // 5:
                    self.q_network.save(output_add + '/qnet-1of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-1of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-1of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-1of5.p", "wb"))
                    pickle.dump(kl_reward_list, open(output_add + "/reward-1of5.p", "wb"))
                elif Q_update_counter == num_iterations // 5 * 2:
                    self.q_network.save(output_add + '/qnet-2of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-2of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-2of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-2of5.p", "wb"))
                    pickle.dump(kl_reward_list, open(output_add + "/reward-2of5.p", "wb"))
                elif Q_update_counter == num_iterations // 5 * 3:
                    self.q_network.save(output_add + '/qnet-3of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-3of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-3of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-3of5.p", "wb"))
                    pickle.dump(kl_reward_list, open(output_add + "/reward-3of5.p", "wb"))
                elif Q_update_counter == num_iterations // 5 * 4:
                    self.q_network.save(output_add + '/qnet-4of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-4of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-4of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-4of5.p", "wb"))
                    pickle.dump(kl_reward_list, open(output_add + "/reward-4of5.p", "wb"))
                elif Q_update_counter == num_iterations:
                    self.q_network.save(output_add + '/qnet-5of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-5of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-5of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-5of5.p", "wb"))
                    pickle.dump(kl_reward_list, open(output_add + "/reward-5of5.p", "wb"))

                # Update the Q net using minibatch from replay memory and update the target Q net
                if self.memory.current_size > self.num_burn_in :
                    # print(self.memory.current_size)
                    # Update the Q network every self.train_freq steps
                    if Q_update_counter % self.train_freq == 0:
                        # print(evalQ_update_counter)
                        tmp_value = []
                        # for _ in range(1):
                        evalQ_update_counter += 1
                        if (Q_update_counter == 6e+6) :
                            K.set_value(self.q_network.optimizer.lr, lr / 10)
                        # if (Q_update_counter == 2e+5) :
                        #     K.set_value(self.q_network.optimizer.lr, lr / 2)
                        tmp_value = [evalQ_update_counter, self.update_policy(target_q)]
                        # print("loss decreses")
                        # print(tmp_value)
                        # print(evalQ_update_counter)
                        # evaluate_counter += 1
                        # if evaluate_counter % 20000 == 0:
                        #     # if evaluate_counter % 100 == 0:
                        if evalQ_update_counter % 50 == 0:
                            loss.append(tmp_value)
                            _eval_reward = self.evaluate(str(episode_counter), eval_env, eval_memory, eval_policy, 10, 5)
                            score.append([Q_update_counter, _eval_reward[0]])
                            kl_reward_list.append([Q_update_counter, _eval_reward[-1]])
                            print("1 The average total score for 10 episodes after ", evalQ_update_counter, " updates is ", score[-1])
                            print("2 The loss after ", evalQ_update_counter, " updates is: ", loss[-1])
                            print("3 The KL divergence after ", evalQ_update_counter, " updates is: ", _eval_reward[-1])
                            print(action_map)
                            if _eval_reward[-1] > best_reward:
                                self.q_network.save(output_add + '/qnet-best.h5')
                                # Save the episode_len, loss, score into files
                                pickle.dump(episode_len, open(output_add + "/episode_best.p", "wb"))
                                pickle.dump(loss, open(output_add + "/loss-best.p", "wb"))
                                pickle.dump(score, open(output_add + "/score-best.p", "wb"))
                                pickle.dump(kl_reward_list, open(output_add + "/reward-best.p", "wb"))
                                best_reward = float(_eval_reward[-1])
                        # Update the target Q network every self.target_update_freq steps
                        targetQ_update_counter += 1
                        print(targetQ_update_counter)
                        if targetQ_update_counter == self.target_update_freq:
                            targetQ_update_counter = 0
                            weights = self.q_network.get_weights()
                            target_q.set_weights(weights)

                    # if evalQ_update_counter % PRINT_INTERV == 0:
                    #     avg_loss = sum(item[1] for item in episode_loss[-PRINT_INTERV:])/PRINT_INTERV
                    #     print('Update {cnt} times, loss {loss}'.format(cnt=evalQ_update_counter,loss=avg_loss))
                    #     with open('log/loss', 'a') as log_loss:
                    #         log_loss.write(str(avg_loss)+'\n')

                if is_terminal:
                    break

                if t < max_episode_length - 1:
                    prev_state = copy.deepcopy(next_state)
                    self.memory.append_state(prev_state)

                # evaluate
                # if (episode_counter + 1) % 5 == 0:
                #     mean_reward = self.evaluate(str(episode_counter), eval_env, eval_memory, eval_policy, 10, 50)
                #     self.reward_log.write(str(mean_reward) + '\n')
                #     with open('log/reward', 'a') as log_reward:
                #         log_reward.write(str(mean_reward) + '\n')
                #     print(' ********** episode {cnt} : average reward {rwd}'.format(cnt=episode_counter, rwd=mean_reward))
            episode_len.append(t)

    # def load_weights(self, filepath):
    #     self.model.load_weights(filepath)
    #     self.update_target_model_hard()
    #
    # def save_weights(self, filepath, overwrite=False):
    #     self.model.save_weights(filepath, overwrite=overwrite)
    #
    # def reset_states(self):
    #     self.recent_action = None
    #     self.recent_observation = None
    #     if self.compiled:
    #         self.model.reset_states()
    #         self.target_model.reset_states()
    # def update_target_model_hard(self):
    #     self.target_model.set_weights(self.model.get_weights())
    def _tmp_compute_reward(self, state):
        objective = np.ones((SIZE_R, SIZE_C))
        objective /= np.sum(objective)
        state = np.copy(state) + 1e-7

        # normalize
        state /= np.sum(state)

        # KL divergence
        # return np.sum((np.where(state != 0, state*np.log(state / objective), 0)))
        # return np.sum(np.sum(state * np.log(state / objective)))
        return np.sum(np.sum(objective * np.log(objective / state)))

    def evaluate(self, env_name, eval_env, eval_memory, eval_policy, num_episodes, max_episode_length=5):
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
        pricing_fp = open('log/pricing_'+env_name+'.txt','w')
        seed_ = 2*np.arange(0,num_episodes, 1)
        for tmp_ in range(num_episodes):
            total_reward = 0
            eval_memory.clear()
            prev_state = eval_env.reset(seed=seed_[tmp_])
            print("starting state is ", prev_state[1, :, :])
            eval_memory.append_state(prev_state)
            for t in range(max_episode_length):
                # Generate samples according to different policy
                fwd_states = eval_memory.gen_forward_state()
                fwd_res = self.calc_q_values(np.asarray(fwd_states), self.q_network)
                action_map = eval_policy.select_action(fwd_res, False)

                # action_map = np.reshape(_action, (SIZE_R, SIZE_C))
                mean_cost += np.sum(action_map)

                # Take 1 action
                next_state, reward, is_terminal = eval_env.step(action_map)
                pricing_fp.write('state\n'+np.array2string(prev_state)+'\n')
                pricing_fp.write('price\n'+np.array2string(action_map)+'\n')

                total_reward += np.mean(reward)
                eval_memory.append_other(action_map, reward, t, is_terminal)
                prev_state = copy.deepcopy(next_state)
                eval_memory.append_state(prev_state)
            kl_reward += self._tmp_compute_reward(next_state[1,:,:])

            mean_reward += total_reward
            print("evaluating action map is ", action_map)
            print("evaluating state is ", next_state[1,:,:])
        pricing_fp.close()
        print(' ********** average cost',mean_cost/num_episodes/max_episode_length, "*********")
        return mean_reward / num_episodes/max_episode_length, kl_reward/num_episodes

