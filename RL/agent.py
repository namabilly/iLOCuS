"""Main DQN agent."""

import keras
from keras.models import Model
from keras.optimizers import Adam

from objectives import mean_huber_loss

import numpy as np
import pickle
import gc

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
                 batch_size):
        self.q_network = q_network
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size

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
        y_max = np.amax(y_next, axis=1)
        # Q learning update
        for _sample_index, (action, is_terminal, reward) in enumerate(other_infos):
            if is_terminal:
                y[_sample_index, action] = reward
            else:
                y[_sample_index, action] = reward + self.gamma * y_max[_sample_index]
        # print(y)
        train_loss = self.q_network.train_on_batch(x, y)
        return train_loss

    def fit(self, env, output_dir, num_iterations, max_episode_length):
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
        while True:
            if Q_update_counter > num_iterations:
                break
            gc.collect()
            # For every new episode, reset the environment and the preprocessor
            episode_counter += 1
            episode_reward = []
            # print("********  0 Begin the training episode: ", episode_counter, ", currently ", Q_update_counter,
            # " step  *******************")
            prev_state = env.reset()
            for t in range(1, max_episode_length):
                # Generate samples according to different policy
                if self.memory.current_size > self.num_burn_in:
                    fwd_states = self.memory.gen_forward_state()
                    fwd_res = self.calc_q_values(np.asarray(fwd_states), self.q_network)
                    _action = self.policy.select_action(fwd_res, True)
                else:
                    _action = np.random.randint(self.policy.num_actions, size=(225, 1))

                action_map = np.reshape(_action, (15, 15))
                # Take 1 action
                next_state, reward, is_terminal = env.step(action_map)
                if t == max_episode_length - 1:
                  is_terminal = True
                episode_reward.append(reward)

                # append other infor to replay memory (action, reward, t, is_terminal)
                self.memory.append(prev_state, action_map, reward, t, is_terminal)

                # Save the trained Q-net at 5 check points
                Q_update_counter += 1
                if Q_update_counter % (num_iterations // 5) == 0:
                    cp_name = Q_update_counter // (num_iterations // 5)
                    self.q_network.save(output_dir + '/qnet-{cp}.h5'.format(cp=cp_name))
                    # Save the episode_len, loss, score into files
                    pickle.dump(loss, open(output_dir + "/loss-{cp}.p".format(cp=cp_name), "wb"))
                    pickle.dump(score, open(output_dir + "/score-{cp}.p".format(cp=cp_name), "wb"))

                # Update the Q net using minibatch from replay memory and update the target Q net
                if self.memory.current_size > self.num_burn_in:
                    # Update the Q network every self.train_freq steps
                    for _ in range(self.train_freq):
                        evalQ_update_counter += 1
                        tmp_value = [evalQ_update_counter, self.update_policy(target_q)]
                        print('Update {cnt} times, loss {loss}'.format(cnt=tmp_value[0],loss=tmp_value[1]))
                        # print('action', action_map)
                        # evaluate_counter += 1
                        # if evaluate_counter % 20000 == 0:
                        #     # if evaluate_counter % 100 == 0:
                        #     loss.append(tmp_value)
                            # score.append([Q_update_counter, self.evaluate(env_name, 10, max_episode_length)])
                            # print("1 The average total score for 10 episodes after ", evaluate_counter, " updates is ", score[-1])
                            # print("2 The loss after ", evaluate_counter, " updates is: ", loss[-1])
                        # Update the target Q network every self.target_update_freq steps
                        if evalQ_update_counter % self.target_update_freq == 0:
                            weights = self.q_network.get_weights()
                            target_q.set_weights(weights)

                if is_terminal:
                    break

                prev_state = next_state
            mean_reward = sum(episode_reward) / len(episode_reward)
            print('*********** episode {cnt} : average reward {rwd}'.format(cnt=episode_counter, rwd=mean_reward))

    # def evaluate(self, env_name, num_episodes, max_episode_length=None):
    #     """Test your agent with a provided environment.
        
    #     You shouldn't update your network parameters here. Also if you
    #     have any layers that vary in behavior between train/test time
    #     (such as dropout or batch norm), you should set them to test.
    #     Basically run your policy on the environment and collect stats
    #     like cumulative reward, average episode length, etc.
    #     You can also call the render function here if you want to
    #     visually inspect your policy.
    #     """

    #     # Run the policy for 20 episodes and calculate the mean total reward (final score of game)
    #     env = gym.make(env_name)
    #     mean_reward = 0
    #     for episode in range(num_episodes):
    #         initial_frame = env.reset()
    #         state = np.zeros((4, 84, 84), dtype=np.float32)
    #         # Preprocess the state      
    #         prev_frame = self.preprocessor.process_frame_for_memory(initial_frame).astype(dtype=np.float32)
    #         prev_frame = prev_frame / 255
    #         state[:-1] = state[1:]
    #         state[-1] = np.copy(prev_frame)
    #         # Initialize the total reward and then begin an episode
    #         total_reward = 0
    #         for t in range(max_episode_length):
    #             _tmp = self.calc_q_values(np.asarray([state, ]), self.q_network)
    #             _action = self.policy.select_action(_tmp[0], False)
    #             next_frame, reward, is_terminal, debug_info = env.step(_action)
    #             # Use the original reward to calculate total reward
    #             total_reward += reward
    #             if is_terminal:
    #                 break
    #             # Update the state
    #             prev_frame = self.preprocessor.process_frame_for_memory(next_frame).astype(dtype=np.float32)
    #             prev_frame = prev_frame / 255
    #             state[:-1] = state[1:]
    #             state[-1] = np.copy(prev_frame)
    #         mean_reward += total_reward
    #     return mean_reward / num_episodes
