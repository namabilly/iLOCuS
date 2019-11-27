"""Main DQN agent."""

import keras
from keras.models import Model
from keras.optimizers import Adam

from objectives import mean_huber_loss

import numpy as np
import pickle

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
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size):
        self.q_network = q_network
        self.preprocessor = preprocessor
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
        y_next = self.calc_q_values(x_next, target_q)  # reserve the order in mini_batch
        y_max = np.amax(y_next, axis=1)

        # Q learning update
        for _sample_index, (action, is_terminal, reward) in enumerate(other_infos):
            if is_terminal:
                y[_sample_index, action] = reward
            else:
                y[_sample_index, action] = reward + self.gamma * y_max[_sample_index]

        train_loss = self.q_network.train_on_batch(x, y)
        return train_loss

    def fit(self, env, env_name, output_add, num_iterations, max_episode_length=None):
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
        episode_len = []
        Q_update_counter = 0
        targetQ_update_counter = 0
        evaluate_counter = 0
        episode_counter = 0
        while True:
            if Q_update_counter > num_iterations:
                break
            # For every new episode, reset the environment and the preprocessor
            episode_counter += 1
            # print("********  0 Begin the training episode: ", episode_counter, ", currently ", Q_update_counter,
            # " step  *******************")
            initial_state = env.reset()
            for t in range(1, max_episode_length):
                # Generate samples according to different policy
                if self.memory.current_size > self.num_burn_in:
                    if self.memory.index == 0:
                        _state = self.stack_frames(self.memory.index - 1, t)
                    else:
                        _state = self.stack_frames(self.memory.current_size - 1, t)
                    _tmp = self.calc_q_values(np.asarray([_state, ]), self.q_network)
                    _action = self.policy.select_action(_tmp[0], True)
                else:
                    _action = np.random.randint(0, self.policy.epsilon_greedy_policy.num_actions)

                # Take 1 action
                next_state, reward, is_teminal = env.step(_action)

                # Process the raw reward
                # reward = self.preprocessor.process_reward(reward)
                if current_lives > env.env.ale.lives():
                    reward -= 50
                elif current_lives < env.env.ale.lives():
                    reward += 50
                current_lives = env.env.ale.lives()

                # append other infor to replay memory (action, reward, t, is_terminal)
                self.memory.append_other(_action, reward, t, is_terminal)

                # Save the trained Q-net at 5 check points
                Q_update_counter += 1
                if Q_update_counter == 1:
                    self.q_network.save(output_add + '/qnet-0of5.h5')
                elif Q_update_counter == num_iterations // 5:
                    self.q_network.save(output_add + '/qnet-1of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-1of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-1of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-1of5.p", "wb"))
                elif Q_update_counter == num_iterations // 5 * 2:
                    self.q_network.save(output_add + '/qnet-2of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-2of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-2of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-2of5.p", "wb"))
                elif Q_update_counter == num_iterations // 5 * 3:
                    self.q_network.save(output_add + '/qnet-3of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-3of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-3of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-3of5.p", "wb"))
                elif Q_update_counter == num_iterations // 5 * 4:
                    self.q_network.save(output_add + '/qnet-4of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-4of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-4of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-4of5.p", "wb"))
                elif Q_update_counter == num_iterations:
                    self.q_network.save(output_add + '/qnet-5of5.h5')
                    # Save the episode_len, loss, score into files
                    pickle.dump(episode_len, open(output_add + "/episode_length-5of5.p", "wb"))
                    pickle.dump(loss, open(output_add + "/loss-5of5.p", "wb"))
                    pickle.dump(score, open(output_add + "/score-5of5.p", "wb"))

                # Update the Q net using minibatch from replay memory and update the target Q net
                if self.memory.current_size > self.num_burn_in:
                    # Update the Q network every self.train_freq steps
                    if Q_update_counter % self.train_freq == 0:
                        tmp_value = [Q_update_counter, self.update_policy(target_q)]
                        evaluate_counter += 1
                        if evaluate_counter % 20000 == 0:
                            # if evaluate_counter % 100 == 0:
                            loss.append(tmp_value)
                            score.append([Q_update_counter, self.evaluate(env_name, 10, max_episode_length)])
                            # print("1 The average total score for 10 episodes after ", evaluate_counter, " updates is ", score[-1])
                            # print("2 The loss after ", evaluate_counter, " updates is: ", loss[-1])
                    # Update the target Q network every self.target_update_freq steps
                    targetQ_update_counter += 1
                    if targetQ_update_counter == self.target_update_freq:
                        targetQ_update_counter = 0
                        weights = self.q_network.get_weights()
                        target_q.set_weights(weights)

                if is_terminal:
                    break

                # if it is not terminal, process the frame and append to replay memory
                prev_frame = self.preprocessor.process_frame_for_memory(next_frame)
                self.memory.append_frame(prev_frame)
            # Store the episode length
            episode_len.append(t)

    def evaluate(self, env_name, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.
        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.
        You can also call the render function here if you want to
        visually inspect your policy.
        """

        # Run the policy for 20 episodes and calculate the mean total reward (final score of game)
        env = gym.make(env_name)
        mean_reward = 0
        for episode in range(num_episodes):
            initial_frame = env.reset()
            state = np.zeros((4, 84, 84), dtype=np.float32)
            # Preprocess the state      
            prev_frame = self.preprocessor.process_frame_for_memory(initial_frame).astype(dtype=np.float32)
            prev_frame = prev_frame / 255
            state[:-1] = state[1:]
            state[-1] = np.copy(prev_frame)
            # Initialize the total reward and then begin an episode
            total_reward = 0
            for t in range(max_episode_length):
                _tmp = self.calc_q_values(np.asarray([state, ]), self.q_network)
                _action = self.policy.select_action(_tmp[0], False)
                next_frame, reward, is_terminal, debug_info = env.step(_action)
                # Use the original reward to calculate total reward
                total_reward += reward
                if is_terminal:
                    break
                # Update the state
                prev_frame = self.preprocessor.process_frame_for_memory(next_frame).astype(dtype=np.float32)
                prev_frame = prev_frame / 255
                state[:-1] = state[1:]
                state[-1] = np.copy(prev_frame)
            mean_reward += total_reward
        return mean_reward / num_episodes
