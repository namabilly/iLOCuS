"""Core classes."""

from numpy import random
import numpy as np

SIZE_R = 5
SIZE_C = 5

class Sample:
    """Represents a reinforcement learning sample.
    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.
    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.
    """
    def __init__(self, state, action, reward, timestamp, is_terminal):
        self.state = np.copy(state)  # (m, d, n, n0)
        self.action = np.copy(action) # pricing table, (15, 15)
        self.reward = reward
        self.timestamp = timestamp
        self.is_terminal = is_terminal

def gen_map(action):
    res_map = np.zeros((SIZE_R, SIZE_C))
    res_map[action//SIZE_C, action%SIZE_C] = 1
    return res_map

class ReplayMemory:
    """Interface for replay memories.
    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.
    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.
    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw samples saved in your memory).
    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.
    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size, look_back_steps):
        """Setup memory.
        You should specify the maximum size of the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.
        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.buffer_size = max_size
        self.current_size = 0
        self.buffer = [None for _ in range(max_size)]
        self.index = -1 # track the index where the next sample should be stored
        self.look_back_steps = look_back_steps

    def append_state(self, state):
        _sample = Sample(state, None, 0, -1, True)
        self.index = (self.index + 1) % self.buffer_size
        # Store the frame into replay memory
        self.buffer[self.index] = _sample
        # Update the current_size and the index for next storage
        self.current_size = max(self.current_size, self.index + 1)

    def append_other(self, action, reward, timestamp, is_terminal):
        # Store the other info into replay memory
        self.buffer[self.index].action = np.copy(action)
        self.buffer[self.index].reward = reward
        self.buffer[self.index].timestamp = timestamp
        self.buffer[self.index].is_terminal = is_terminal

    def sample(self, batch_size):
        # sample a minibatch of index
        indexes = random.choice(self.current_size * SIZE_R * SIZE_C, batch_size, replace=False)
        x = []
        x_next = []  # For next state
        other_infos = []
        for _index in indexes:
            sample_index = _index // (SIZE_R * SIZE_C)
            location = _index % (SIZE_R * SIZE_C)
            row, col = location // SIZE_C, location % SIZE_C
            x.append(self.stacked_retrieve(sample_index, location))
            other_infos.append((self.buffer[sample_index].action[row, col],
                                self.buffer[sample_index].is_terminal,
                                self.buffer[sample_index].reward[row, col],
                                ))
            _next_index = (sample_index + 1) % self.current_size
            # print(_next_index)
            x_next.append(self.stacked_retrieve(_next_index, location))

        x = np.asarray(x)
        x_next = np.asarray(x_next)
        # print(x[0,0], x_next[0,0])
        return x, x_next, other_infos
    
    def gen_forward_state(self):
        forward_states = [self.stacked_retrieve(self.index, i) for i in range(SIZE_R * SIZE_C)]
        return np.stack(forward_states)
        # forward_states shape: (225, 5 + self.look_back_steps, 15, 15)

    def stacked_retrieve(self, sample_index, location):
        # m, d, n, n0, location, p0-pt
        stacked_state = np.zeros((4 + self.look_back_steps, SIZE_R, SIZE_C))
        stacked_state[0:3,:,:] = self.buffer[sample_index].state
        stacked_state[3,:,:] = gen_map(location)
        timestamp = self.buffer[sample_index].timestamp
        for t in range(1, min(timestamp, self.look_back_steps) + 1):
            local_index = (sample_index - t) % self.buffer_size
            # print(sample_index, timestamp, local_index)
            stacked_state[- t] = self.buffer[local_index].action
        
        return stacked_state

    def clear(self):
        self.current_size = 0
        self.buffer = [None for _ in range(self.buffer_size)]
        self.index = -1 # track the index where the next sample should be stored
