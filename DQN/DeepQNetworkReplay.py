import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import os

from DQN.FeatureExtractor import FeatureExtractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepQNetworkReplay:
    def __init__(self, num_games=None, epsilon=0.6, gamma=0.8, alpha=0.01, minibatch_size=32):
        self.epsilon, self.gamma, self.alpha, self.minibatch_size = epsilon, gamma, alpha, minibatch_size

        self.epsilon_min = 0.20
        self.epsilon_diff = (epsilon - self.epsilon_min) / (num_games + 0.00000000000000001)

        self.int_to_action = {0: 'N', 1: 'S', 2: 'W', 3: 'E'}
        self.action_to_int = {'N': 0, 'S': 1, 'W': 2, 'E': 3}

        self.state_dim = 6
        self.action_dim = 4
        self.construct_q_network()
        self.losses = []

        self.feature_extractor = FeatureExtractor()
        self.replay_memory = np.empty(shape=(0, 5))
        self.mem_max_size = 100000

        self.int_to_action = {0: 'N', 1: 'S', 2: 'W', 3: 'E'}
        self.action_to_int = {'N': 0, 'S': 1, 'W': 2, 'E': 3}

    def construct_q_network(self):
        self.q_network = Sequential()
        self.q_network.add(Dense(64, input_dim=self.state_dim, activation='relu'))
        self.q_network.add(Dense(32, activation='relu'))
        self.q_network.add(Dense(self.action_dim, activation='linear'))
        self.q_network.compile(optimizer=tf.optimizers.Adam(), loss='mse')  # learning rate not used !

    def predict_move(self, agent, _):
        state = self.feature_extractor.get_state(agent)[0].numpy()
        # print(state, type(state))
        qvals_s = self.q_network.predict(state.reshape(1, self.state_dim))

        # Choose action to be epsilon-greedy
        if np.random.random() < self.epsilon:
            action_int = random.choice(range(self.action_dim))
        else:
            action_int = np.argmax(qvals_s)

        # Take step, store results
        action = self.int_to_action[action_int]
        reward = agent.get_reward(action)
        state_next = self.feature_extractor.get_state_a(agent, action)
        if state_next is None:
            state_next = np.zeros((1, 6), dtype=int)[0]
            # print(state_next, type(state_next))
        else:
            state_next = state_next[0].numpy()
        done = state_next is None

        # add to memory, respecting memory buffer limit
        if len(self.replay_memory) > self.mem_max_size:
            # self.replay_memory.pop(0)
            self.replay_memory = np.delete(self.replay_memory, 0, 0)

        # self.replay_memory.append({"s": state, "a": action_int, "r": reward, "state_next": state_next, "done": done})
        print(self.replay_memory)
        self.replay_memory = np.append(self.replay_memory, np.array([[state, action_int, reward, state_next, done]]),
                                       axis=0)
        print(self.replay_memory)

        # Train the nnet that approximates q(s,a), using the replay memory
        self.replay()

        # self.update_step()

        return action

    def replay(self):
        # choose <s,a,r,s',done> experiences randomly from the memory
        print(self.replay_memory.shape[0])
        rows_ind = np.random.choice(self.replay_memory.shape[0], size=self.minibatch_size, replace=True)
        minibatch = self.replay_memory[rows_ind, :]

        # number_of_rows = an_array.shape[0]
        # random_indices = np.random.choice(number_of_rows, size=2, replace=False)
        #
        # random_rows = an_array[random_indices, :]

        # create one list containing s, one list containing a, etc
        # s_l = np.array(list(map(lambda x: x['s'], minibatch)))
        # a_l = np.array(list(map(lambda x: x['a'], minibatch)))
        # r_l = np.array(list(map(lambda x: x['r'], minibatch)))
        # sprime_l = np.array(list(map(lambda x: x['state_next'], minibatch)))
        # done_l = np.array(list(map(lambda x: x['done'], minibatch)))


        s_l = minibatch[:, 0]
        a_l = minibatch[:, 1]
        r_l = minibatch[:, 2]
        sprime_l = minibatch[:, 3]
        done_l = minibatch[:, 4]

        # Find q(s', a') for all possible actions a'. Store in list
        # We'll use the maximum of these values for q-update
        print(sprime_l.shape, sprime_l[0])
        print(sprime_l.reshape(32,1))

        qvals_sprime_l = self.q_network.predict(sprime_l.reshape(32,1))

        # Find q(s,a) for all possible actions a. Store in list
        target_f = self.q_network.predict(s_l)

        # q-update target
        # For the action we took, use the q-update value
        # For other actions, use the current nnet predicted value
        for i, (s, a, r, qvals_sprime, done) in enumerate(zip(s_l, a_l, r_l, qvals_sprime_l, done_l)):
            if not done:
                target = r + self.gamma * np.max(qvals_sprime)
            else:
                target = r
            target_f[i][a] = target

        # Update weights of neural network with fit()
        # Loss function is 0 for actions we didn't take
        history = self.q_network.fit(s_l, target_f, epochs=1, verbose=0)
        self.losses.append(history.history['loss'][0])

    def update_step(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon -= self.epsilon_diff

    def get_model(self):
        return self.q_network

    def load_model(self, filepath):
        self.q_network = tf.keras.models.load_model(filepath)
