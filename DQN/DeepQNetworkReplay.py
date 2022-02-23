import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import os

from DQN.FeatureExtractor import FeatureExtractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepQNetworkReplay:
    def __init__(self, num_games=0, epsilon=0.8, gamma=0.8, alpha=0.001, minibatch_size=32):
        self.epsilon, self.gamma, self.alpha = epsilon, gamma, alpha

        if num_games == 0:
            self.num_games = 0.00000000000000001
        else:
            self.num_games = num_games

        self.epsilon_min = 0.2
        self.epsilon_diff = (epsilon - self.epsilon_min) / self.num_games

        self.int_to_action = {0: 'N', 1: 'S', 2: 'W', 3: 'E'}
        self.action_to_int = {'N': 0, 'S': 1, 'W': 2, 'E': 3}

        self.state_dim = 8
        self.action_dim = 4
        self.nn_target = self.nn_q = self.construct_q_network()

        self.replay_memory_boring = []
        self.replay_memory_interesting = []
        self.mem_max_size = 200000
        interesting_ext_to_boring_ration = 0.7
        self.n_interesting = int(minibatch_size * interesting_ext_to_boring_ration)
        self.n_boring = minibatch_size - self.n_interesting

        self.feature_extractor = FeatureExtractor()
        self.losses = []

    def construct_q_network(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.alpha), loss='mse')
        return model

    def predict_move(self, agent, _):
        state, _ = self.feature_extractor.get_state(agent)

        # Choose action to be epsilon-greedy
        if np.random.random() < self.epsilon:
            action_int = random.choice(range(self.action_dim))
        else:
            qvals_s = self.nn_q.predict(tf.convert_to_tensor([state]))
            action_int = np.argmax(qvals_s)

        # test mode
        if self.epsilon == 0:
            return self.int_to_action[action_int]

        # Take step, store results
        action = self.int_to_action[action_int]
        reward = agent.get_reward(action)
        state_next, terminal = self.feature_extractor.get_state_a(agent, action)

        # add to memory, respecting memory buffer limit
        if len(self.replay_memory_boring) > self.mem_max_size:
            self.replay_memory_boring.pop(0)
        if len(self.replay_memory_interesting) > self.mem_max_size:
            self.replay_memory_interesting.pop(0)

        if self.is_experience_interesting(reward, state):
            self.replay_memory_interesting.append(
                {"s": state, "a": action_int, "r": reward, "state_next": state_next, "terminal": terminal})
        else:
            self.replay_memory_boring.append(
                {"s": state, "a": action_int, "r": reward, "state_next": state_next, "terminal": terminal})

        # Train the nnet that approximates q(s,a), using the replay memory
        self.replay()

        return action

    @staticmethod
    def is_experience_interesting(reward, state):
        # return reward != 10 or 1 < np.sum([s <= 0.15 for s in state])
        return reward != 10 or 1 < np.sum([s <= 0.10 for s in state[:4]]) or 1 < np.sum([s <= 0.10 for s in state[4:]])

    def replay(self):
        # choose <s,a,r,s',done> experiences randomly from the memory

        if len(self.replay_memory_interesting) > self.n_interesting:
            minibatch = np.random.choice(self.replay_memory_interesting, self.n_interesting)
        else:
            minibatch = np.array(self.replay_memory_interesting)

        if len(self.replay_memory_boring) > self.n_boring:
            minibatch = np.append(minibatch, np.random.choice(self.replay_memory_boring, self.n_boring), axis=0)
        else:
            minibatch = np.append(minibatch, self.replay_memory_boring)

        np.random.shuffle(minibatch)

        # create one list containing s, one list containing a, etc
        s_l = np.array(list(map(lambda x: x['s'], minibatch)))
        a_l = np.array(list(map(lambda x: x['a'], minibatch)))
        r_l = np.array(list(map(lambda x: x['r'], minibatch)))
        sprime_l = np.array(list(map(lambda x: x['state_next'], minibatch)))
        terminal_l = np.array(list(map(lambda x: x['terminal'], minibatch)))

        # Find q(s', a') for all possible actions a'. Store in list
        # We'll use the maximum of these values for q-update
        qvals_sprime_l = self.nn_target.predict(sprime_l)

        # Find q(s,a) for all possible actions a. Store in list
        target_f = self.nn_q.predict(s_l)

        # q-update target
        # For the action we took, use the q-update value
        # For other actions, use the current nnet predicted value
        for i, (s, a, r, qvals_sprime, done) in enumerate(zip(s_l, a_l, r_l, qvals_sprime_l, terminal_l)):
            if not done:
                target = r + self.gamma * np.max(qvals_sprime)
            else:
                target = r
            target_f[i][a] = target

        # Update weights of neural network with fit()
        # Loss function is 0 for actions we didn't take
        history = self.nn_q.fit(s_l, target_f, epochs=10, verbose=0)
        self.losses.append(history.history['loss'][0])

    def update_step(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon -= self.epsilon_diff

    def get_model(self):
        return self.nn_q

    def load_model(self, filepath):
        self.nn_q = tf.keras.models.load_model(filepath)
