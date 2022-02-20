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
        self.epsilon, self.gamma, self.alpha, self.minibatch_size = epsilon, gamma, alpha, minibatch_size

        if num_games == 0:
            self.num_games = 0.00000000000000001
        else:
            self.num_games = num_games

        self.epsilon_min = 0.20
        self.epsilon_diff = (epsilon - self.epsilon_min) / self.num_games

        self.int_to_action = {0: 'N', 1: 'S', 2: 'W', 3: 'E'}
        self.action_to_int = {'N': 0, 'S': 1, 'W': 2, 'E': 3}

        self.state_dim = 5
        self.action_dim = 4
        self.replay_memory = []
        self.replay_interesting_actions_count = -0.1
        self.mem_max_size = 100000
        self.nn_target = self.nn_q = self.construct_q_network()

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
            qvals_s = self.nn_q.predict(state.reshape(1, self.state_dim))
            action_int = np.argmax(qvals_s)

        # Take step, store results
        action = self.int_to_action[action_int]
        reward = agent.get_reward(action)
        state_next, terminal = self.feature_extractor.get_state_a(agent, action)

        # add to memory, respecting memory buffer limit
        if len(self.replay_memory) > self.mem_max_size:
            exp = self.replay_memory[0]
            if self.is_experience_interesting(exp["r"], exp["s"]):
                self.replay_interesting_actions_count -= 1
            self.replay_memory.pop(0)

        if self.is_experience_interesting(reward, action):
            self.replay_interesting_actions_count += 1
            self.replay_memory.append(
                {"s": state, "a": action_int, "r": reward, "state_next": state_next, "terminal": terminal})
        else:
            if not self.replay_memory or self.replay_interesting_actions_count / len(self.replay_memory) < 0.5:
                self.replay_memory.append(
                    {"s": state, "a": action_int, "r": reward, "state_next": state_next, "terminal": terminal})

        # Train the nnet that approximates q(s,a), using the replay memory
        self.replay()

        return action

    @staticmethod
    def is_experience_interesting(reward, state):
        return reward != 10 or any(s < 0.2 for s in state[1:])

    def replay(self):
        # choose <s,a,r,s',done> experiences randomly from the memory
        minibatch = np.random.choice(self.replay_memory, self.minibatch_size, replace=True)

        # create one list containing s, one list containing a, etc
        s_l = np.array(list(map(lambda x: x['s'], minibatch)))
        a_l = np.array(list(map(lambda x: x['a'], minibatch)))
        r_l = np.array(list(map(lambda x: x['r'], minibatch)))
        sprime_l = np.array(list(map(lambda x: x['state_next'], minibatch)))
        terminal_l = np.array(list(map(lambda x: x['terminal'], minibatch)))

        # Find q(s', a') for all possible actions a'. Store in list
        # We'll use the maximum of these values for q-update
        # print("memory sprime:", sprime_l)
        # print(sprime_l)
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
        history = self.nn_q.fit(s_l, target_f, epochs=1, verbose=0)
        self.losses.append(history.history['loss'][0])

    def update_step(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon -= self.epsilon_diff

    def get_model(self):
        return self.nn_q

    def load_model(self, filepath):
        self.nn_q = tf.keras.models.load_model(filepath)
