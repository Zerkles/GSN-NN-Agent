import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import os

from DQN.FeatureExtractor import FeatureExtractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepQNetworkReplayFull:
    def __init__(self, num_games=None, epsilon=0.8, gamma=0.8, alpha=0.001, minibatch_size=32):
        self.epsilon, self.gamma, self.alpha, self.minibatch_size = epsilon, gamma, alpha, minibatch_size

        self.epsilon_min = 0.20
        self.epsilon_diff = (epsilon - self.epsilon_min) / (num_games + 0.00000000000000001)

        self.int_to_action = {0: 'N', 1: 'S', 2: 'W', 3: 'E'}
        self.action_to_int = {'N': 0, 'S': 1, 'W': 2, 'E': 3}

        self.state_dim = 6
        self.action_dim = 4
        self.replay_memory = []
        self.mem_max_size = 100000
        self.nn_q = self.construct_q_network()
        self.nn_target = self.construct_q_network()
        self.C = 100

        self.feature_extractor = FeatureExtractor()
        self.losses = []

    def construct_q_network(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.alpha), loss='huber_loss')
        return model

    def predict_move(self, agent, _):
        state = self.feature_extractor.get_state(agent)[0].numpy()
        # print(state, type(state))

        # Choose action to be epsilon-greedy
        if np.random.random() < self.epsilon:
            action_int = random.choice(range(self.action_dim))
        else:
            qvals_s = self.nn_q.predict(state.reshape(1, self.state_dim))
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
            self.replay_memory.pop(0)
        self.replay_memory.append({"s": state, "a": action_int, "r": reward, "state_next": state_next, "done": done})

        if agent.model.schedule.steps % self.C == 0:
            self.nn_target.set_weights(self.nn_q.get_weights())

        # Train the nnet that approximates q(s,a), using the replay memory
        self.replay()

        # self.update_step()

        return action

    def replay(self):
        # choose <s,a,r,s',done> experiences randomly from the memory
        minibatch = np.random.choice(self.replay_memory, self.minibatch_size, replace=True)

        # create one list containing s, one list containing a, etc
        s_l = np.array(list(map(lambda x: x['s'], minibatch)))
        a_l = np.array(list(map(lambda x: x['a'], minibatch)))
        r_l = np.array(list(map(lambda x: x['r'], minibatch)))
        sprime_l = np.array(list(map(lambda x: x['state_next'], minibatch)))
        done_l = np.array(list(map(lambda x: x['done'], minibatch)))

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
        for i, (s, a, r, qvals_sprime, done) in enumerate(zip(s_l, a_l, r_l, qvals_sprime_l, done_l)):
            if not done:
                target = r + self.gamma * np.max(qvals_sprime)
            else:
                target = r
            target_f[i][a] = target

        # Update weights of neural network with fit()
        # Loss function is 0 for actions we didn't take
        history = self.nn_q.fit(s_l, target_f, epochs=1, verbose=0)
        self.losses.append(history.history['loss'][0])

    @staticmethod
    def huber_loss1(y, yhat, delta=3):
        """Adapted from here: https://jaromiru.com/2016/10/21/lets-make-a-dqn-full-dqn/
        y: true value, yhat: predicted value"""
        error = y - yhat
        cond = tf.keras.abs(error) < delta
        L2 = 0.5 * tf.keras.square(error)
        L1 = delta * (tf.keras.abs(error) - 0.5 * delta)
        loss = tf.where(cond, L2, L1)
        return tf.keras.mean(loss)

    @staticmethod
    def huber_loss(a, b, in_keras=True):
        error = a - b
        quadratic_term = error * error / 2
        linear_term = abs(error) - 1 / 2
        use_linear_term = (abs(error) > 1.0)
        if in_keras:
            # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
            use_linear_term = tf.keras.cast(use_linear_term, 'float32')
        return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term

    def update_step(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon -= self.epsilon_diff

    def get_model(self):
        return self.nn_q #target ?

    def load_model(self, filepath):
        self.nn_q = tf.keras.models.load_model(filepath)
