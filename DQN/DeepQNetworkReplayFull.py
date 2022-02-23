import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import os

from DQN.DeepQNetworkReplay import DeepQNetworkReplay
from DQN.FeatureExtractor import FeatureExtractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepQNetworkReplayFull(DeepQNetworkReplay):
    def __init__(self, num_games=0, epsilon=0.8, gamma=0.8, alpha=0.001, minibatch_size=32):
        super().__init__(num_games, epsilon, gamma, alpha, minibatch_size)

        self.epsilon_min = 0.2
        self.epsilon_diff = (epsilon - self.epsilon_min) / self.num_games

        self.nn_q = self.construct_q_network()
        self.nn_target = self.construct_q_network()
        self.C = int(self.num_games / 10.0)

        if self.C == 0:
            self.C = 1

    def construct_q_network(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.alpha), loss='huber_loss')
        return model

    def predict_move(self, agent, _):
        if agent.model.schedule.steps % self.C == 0:
            self.nn_target.set_weights(self.nn_q.get_weights())
        return super().predict_move(agent, _)
