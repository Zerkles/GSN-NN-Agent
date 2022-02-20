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

        self.nn_q = self.construct_q_network()
        self.nn_target = self.construct_q_network()
        self.C = 100

    def construct_q_network(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.alpha), loss='huber_loss')
        return model

    def predict_move(self, agent, _):
        if agent.model.schedule.steps % self.C == 0:
            self.nn_target.set_weights(self.nn_q.get_weights())
        return super().predict_move(agent, _)

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
