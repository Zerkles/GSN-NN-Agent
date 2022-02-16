import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class FeatureExtractor:
    @staticmethod
    def get_state(agent, pos=None):
        if pos is None:
            pos = agent.pos

        x, y = pos

        return tf.constant([[x, y]])

    @staticmethod
    def get_state_a(agent, action):
        return FeatureExtractor.get_state(agent, agent.calc_next_pos(action))


class RewardGenerator:
    @staticmethod
    def get_reward(agent, action):
        if agent.is_dead(agent.calc_next_pos(action)):  # Died
            if agent.model.n_agents_left() == 1:  # Won game
                return 500
            else:
                return -500  # Lost game

        return 10  # Going forward

        # Killed enemy ???


class DeepNeuralNetwork:
    def __init__(self, num_games=0, epsilon=0.25, gamma=0.8, alpha=0.2):
        self.epsilon, self.gamma, self.alpha = epsilon, gamma, alpha

        self.epsilon_min = 0.20
        self.epsilon_diff = (epsilon - self.epsilon_min) / (num_games + 0.00000000000000001)

        n_input = 2
        n_output = 4
        self.q_network = self.construct_q_network(n_input, n_output)

        self.opt = tf.optimizers.Adam(learning_rate=self.alpha)
        self.feature_extractor = FeatureExtractor()
        self.reward_giver = RewardGenerator()

    def construct_q_network(self, state_dim, action_dim):
        """Construct the q-network with q-values per action as output"""
        inputs = tf.keras.layers.Input(shape=(state_dim,))  # input dimension
        hidden1 = tf.keras.layers.Dense(
            25, activation="relu", kernel_initializer=tf.keras.initializers.he_normal()
        )(inputs)
        hidden2 = tf.keras.layers.Dense(
            25, activation="relu", kernel_initializer=tf.keras.initializers.he_normal()
        )(hidden1)
        hidden3 = tf.keras.layers.Dense(
            25, activation="relu", kernel_initializer=tf.keras.initializers.he_normal()
        )(hidden2)
        q_values = tf.keras.layers.Dense(
            action_dim, kernel_initializer=tf.keras.initializers.Zeros(), activation="linear"
        )(
            hidden3
        )

        return tf.keras.Model(inputs=inputs, outputs=[q_values])

    def predict_move(self, agent):
        with tf.GradientTape() as tape:
            # print(123)
            state = self.feature_extractor.get_state(agent)

            "Obtain Q-values from network"
            q_values = self.q_network(state)

            "Select action using epsilon-greedy policy"
            sample_epsilon = np.random.rand()
            if sample_epsilon <= self.epsilon:  # Select random action
                action = np.random.choice(4)
            else:  # Select action with highest Q-value
                action = np.argmax(q_values[0])

            "Obtain direct reward for selected action"
            reward = self.reward_giver.get_reward(agent, action)

            "Obtain Q-value for selected action"
            current_q_value = q_values[0, action]

            "Determine next state"
            next_state = self.feature_extractor.get_state_a(agent, action)

            "Select next action with highest Q-value"
            # if next_state == agent.is_dead(agent.calc_next_pos(action)):
            if agent.is_dead(agent.calc_next_pos(action)):
                next_q_value = 0  # No Q-value for terminal
            else:
                next_q_values = tf.stop_gradient(self.q_network(next_state))  # No gradient computation
                next_action = np.argmax(next_q_values[0])
                next_q_value = next_q_values[0, next_action]

            "Compute observed Q-value"
            observed_q_value = reward + (self.gamma * next_q_value)

            "Compute loss value"
            loss_value = (observed_q_value - current_q_value) ** 2

            "Compute gradients"
            grads = tape.gradient(loss_value, self.q_network.trainable_variables)

            "Apply gradients to update network weights"
            self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))

            actions_map = {0: 'N', 1: 'S', 2: 'W', 3: 'E'}
            return actions_map[action]

    def update_after_game(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon -= self.epsilon_diff

    def get_model(self):
        return self.q_network

    def load_model(self, filepath):
        self.q_network = tf.keras.models.load_model(filepath)
