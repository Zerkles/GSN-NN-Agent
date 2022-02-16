import os
from mesa import Agent
from agents.NNUtilities import DeepNeuralNetwork
from agents.RandomAgent import RandomAgent


class DeepAgent(RandomAgent):

    def __init__(self, unique_id, pos, direction, model, fov, max_path_length):
        super().__init__(unique_id, pos, direction, model, fov, max_path_length)

        self.network = model.neural_network

    def choose_action(self, fillings):
        return self.network.predict_move(self)
