import pickle
import random

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from agents.LightcycleAgent import LightcycleAgent
from agents.NNUtilities import DeepNeuralNetwork
from agents.RandomAgent import RandomAgent
from agents.DeepAgent import DeepAgent


def getStartingPosition(startingPositions, isRandom):
    if isRandom:
        coords = (random.randrange(0, 25), random.randrange(0, 25))
        while coords in startingPositions:
            coords = (random.randrange(0, 25), random.randrange(0, 25))
        return coords
    else:
        options = [(2, 13), (23, 13), (13, 2), (13, 23), (2, 6), (23, 20), (2, 20), (23, 6), (6, 2), (20, 23), (20, 2),
                   (6, 23)]
        return next(x for x in options if x not in startingPositions)


def getStartingDirection(position, isRandom):
    if isRandom:
        return random.choice(['N', 'S', 'W', 'E'])
    if max(26 - position[0], position[0]) > max(26 - position[1], position[1]):
        if 26 - position[0] > position[0]:
            return 'E'
        else:
            return 'W'
    else:
        if 26 - position[1] > position[1]:
            return 'N'
        else:
            return 'S'


class TronModel(Model):
    def __init__(self, n_random_agents, n_light_agents, n_deep_agents, max_path_length, fov, isStartingPositionRandom,
                 neural_network=None, mode='show'):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(26, 26, torus=False)
        self.startingPositions = []

        self.neural_network = neural_network
        if mode == 'show':
            self.neural_network=DeepNeuralNetwork()
            self.neural_network.load_model("model.save")

        ag_lst = ['random'] * n_random_agents + ['light'] * n_light_agents + ['deep'] * n_deep_agents

        for i in range(len(ag_lst)):
            self.startingPositions.append(getStartingPosition(self.startingPositions, isStartingPositionRandom))

            if ag_lst[i] == 'random':
                a = RandomAgent(i, self.startingPositions[-1],
                                getStartingDirection(self.startingPositions[-1], isStartingPositionRandom), self, fov,
                                max_path_length)
            elif ag_lst[i] == 'light':
                a = LightcycleAgent(i, self.startingPositions[-1],
                                    getStartingDirection(self.startingPositions[-1], isStartingPositionRandom), self,
                                    fov,
                                    max_path_length)
            else:
                a = DeepAgent(i, self.startingPositions[-1],
                              getStartingDirection(self.startingPositions[-1], isStartingPositionRandom), self, fov,
                              max_path_length)

            self.schedule.add(a)
            self.grid.place_agent(a, self.startingPositions[-1])

    def n_agents_left(self):
        return sum([isinstance(x, (LightcycleAgent, RandomAgent, DeepAgent)) for x in self.schedule.agents])

    def step(self):
        self.schedule.step()

        if self.n_agents_left() == 0:
            # save network
            self.running = False


if __name__ == '__main__':
    num_games = 1000
    neural_network = DeepNeuralNetwork(num_games, 0.25, 0.8, 0.2)


    for i in range(num_games):
        print("Run:", i)

        model = TronModel(n_random_agents=0, n_light_agents=0, n_deep_agents=4, max_path_length=676, fov=26,
                          isStartingPositionRandom=0, neural_network=neural_network, mode='train')

        model.run_model()
        model.neural_network.update_after_game()

    neural_network.get_model().save("model.save")
