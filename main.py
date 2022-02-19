import random

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from agents.ReflexAgent import ReflexAgent
from DQN.DeepQNetwork import DeepQNetwork
from agents.RandomAgent import RandomAgent
from agents.DeepAgent import DeepAgent

MAP_DIM = 26


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
    if max(MAP_DIM - position[0], position[0]) > max(MAP_DIM - position[1], position[1]):
        if MAP_DIM - position[0] > position[0]:
            return 'E'
        else:
            return 'W'
    else:
        if MAP_DIM - position[1] > position[1]:
            return 'N'
        else:
            return 'S'


class TronModel(Model):
    def __init__(self, n_random_agents, n_reflex_agents, n_deep_agents, max_path_length, isStartingPositionRandom,
                 isTestMode,
                 neural_network=None):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(MAP_DIM, MAP_DIM, torus=False)
        self.startingPositions = []
        self.alive_agents = n_random_agents + n_reflex_agents + n_deep_agents

        self.dqn_obj = neural_network
        if isTestMode:
            self.dqn_obj = DeepQNetwork(epsilon=0)
            self.dqn_obj.load_model("model.save")

        ag_lst = ['random'] * n_random_agents + ['light'] * n_reflex_agents + ['deep'] * n_deep_agents

        for i in range(len(ag_lst)):
            self.startingPositions.append(getStartingPosition(self.startingPositions, isStartingPositionRandom))

            if ag_lst[i] == 'random':
                a = RandomAgent(self.next_id(), self.startingPositions[-1], self, max_path_length)

            elif ag_lst[i] == 'light':
                a = ReflexAgent(self.next_id(), self.startingPositions[-1], self, max_path_length)

            else:
                a = DeepAgent(self.next_id(), self.startingPositions[-1], self, max_path_length)

            self.schedule.add(a)
            self.grid.place_agent(a, self.startingPositions[-1])

    def step(self):
        if self.alive_agents == 0:
            self.running = False
        else:
            self.schedule.step()


if __name__ == '__main__':
    num_games = 1000
    dqn_obj = DeepQNetwork(num_games, 0.25, 0.8, 0.2)

    for n in range(num_games):
        print("Run:", n)

        model = TronModel(n_random_agents=0, n_reflex_agents=0, n_deep_agents=4, max_path_length=676,
                          isStartingPositionRandom=0, isTestMode=False, neural_network=dqn_obj)

        model.run_model()
        model.dqn_obj.update_step()

    dqn_model = dqn_obj.get_model()
    dqn_model.compile(optimizer=dqn_obj.opt, loss='mse')
    dqn_model.save("model.save")
