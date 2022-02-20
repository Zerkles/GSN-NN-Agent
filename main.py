import random
import time
from statistics import mean

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from Agents.ReflexAgent import ReflexAgent
from DQN.DeepQNetwork import DeepQNetwork
from Agents.RandomAgent import RandomAgent
from Agents.DeepAgent import DeepAgent
import matplotlib.pyplot as plt

from DQN.DeepQNetworkReplay import DeepQNetworkReplay
from DQN.DeepQNetworkReplayFull import DeepQNetworkReplayFull

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
                 isTestMode, dqn_obj):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(MAP_DIM, MAP_DIM, torus=False)
        self.startingPositions = []
        self.alive_agents = n_random_agents + n_reflex_agents + n_deep_agents
        self.scores = {}

        self.dqn_obj = dqn_obj
        if isTestMode:
            self.dqn_obj.epsilon = 0
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
            print(self.scores)
            # print(self.dqn_obj.epsilon)
        else:
            self.schedule.step()


N_GAMES = 300
DQN_OBJ_TYPE = DeepQNetworkReplayFull

if __name__ == '__main__':
    losses = []
    dqn_obj = DQN_OBJ_TYPE(num_games=N_GAMES)

    for n in range(N_GAMES):
        print("Run:", n)

        model = TronModel(n_random_agents=0, n_reflex_agents=0, n_deep_agents=4, max_path_length=676,
                          isStartingPositionRandom=0, isTestMode=False, dqn_obj=dqn_obj)

        model.run_model()
        model.dqn_obj.update_step()
        losses.append(mean(model.dqn_obj.losses))
        model.dqn_obj.losses = []

    plt.plot(range(len(losses)), losses)
    plt.yscale('log')

    plt.savefig(f"plots/log_loss_{time.strftime('%H-%M-%S', time.localtime())}.png")

    dqn_model = dqn_obj.get_model()
    dqn_model.save("model.save")
