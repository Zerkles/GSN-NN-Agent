import random
import time
from statistics import mean

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from Agents.ReflexAgent import ReflexAgent
from Agents.RandomAgent import RandomAgent
from Agents.DeepAgent import DeepAgent
import matplotlib.pyplot as plt

from DQN.DeepQNetworkReplay import DeepQNetworkReplay

MAP_DIM = 20


def getStartingPosition(startingPositions, isRandom):
    if isRandom:
        coords = (random.randrange(0, MAP_DIM), random.randrange(0, MAP_DIM))
        while coords in startingPositions:
            coords = (random.randrange(0, MAP_DIM), random.randrange(0, MAP_DIM))
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
    def __init__(self, n_random_agents, n_reflex_agents, n_deep_agents, max_path_length,
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

        ag_lst = ['random'] * n_random_agents + ['reflex'] * n_reflex_agents + ['deep'] * n_deep_agents

        for i in range(len(ag_lst)):
            self.startingPositions.append(getStartingPosition(self.startingPositions, True))

            if ag_lst[i] == 'random':
                a = RandomAgent(self.next_id(), self.startingPositions[-1], self, max_path_length)

            elif ag_lst[i] == 'reflex':
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


def train(n_games):
    losses = []
    scores = []
    dqn_obj = DQN_OBJ_TYPE(num_games=n_games)

    for n in range(n_games):
        print("Run:", n)

        model = TronModel(n_random_agents=0, n_reflex_agents=1, n_deep_agents=2, max_path_length=676,
                          isTestMode=False, dqn_obj=dqn_obj)

        model.run_model()
        model.dqn_obj.update_step()

        losses.append(mean(model.dqn_obj.losses))
        model.dqn_obj.losses = []
        scores.append(mean([model.scores[k]["score"] for k in model.scores.keys()]))
        print(len(model.dqn_obj.replay_memory_interesting), len(model.dqn_obj.replay_memory_boring))

    t = time.strftime('%H-%M-%S', time.localtime())
    plt.plot(range(len(scores)), scores)
    plt.savefig(f"plots/scores_{t}.png")
    plt.clf()

    plt.plot(range(len(losses)), losses)
    plt.yscale('log')
    plt.savefig(f"plots/log_loss_{t}.png")
    plt.clf()

    dqn_model = dqn_obj.get_model()
    dqn_model.save("model.save")


def test(n_games):
    scores = {"DeepAgent": {"pos": [], "score": []}, "RandomAgent": {"pos": [], "score": []},
              "ReflexAgent": {"pos": [], "score": []}}
    dqn_obj = DQN_OBJ_TYPE(num_games=n_games)

    for n in range(n_games):
        print("Run:", n)

        model = TronModel(n_random_agents=2, n_reflex_agents=2, n_deep_agents=2, max_path_length=676,
                          isTestMode=True, dqn_obj=dqn_obj)

        model.run_model()

        for k in model.scores.keys():
            if model.scores[k]["type"].find("RandomAgent") == 0:
                scores["RandomAgent"]["score"].append(model.scores[k]["score"])
                scores["RandomAgent"]["pos"].append(k)
            elif model.scores[k]["type"].find("ReflexAgent") == 0:
                scores["ReflexAgent"]["score"].append(model.scores[k]["score"])
                scores["ReflexAgent"]["pos"].append(k)
            elif model.scores[k]["type"].find("DeepAgent") == 0:
                scores["DeepAgent"]["score"].append(model.scores[k]["score"])
                scores["DeepAgent"]["pos"].append(k)

    t = time.strftime('%H-%M-%S', time.localtime())
    plt.bar(scores.keys(), [mean(scores[k]["score"]) for k in scores.keys()])
    plt.savefig(f"plots/scores_bars_{t}.png")
    plt.clf()

    plt.bar(scores.keys(), [mean(scores[k]["pos"]) for k in scores.keys()])
    plt.savefig(f"plots/pos_bars_{t}.png")
    plt.clf()


DQN_OBJ_TYPE = DeepQNetworkReplay

if __name__ == '__main__':
    train(3)
    test(3)
