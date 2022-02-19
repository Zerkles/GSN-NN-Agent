import random
from copy import copy

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.UserParam import UserSettableParameter

from Agents.DeepAgent import DeepAgent
from Agents.ReflexAgent import ReflexAgent
from Agents.RandomAgent import RandomAgent
from main import TronModel

number_of_colors = 12
MAP_DIM = 26

color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         for i in range(number_of_colors)]
color_dict = dict(zip(list(range(12)), color))


def darken(rgb, rate=0.5):
    rgb = rgb.replace('#', '')
    s = '#'
    for i in [0, 2, 4]:
        c = rgb[i:i + 2]
        c = int(c, 16)
        c = int(c * rate)
        c = format(c, '02x')
        s += c
    return s


def tronPortrayal(agent):
    # if agent is None:
    #     return

    if agent.portrayal is not None:
        return copy(agent.portrayal)

    if isinstance(agent, RandomAgent):
        portrayal = {"Shape": "circle",
                     "Filled": "false",
                     "Layer": 0,
                     "Color": color_dict[agent.unique_id],
                     "r": 0.5
                     }

        if isinstance(agent, ReflexAgent):
            portrayal['Filled'] = "true"

        elif isinstance(agent, DeepAgent):
            portrayal.pop("r")
            portrayal['Shape'] = 'rect'
            portrayal['w'], portrayal['h'] = 0.5, 0.5
            portrayal['Filled'] = "true"

        agent.portrayal_child = copy(portrayal)
        portrayal["Color"] = darken(portrayal["Color"], 0.5)
        agent.portrayal = portrayal

    return agent.portrayal


grid = CanvasGrid(tronPortrayal, MAP_DIM, MAP_DIM, 500, 500)

server = ModularServer(TronModel,
                       [grid],
                       "Tron Agent Simulator",
                       {
                           "n_random_agents": UserSettableParameter("slider", "Number of Random Agents", 0, 0, 12, 1),
                           "n_reflex_agents": UserSettableParameter("slider", "Number of Reflex Agents", 0, 0, 12, 1),
                           "n_deep_agents": UserSettableParameter("slider", "Number of Deep Agents", 1, 0, 12, 1),
                           "max_path_length": UserSettableParameter("slider", "Max Lightpath Length", 676, 10, 676, 1),
                           "isTestMode": UserSettableParameter("checkbox", "Test Mode", True),
                           "isStartingPositionRandom": UserSettableParameter("checkbox", "Random Starting Positions",
                                                                             False)
                       }
                       )
server.port = 8521
server.launch()
