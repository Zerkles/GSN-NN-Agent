import random

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.UserParam import UserSettableParameter

from agents.LightcycleAgent import LightcycleAgent
from agents.RandomAgent import RandomAgent
from main import TronModel

number_of_colors = 12

color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         for i in range(number_of_colors)]
color_dict = dict(zip(list(range(12)), color))


def tronPortrayal(agent):
    if agent is None:
        return
    if isinstance(agent, RandomAgent) == 0:
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 0,
                     "Color": "#FF0000",
                     "r": 0.5
                     }
    elif isinstance(agent, LightcycleAgent) == 1:
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 0,
                     "Color": "#0000FF",
                     "r": 0.5
                     }
    else:

        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 0,
                     "Color": color_dict[agent.unique_id],
                     "r": 0.5}

    return portrayal


grid = CanvasGrid(tronPortrayal, 26, 26, 500, 500)

# value, minimum, maximum,step

server = ModularServer(TronModel,
                       [grid],
                       "Tron Agent Simulator",
                       {
                           "n_random_agents": UserSettableParameter("slider", "Number of Random Agents", 0, 0, 12, 1),
                           "n_light_agents": UserSettableParameter("slider", "Number of Light Agents", 1, 0, 12, 1),
                           "n_deep_agents": UserSettableParameter("slider", "Number of Deep Agents", 1, 0, 12, 1),
                           "max_path_length": UserSettableParameter("slider", "Max Lightpath Length", 676, 10, 676, 1),
                           "fov": UserSettableParameter("slider", "Field of View", 26, 1, 26, 1),
                           "isStartingPositionRandom": UserSettableParameter("checkbox", "Random Starting Positions",
                                                                             False),
                       }
                       )
server.port = 8521
server.launch()
