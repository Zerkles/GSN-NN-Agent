import random
from mesa import Agent

from Agents.Lightpath import Lightpath


class RandomAgent(Agent):

    def __init__(self, unique_id, pos, model, max_path_length):
        super().__init__(unique_id, model)

        self.pos = pos
        self.last_action = None
        self.lightpath = []
        self.max_path_length = max_path_length
        self.portrayal, self.portrayal_child = None, None
        self.score = 0

    def choose_action(self, legal_actions):
        return random.choice(legal_actions)

    def step(self):
        legal_actions = self.get_legal_actions()
        action = self.choose_action(legal_actions)
        self.score += self.get_reward(action)
        # print(self.pos, self.get_legal_actions(), action)

        if self.will_die(action):
            self.on_death()
        else:
            self.make_trail()
            self.model.grid.move_agent(self, self.calc_pos(action))
            self.last_action = action
            self.eat_your_tail()

    def make_trail(self):
        lightpath = Lightpath(self.model.next_id(), self.model, self.pos, self.unique_id, self.portrayal_child)
        self.lightpath.append(lightpath)
        self.model.grid.place_agent(lightpath, self.pos)

    def eat_your_tail(self):
        if len(self.lightpath) > self.max_path_length:
            self.model.grid.remove_agent(self.lightpath.pop(0))

    def on_death(self):
        for obj in self.lightpath:
            self.model.grid.remove_agent(obj)

        self.model.grid.remove_agent(self)
        self.model.schedule.remove(self)

        self.model.scores.update(
            {self.model.alive_agents: {'type': f"{self.__class__.__name__}{self.unique_id}",
                                       'score': self.score}})
        self.model.alive_agents -= 1

    def will_die(self, action):
        return action not in self.get_legal_actions() or not self.model.grid.is_cell_empty(self.calc_pos(action))

    def calc_pos(self, action):
        x, y = self.pos

        if action == 'N':
            y += 1
        elif action == 'S':
            y -= 1
        elif action == 'W':
            x -= 1
        elif action == 'E':
            x += 1

        return tuple([x, y])

    def get_legal_actions(self):
        legal_actions = []
        x, y = self.pos

        if y + 1 < self.model.grid.height and self.last_action != 'S':
            legal_actions.append('N')
        if y - 1 >= 0 and self.last_action != 'N':
            legal_actions.append('S')

        if x - 1 >= 0 and self.last_action != 'E':
            legal_actions.append('W')
        if x + 1 < self.model.grid.width and self.last_action != 'W':
            legal_actions.append('E')

        return legal_actions

    def get_reward(self, action):
        if self.will_die(action):  # Died
            # if (self.score / 10) > self.max_reward_threshold and isinstance(self,ReflexAgent):
            #     print("WOW")
            #     return 410  # Great result!
            # elif self.model.alive_agents == 1:
            #     return -190  # Won, but still died
            # else:
            return -990  # Lost game
        return 10
