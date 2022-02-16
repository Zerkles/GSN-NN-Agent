import random

from agents.RandomAgent import RandomAgent


class LightcycleAgent(RandomAgent):

    def __init__(self, unique_id, pos, direction, model, fov, max_path_length):
        super().__init__(unique_id, pos, direction, model, fov, max_path_length)
        self.first_move = True

    def choose_action(self, fillings):
        while len(fillings) > 0:

            if self.first_move:
                new_direction = random.choice(list(fillings.keys()))
                self.first_move = False
            else:
                new_direction = min(fillings, key=fillings.get)
            new_pos = list(self.pos)
            if new_direction == 'N':
                new_pos[1] += 1
                if tuple(new_pos) in self.lightpath or tuple(new_pos) in self.boundries or tuple(
                        new_pos) in self.others_lightpaths:
                    del fillings[new_direction]
                else:
                    break

            elif new_direction == 'S':
                new_pos[1] -= 1
                if tuple(new_pos) in self.lightpath or tuple(new_pos) in self.boundries or tuple(
                        new_pos) in self.others_lightpaths:
                    del fillings[new_direction]
                else:
                    break

            elif new_direction == 'W':
                new_pos[0] -= 1
                if tuple(new_pos) in self.lightpath or tuple(new_pos) in self.boundries or tuple(
                        new_pos) in self.others_lightpaths:
                    del fillings[new_direction]
                else:
                    break

            elif new_direction == 'E':
                new_pos[0] += 1
                if tuple(new_pos) in self.lightpath or tuple(new_pos) in self.boundries or tuple(
                        new_pos) in self.others_lightpaths:
                    del fillings[new_direction]
                else:
                    break

            return new_direction

    def observation(self):
        fov_grid = [(self.pos[0], self.pos[1] + n) for n in range(-self.fov, self.fov + 1) if
                    self.pos[1] + n >= 0 and self.pos[1] + n <= 25] + \
                   [(self.pos[0] + n, self.pos[1]) for n in range(-self.fov, self.fov + 1) if
                    self.pos[0] + n >= 0 and self.pos[0] + n <= 25]
        for agent in self.model.schedule.agents:
            if agent.unique_id != self.unique_id:
                for point in agent.lightpath:
                    if point in fov_grid:
                        self.others_lightpaths.add(point)
                if agent.pos in fov_grid:
                    self.others_lightpaths.add(agent.pos)

    def step(self):
        self.lightpath.add(self.pos)
        self.observation()

        all_paths = set.union(self.lightpath, self.others_lightpaths)
        if self.direction == 'N':

            left = len([n for n in all_paths if n[0] < self.pos[0]])
            front = len([n for n in all_paths if n[1] > self.pos[1]])
            right = len([n for n in all_paths if n[0] > self.pos[0]])
            fillings = {'W': left, 'N': front, 'E': right}

        elif self.direction == 'S':
            left = len([n for n in all_paths if n[0] > self.pos[0]])
            front = len([n for n in all_paths if n[1] < self.pos[1]])
            right = len([n for n in all_paths if n[0] < self.pos[0]])
            fillings = {'W': right, 'S': front, 'E': left}

        elif self.direction == 'W':
            left = len([n for n in all_paths if n[1] < self.pos[1]])
            front = len([n for n in all_paths if n[0] < self.pos[0]])
            right = len([n for n in all_paths if n[1] > self.pos[1]])
            fillings = {'N': right, 'W': front, 'S': left}

        else:
            left = len([n for n in all_paths if n[1] > self.pos[1]])
            front = len([n for n in all_paths if n[0] > self.pos[0]])
            right = len([n for n in all_paths if n[1] < self.pos[1]])
            fillings = {'S': right, 'E': front, 'N': left}
        self.move(fillings)
