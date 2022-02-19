from agents.RandomAgent import RandomAgent


class DeepAgent(RandomAgent):

    def __init__(self, unique_id, pos, model, max_path_length):
        super().__init__(unique_id, pos, model, max_path_length)

        self.dqn_boj = model.dqn_obj
        self.simulated_action = None
        self.lost_tail = None

    def choose_action(self, legal_actions):
        return self.dqn_boj.predict_move(self, legal_actions)

    def simulate_step(self, action):
        self.make_trail()
        self.model.grid.move_agent(self, self.calc_pos(action))
        self.simulated_action = action
        self.lost_tail = self.eat_your_tail()

    def reverse_step(self):
        opposite_map = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
        action = opposite_map[self.simulated_action]

        self.model.grid.remove_agent(self.lightpath.pop(-1))
        self.model.grid.move_agent(self, self.calc_pos(action))

        if self.lost_tail is not None:
            self.lightpath.append(self.lost_tail)
            self.model.grid.place_agent(self.lost_tail, self.pos)

    def eat_your_tail(self):
        if len(self.lightpath) > self.max_path_length:
            agent = self.lightpath.pop(0)
            self.model.grid.remove_agent(agent)
            return agent

    def get_reward(self, action):
        if self.will_die(action):  # Died
            if self.model.alive_agents == 1:  # Won game
                return 500
            else:
                return -500  # Lost game

        return 10  # Going forward

        # Killed enemy ???
