import random

from Agents.RandomAgent import RandomAgent
from DQN.FeatureExtractor import FeatureExtractor


class ReflexAgent(RandomAgent):

    def __init__(self, unique_id, pos, model, max_path_length):
        super().__init__(unique_id, pos, model, max_path_length)
        self.last_action = random.choice(self.get_legal_actions())
        self.max_reward_threshold = self.model.grid.width * self.model.grid.height * 0.5

    def choose_action(self, legal_actions):
        # FeatureExtractor.get_state(self)
        if not self.will_die(self.last_action):
            return self.last_action

        random.shuffle(legal_actions)
        action = random.choice(legal_actions[0])

        for a in legal_actions:
            if not self.will_die(a):
                action = a

        return action
