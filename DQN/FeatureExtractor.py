import numpy as np
import tensorflow as tf


class FeatureExtractor:
    action_to_int = {'N': 0, 'S': 1, 'W': 2, 'E': 3}

    @staticmethod
    def get_state(agent):
        grid = agent.model.grid  # coordinate 0,0 is bottom left
        x, y = agent.pos

        n_list = list(range(y + 1, grid.height))
        s_list = list(reversed(range(0, y)))
        w_list = list(reversed(range(0, x)))
        e_list = list(range(x + 1, grid.width))

        obst_dst_n = len(n_list) + 1
        for i in n_list:
            if not grid.is_cell_empty((x, i)):
                obst_dst_n = abs(y - i)
                break

        obst_dst_s = len(s_list) + 1
        for i in s_list:
            if not grid.is_cell_empty((x, i)):
                obst_dst_s = abs(y - i)
                break

        obst_dst_w = len(w_list) + 1
        for i in w_list:
            if not grid.is_cell_empty((i, y)):
                obst_dst_w -= abs(x - i)
                break

        obst_dst_e = len(e_list) + 1
        for i in e_list:
            if not grid.is_cell_empty((i, y)):
                obst_dst_e = abs(x - i)
                break

        obst_dst_n /= grid.height
        obst_dst_s /= grid.height
        obst_dst_w /= grid.width
        obst_dst_e /= grid.width

        # print((x, y), agent.model.alive_agents, obst_dst_n, obst_dst_s, obst_dst_w, obst_dst_e)

        return np.array([agent.model.alive_agents, obst_dst_n, obst_dst_s, obst_dst_w,
                         obst_dst_e]), False

    @staticmethod
    def get_state_a(agent, action):
        if action not in agent.get_legal_actions():
            state, _ = FeatureExtractor.get_state(agent)
            state[FeatureExtractor.action_to_int[action] + 1] = 0
            # print(state)
            return state, True

        agent.simulate_step(action)
        state = FeatureExtractor.get_state(agent)
        agent.reverse_step()

        return state
