import tensorflow as tf


class FeatureExtractor:
    @staticmethod
    def get_state(agent):

        grid = agent.model.grid  # coordinate 0,0 is bottom left
        x, y = agent.pos

        n_list = list(range(y + 1, grid.height))
        s_list = list(reversed(range(0, y)))
        w_list = list(reversed(range(0, x)))
        e_list = list(range(x + 1, grid.width))

        # print(pos, n_list, s_list, w_list, e_list)

        obstacle_distance_N = len(n_list)
        for i in n_list:
            if not grid.is_cell_empty((x, i)):
                obstacle_distance_N -= i
                break

        obstacle_distance_S = len(s_list)
        for i in s_list:
            if not grid.is_cell_empty((x, i)):
                obstacle_distance_S -= i
                break

        obstacle_distance_W = len(w_list)
        for i in w_list:
            if not grid.is_cell_empty((i, y)):
                obstacle_distance_W -= i
                break

        obstacle_distance_E = len(e_list)
        for i in e_list:
            if not grid.is_cell_empty((i, y)):
                obstacle_distance_E -= i
                break

        # print((x, y), obstacle_distance_N, obstacle_distance_S, obstacle_distance_W, obstacle_distance_E)

        return tf.constant([[x, y, obstacle_distance_N, obstacle_distance_S, obstacle_distance_W, obstacle_distance_E]])

    @staticmethod
    def get_state_a(agent, action):
        if action not in agent.get_legal_actions():
            return None

        agent.simulate_step(action)
        state = FeatureExtractor.get_state(agent)
        agent.reverse_step()

        return state
