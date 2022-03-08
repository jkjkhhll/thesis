#%%
import numpy as np
import random
import itertools
from imagebuilder import ImageBuilder

AGENT = 9
OTHER_AGENT = 8
N_COINS = 8


class Coingame:
    def __init__(self, n_rows=8, n_cols=8, two_agent=False, other_agent=None):

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_coins = 8
        self.two_agent = two_agent

        move_up = lambda pos: (pos[0], max(pos[1] - 1, 0))
        move_down = lambda pos: (pos[0], min(pos[1] + 1, self.n_cols - 1))
        move_left = lambda pos: (max(pos[0] - 1, 0), pos[1])
        move_right = lambda pos: (min(pos[0] + 1, self.n_rows - 1), pos[1])

        self.action_defs = {0: move_up, 1: move_right, 2: move_down, 3: move_left}

        # grid_squares = list(itertools.product(range(n_rows), range(n_cols)))

        # n_objs = 1 + n_coins
        # if two_agent:
        #     n_objs += 1

        # obj_positions = random.sample(grid_squares, n_objs)

        # self.agent_position = obj_positions.pop()
        # if two_agent:
        #     self.other_agent_position = obj_positions.pop()
        # else:
        #     self.other_agent_position = None

        # self.coin_positions = obj_positions

        self.coin_positions = [
            (1, 2),
            (3, 3),
            (7, 4),
            (0, 3),
            (6, 6),
            (8, 10),
            (3, 9),
            (9, 3),
        ]
        self.agent_position = (4, 1)

    def action(self, act):
        got_coin = False

        self.agent_position = self.action_defs[act](self.agent_position)
        if self.agent_position in self.coin_positions:
            self.coin_positions[self.coin_positions.index(self.agent_position)] = None
            self.n_coins -= 1
            got_coin = True

        if self.n_coins == 0:
            return got_coin, True
        else:
            return got_coin, False

    def to_image(self):
        i = ImageBuilder(self.n_rows, self.n_cols)
        return i.get_image(self)

    def to_vector(self):
        vec = [0] * (self.n_cols * self.n_rows)
        for i, c in enumerate(self.coin_positions):
            if c:
                vec[c[1] * self.n_cols + c[0]] = i + 1

        vec[self.agent_position[1] * self.n_cols + self.agent_position[0]] = AGENT
        if self.two_agent:
            vec[
                self.other_agent_position[1] * self.n_cols
                + self.other_agent_position[0]
            ] = OTHER_AGENT

        return vec

    def get_state(self):
        state = np.zeros((self.n_rows, self.n_cols))
        for i, c in enumerate(self.coin_positions):
            if c:
                state[c[1], c[0]] = i + 1
        state[self.agent_position[1], self.agent_position[1]] = 9
        return state

    def to_observation(self):
        return [self.agent_position] + self.coin_positions
