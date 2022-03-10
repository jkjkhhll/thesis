import random
import itertools
from re import I
import numpy as np
from gridworld.agents import Agent


class Gridworld:

    AGENT = 10
    AGENT2 = 11
    WALL = 12

    def __init__(
        self, name: str, other_agent: Agent = None, randomize_agent_positions=False
    ):
        n_rows = 0
        n_cols = 0
        coin_positions = []
        wall_positions = []

        agent1_position = None
        agent2_position = None

        with open(f"gridworld/env_definitions/{name}.grid") as f:
            for l in f:
                row = l.strip().split("|")[1:-1]
                n_cols = len(row)
                for col, ch in enumerate(row):
                    pos = (col, n_rows)
                    if ch == "A":
                        agent1_position = pos
                    if ch == "B":
                        agent2_position = pos
                    if ch == "O":
                        coin_positions.append(pos)
                    if ch == "#":
                        wall_positions.append(pos)
                n_rows += 1

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.vector_size = n_rows * n_cols

        self.coin_positions = coin_positions
        self.n_coins = len(coin_positions)
        self.coins_left = self.n_coins

        self.wall_positions = wall_positions

        self.agent1_position = agent1_position
        self.agent2_position = agent2_position

        self.all_squares = list(
            itertools.product(range(self.n_rows), range(self.n_cols))
        )

        self.randomize_agent_positions = randomize_agent_positions
        if self.randomize_agent_positions:
            self._randomize_agents()

        self.startstate = {
            "coin_positions": coin_positions.copy(),
            "coins_left": self.coins_left,
            "agent1_position": agent1_position,
            "agent2_position": agent2_position,
        }

        move_up = lambda pos: (pos[0], max(pos[1] - 1, 0))
        move_down = lambda pos: (pos[0], min(pos[1] + 1, self.n_cols - 1))
        move_left = lambda pos: (max(pos[0] - 1, 0), pos[1])
        move_right = lambda pos: (min(pos[0] + 1, self.n_rows - 1), pos[1])

        self.action_defs = {0: move_up, 1: move_right, 2: move_down, 3: move_left}

    def agent1_action(self, action: int):
        hit_coin = None
        hit_wall = False

        new_position = self.action_defs[action](self.agent1_position)

        # Hit a wall
        if new_position in self.wall_positions:
            hit_wall = True
        else:
            self.agent1_position = new_position

        # Hit a coin
        if self.agent1_position in self.coin_positions:
            coin = self.coin_positions.index(self.agent1_position) + 1
            self.coin_positions[self.coin_positions.index(self.agent1_position)] = None
            self.coins_left -= 1
            hit_coin = coin

        # Are all coins collected
        if self.coins_left == 0:
            return hit_coin, hit_wall, True
        else:
            return hit_coin, hit_wall, False

    def to_vector(self):
        vec = [0] * (self.n_cols * self.n_rows)
        for i, c in enumerate(self.coin_positions):
            if c:
                vec[c[1] * self.n_cols + c[0]] = i + 1

        for w in self.wall_positions:
            vec[w[1] * self.n_cols + w[0]] = self.WALL

        vec[
            self.agent1_position[1] * self.n_cols + self.agent1_position[0]
        ] = self.AGENT
        if self.agent2_position:
            vec[
                self.agent2_position[1] * self.n_cols + self.agent2_position[0]
            ] = self.AGENT2

        return vec

    @staticmethod
    def from_vector(name, vec):
        g = Gridworld(name)
        g.coin_positions = [None] * g.n_coins
        g.wall_positions = []
        g.agent1_position = None
        g.agent2_position = None

        for i, n in enumerate(vec):
            if n == Gridworld.AGENT:
                g.agent1_position = (i % g.n_cols, i // g.n_cols)
                continue
            if n == Gridworld.AGENT2:
                g.agent2_position = (i % g.n_cols, i // g.n_cols)
                continue
            if n == Gridworld.WALL:
                g.wall_positions.append((i % g.n_cols, i // g.n_cols))
                continue
            if n != 0:
                g.coin_positions[n - 1] = (i % g.n_cols, i // g.n_cols)

        return g

    def reset(self):
        self.coin_positions = self.startstate["coin_positions"].copy()
        self.coins_left = self.startstate["coins_left"]
        self.agent1_position = self.startstate["agent1_position"]
        self.agent2_position = self.startstate["agent2_position"]

        if self.randomize_agent_positions:
            self._randomize_agents()

    def _randomize_agents(self):
        possible_starts = [
            s
            for s in self.all_squares
            if s not in self.coin_positions + self.wall_positions
        ]
        self.agent1_position = random.sample(possible_starts, 1)[0]
        possible_starts.remove(self.agent1_position)
        if self.agent2_position:
            self.agent2_position = random.sample(possible_starts, 1)[0]
