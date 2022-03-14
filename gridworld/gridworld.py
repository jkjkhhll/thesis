#%%
import random
import itertools
import numpy as np
from gridworld.agents import Agent
from gridworld.legend import AGENT, AGENT2, WALL


class Gridworld:
    def __init__(
        self,
        name: str,
        agent2: Agent = None,
        hide_agent2=False,
        randomize_agent_positions=False,
        randomly_remove_coins=True,
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
        self.hide_agent2 = hide_agent2

        if self.hide_agent2:
            self.agent2 = None
            self.agent2_position = None
        else:
            self.agent2 = agent2

        self.all_squares = list(
            itertools.product(range(self.n_rows), range(self.n_cols))
        )

        self.startstate = {
            "coin_positions": coin_positions.copy(),
            "coins_left": self.coins_left,
            "agent1_position": agent1_position,
            "agent2_position": agent2_position,
        }

        self.randomize_agent_positions = randomize_agent_positions
        if self.randomize_agent_positions:
            self._randomize_agents()

        self.randomly_remove_coins = randomly_remove_coins

        if self.randomly_remove_coins:
            self._randomly_remove_coins()

        move_up = lambda pos: (pos[0], max(pos[1] - 1, 0))
        move_down = lambda pos: (pos[0], min(pos[1] + 1, self.n_cols - 1))
        move_left = lambda pos: (max(pos[0] - 1, 0), pos[1])
        move_right = lambda pos: (min(pos[0] + 1, self.n_rows - 1), pos[1])

        self.action_defs = {0: move_up, 1: move_right, 2: move_down, 3: move_left}

    def agent1_action(self, action: int):
        hit_coin = None
        hit_obstacle = False

        new_position = self.action_defs[action](self.agent1_position)

        # Hit obstacle (wall or other agent)
        if new_position in self.wall_positions + [self.agent2_position]:
            hit_obstacle = True
        else:
            self.agent1_position = new_position

        # Hit a coin
        if self.agent1_position in self.coin_positions:
            coin = self.coin_positions.index(self.agent1_position) + 1
            self.coin_positions[self.coin_positions.index(self.agent1_position)] = None
            self.coins_left -= 1
            hit_coin = coin

        if self.agent2:
            self._agent2_action()

        # Are all coins collected
        if self.coins_left == 0:
            return hit_coin, hit_obstacle, True
        else:
            return hit_coin, hit_obstacle, False

    def _agent2_action(self):
        new_position = self.action_defs[self.agent2.step(self)](self.agent2_position)

        if new_position not in self.wall_positions + [self.agent1_position]:
            self.agent2_position = new_position

        # Hit a coin
        if self.agent2_position in self.coin_positions:
            coin = self.coin_positions.index(self.agent2_position) + 1
            self.coin_positions[self.coin_positions.index(self.agent2_position)] = None
            self.coins_left -= 1

    def get_agent_view(self, as_vector=False):
        arr = self.to_array()

        # Pad with zeros to always get a 5 x 5 window
        padded = np.zeros((self.n_rows + 2, self.n_cols + 2), dtype="uint8")
        padded[1 : self.n_rows + 1, 1 : self.n_cols + 1] = arr

        r = self.agent1_position[1] + 1
        c = self.agent1_position[0] + 1

        r1 = max(r - 2, 0)
        r2 = min(r + 3, padded.shape[0])
        c1 = max(c - 2, 0)
        c2 = min(c + 3, padded.shape[1])

        view = padded[r1:r2, c1:c2]
        if as_vector:
            return np.concatenate((view.flatten(), list(self.agent1_position)), axis=0)
        else:
            return view

    def to_array(self):
        arr = np.zeros((self.n_rows, self.n_cols), dtype="uint8")

        for i, coin in enumerate(self.coin_positions):
            if coin:
                arr[coin[1], coin[0]] = i + 1

        for col, row in self.wall_positions:
            arr[row, col] = WALL

        arr[self.agent1_position[1], self.agent1_position[0]] = AGENT
        if not self.hide_agent2:
            arr[self.agent2_position[1], self.agent2_position[0]] = AGENT2

        return arr

    def to_vector(self):
        return self.to_array().flatten()

    def to_tight_vector(self):
        vec = []
        vec.append(self.agent1_position[1] * self.n_cols + self.agent1_position[0])

        for c in self.coin_positions:
            if not c:
                vec.append(0)
            else:
                vec.append(c[1] * self.n_cols + c[0])

        for w in self.wall_positions:
            vec.append(w[1] * self.n_cols + w[0])

        return np.array(vec)

    @staticmethod
    def from_vector(name, vec):
        g = Gridworld(name)
        g.coin_positions = [None] * g.n_coins
        g.wall_positions = []
        g.agent1_position = None
        g.agent2_position = None

        for i, n in enumerate(vec):
            if n == AGENT:
                g.agent1_position = (i % g.n_cols, i // g.n_cols)
                continue
            if n == AGENT2:
                g.agent2_position = (i % g.n_cols, i // g.n_cols)
                continue
            if n == WALL:
                g.wall_positions.append((i % g.n_cols, i // g.n_cols))
                continue
            if n != 0:
                g.coin_positions[n - 1] = (i % g.n_cols, i // g.n_cols)

        return g

    def reset(self):
        self.coin_positions = self.startstate["coin_positions"].copy()
        self.coins_left = self.startstate["coins_left"]
        self.agent1_position = self.startstate["agent1_position"]
        if self.agent2:
            self.agent2_position = self.startstate["agent2_position"]

        if self.randomize_agent_positions:
            self._randomize_agents()

        if self.randomly_remove_coins:
            self._randomly_remove_coins()

    def _randomize_agents(self):
        possible_starts = [
            s
            for s in self.all_squares
            if s not in self.coin_positions + self.wall_positions
        ]
        self.agent1_position = random.sample(possible_starts, 1)[0]
        possible_starts.remove(self.agent1_position)

        if not self.hide_agent2:
            self.agent2_position = random.sample(possible_starts, 1)[0]

    def _randomly_remove_coins(self):
        remove = random.sample(self.coin_positions, random.randrange(self.n_coins - 1))
        for c in remove:
            self.coin_positions[self.coin_positions.index(c)] = None
            self.coins_left = self.n_coins - len(remove)
