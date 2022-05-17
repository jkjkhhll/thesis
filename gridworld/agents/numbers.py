from gridworld import Gridworld
from gridworld.agents import Agent
from gridworld.utils.bfs import bfs


class NumbersAgent(Agent):
    def __init__(self, flip_agents=False):
        self.target_coin_position = None
        self.current_path = None
        self.last_position = None
        self.last_action = None
        self.flip_agents = flip_agents

    def _update_target(self, g):
        start = Gridworld.from_vector(g.name, g.to_vector())
        start.hide_agent2 = True
        target_coin, self.current_path = bfs(
            start,
        )
        self.target_coin_position = g.coin_positions[target_coin - 1]

    def step(self, gridworld: Gridworld):
        if (
            not self.target_coin_position
            or self.target_coin_position not in gridworld.coin_positions
        ):
            # No target yet or target taken
            self._update_target(gridworld)

        # Check if agent position has changed (for collisions with other player)
        if self.last_position == gridworld.agent1_position:
            return self.last_action
        else:
            action = self.current_path.pop(0)
            self.last_action = action
            self.last_position = gridworld.agent1_position
            return action

    def reset(self):
        self.last_action = None
        self.last_position = None
        self.target_coin_position = None
        self.current_path = None
