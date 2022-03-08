#%%
from tsp import manhattan, shortest_route

# Agent interface
class Agent:
    def step(state):
        pass


class GreedyAgent(Agent):
    def __init__(self):
        self.next_target = None

    def find_nearest(self, pos, coins):
        nearest = None
        nearest_dist = 1000
        for c in coins:
            if c:
                dist = manhattan(pos, c)
                if dist < nearest_dist:
                    nearest = c
                    nearest_dist = dist
        return nearest

    def next_action(self, pos):
        v_dist = self.next_target[0] - pos[0]
        h_dist = self.next_target[1] - pos[1]

        if abs(h_dist) >= abs(v_dist):
            if h_dist > 0:
                action = 2
            else:
                action = 0
        else:
            if v_dist > 0:
                action = 1
            else:
                action = 3
        return action

    def step(self, coingame):
        obs = coingame.to_observation()
        pos = obs[0]
        coins = obs[1:]

        if not self.next_target:
            self.next_target = self.find_nearest(pos, coins)

        if self.next_target in coins:
            action = self.next_action(pos)
        else:
            self.next_target = self.find_nearest(pos, coins)
            action = self.next_action(pos)

        return action


class OptimalAgent(Agent):
    def __init__(self):
        self.next_target = None
