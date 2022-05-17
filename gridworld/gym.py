#%%
import gym
from gridworld.rendering import PygameRenderer
from gridworld import Gridworld
import torch
from torch.nn.functional import normalize


class GridworldGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        gridworld_name,
        randomize_agent_positions=False,
        hide_agent2=False,
        randomly_remove_coins=False,
        limit_agent_view=True,
        normalize_observation=False,
        coin_reward=0.1,
        finish_reward=0.5,
        step_cost=0.001,
        obstacle_hit_cost=0.005,
        max_steps=None,
        render_delay=0.5,
    ):
        self.gridworld_name = gridworld_name
        self.hide_agent2 = hide_agent2
        self.limit_agent_view = limit_agent_view
        self.normalize_observation = normalize_observation

        self.coin_reward = coin_reward
        self.finish_reward = finish_reward
        self.step_cost = step_cost
        self.obstacle_hit_cost = obstacle_hit_cost

        self.max_steps = max_steps
        self.render_delay = render_delay

        self.gridworld = Gridworld(
            gridworld_name,
            randomize_agent_positions=randomize_agent_positions,
            randomly_remove_coins=randomly_remove_coins,
            hide_agent2=hide_agent2,
        )

        self.action_space = gym.spaces.Discrete(4)
        if limit_agent_view:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(27,), dtype="uint8"
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=144, shape=(69,), dtype="uint8"
            )

        self.steps = 0
        self.screen = None

    def step(self, action: int):
        hit_coin, hit_obstacle, finished = self.gridworld.agent1_action(action)
        if hit_coin:
            reward = self.coin_reward - self.step_cost
        else:
            reward = -self.step_cost

        if hit_obstacle:
            reward = -self.obstacle_hit_cost

        self.steps += 1
        if self.max_steps and self.steps >= self.max_steps:
            finished = True

        if finished:
            reward += self.finish_reward

        if self.limit_agent_view:
            return self.gridworld.get_agent_view(as_vector=True), reward, finished, {}
        else:
            if self.normalize_observation:
                return normalize(
                    torch.tensor(self.gridworld.to_tight_vector(), dtype=float), dim=0
                )
            else:
                return self.gridworld.to_tight_vector(), reward, finished, {}

    def reset(self):
        self.gridworld.reset()
        self.steps = 0

        if self.limit_agent_view:
            return self.gridworld.get_agent_view(as_vector=True)
        else:
            if self.normalize_observation:
                return normalize(
                    torch.tensor(self.gridworld.to_tight_vector(), dtype=float), dim=0
                )
            else:
                return self.gridworld.to_tight_vector()

    def render(self):
        if not self.screen:
            self.screen = PygameRenderer()

        self.screen.render(self.gridworld)
