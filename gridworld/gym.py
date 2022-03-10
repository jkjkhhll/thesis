import gym
from gridworld.rendering import PygameRenderer
from gridworld import Gridworld


class GridworldGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        gridworld_name,
        randomize_agent_positions=False,
        coin_reward=100,
        finish_reward=200,
        step_cost=1,
        wall_hit_cost=5,
        max_steps=None,
        render_delay=0.5,
    ):
        self.gridworld_name = gridworld_name
        self.coin_reward = coin_reward
        self.finish_reward = finish_reward
        self.step_cost = step_cost
        self.wall_hit_cost = wall_hit_cost
        self.max_steps = max_steps
        self.render_delay = render_delay
        self.gridworld = Gridworld(
            gridworld_name, randomize_agent_positions=randomize_agent_positions
        )

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, self.gridworld.vector_size), dtype="uint8"
        )

        self.steps = 0
        self.screen = None

    def step(self, action: int):
        hit_coin, hit_wall, finished = self.gridworld.agent1_action(action)
        if hit_coin:
            reward = self.coin_reward - self.step_cost
        else:
            reward = -self.step_cost

        if hit_wall:
            reward = -self.wall_hit_cost

        self.steps += 1
        if self.max_steps and self.steps >= self.max_steps:
            finished = True

        if finished:
            reward += self.finish_reward
            self.reset()

        return self.gridworld.to_vector(), reward, finished, {}

    def reset(self):
        self.gridworld.reset()
        self.steps = 0
        return self.gridworld.to_vector()

    def render(self):
        if not self.screen:
            self.screen = PygameRenderer()

        self.screen.render(self.gridworld)
