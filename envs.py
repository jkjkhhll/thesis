import gym
from coingame import Coingame
from rendering import PygameRenderer

# COIN_REWARD = 0.1
# STEP_COST = 0.001

COIN_REWARD = 100
FINISH_REWARD = 1000
STEP_COST = 1


SIZE = 12

LIMIT_STEPS = False
MAX_STEPS = 10000


class CoingameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_delay=0.5):
        self.render_delay = render_delay
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=9, shape=(SIZE, SIZE), dtype="uint8"
        )
        self.steps = 0
        self.screen = None

    def step(self, action):
        got_coin, finished = self.coingame.action(action)
        if got_coin:
            reward = COIN_REWARD - STEP_COST
        else:
            reward = -STEP_COST

        self.steps += 1
        if self.steps >= MAX_STEPS and LIMIT_STEPS:
            finished = True

        if finished:
            reward = FINISH_REWARD
            self.reset()

        return self.coingame.get_state(), reward, finished, {}

    def reset(self):
        self.coingame = Coingame(n_rows=SIZE, n_cols=SIZE)
        self.steps = 0
        return self.coingame.get_state()

    def render(self):
        if not self.screen:
            screen_size = self.coingame.to_image().shape
            self.screen = PygameRenderer(
                screen_size[0], screen_size[1], self.render_delay
            )

        self.screen.render(self.coingame)
