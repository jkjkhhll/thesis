import gym
import time
from coingame_env import CoingameEnv

# env = gym.make("CartPole-v0")
# env.reset()

# for _ in range(1000):
#     env.render()
#     time.sleep(0.01)
#     env.step(env.action_space.sample())

# env.close()

env = CoingameEnv()
env.reset()


for _ in range(1000):
    env.render()
    time.sleep(0.05)
    env.step(env.action_space.sample())

env.close()
