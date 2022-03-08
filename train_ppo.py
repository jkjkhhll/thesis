import gym
from coingame_env import CoingameEnv

from stable_baselines3 import DQN

env = CoingameEnv()
env.reset()

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

obs = env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
