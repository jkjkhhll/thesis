# #%%
# import torch


# y = torch.rand([2, 5])
# y

# # %%

# from gridworld import Gridworld

# g = Gridworld("12x12_5coin_walls")

# vec = torch.tensor(g.to_vector(), dtype=float)
# vec

# #%%

# from torch.nn.functional import normalize


# normalize(vec, dim=0)


#%%
import time
import os
from gridworld.gym import GridworldGymEnv
from stable_baselines3 import PPO

models_dir = "models/PPO"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logs_dir = "logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

SAVE_EVERY = 10000

env = GridworldGymEnv(
    "12x12_5coin_walls",
    max_steps=1000,
    randomize_agent_positions=True,
    randomly_remove_coins=True,
    hide_agent2=False,
)
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)

i = 0
while True:
    i += 1
    model.learn(
        total_timesteps=SAVE_EVERY, reset_num_timesteps=False, tb_log_name="PPO"
    )
    model.save(f"{models_dir}/{SAVE_EVERY*i}")


# episodes = 10

# for ep in range(episodes):
#     obs = env.reset()

#     done = False
#     while not done:
#         action, _state = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
#         env.render()
#         time.sleep(0.5)

# env.close()
