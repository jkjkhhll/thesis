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
# from datetime import datetime
# import os
# from gridworld.gym import GridworldGymEnv
# from stable_baselines3 import PPO


# for _ in range(5):
#     tag = datetime.now().strftime("%d%m.%H%M")

#     model_dir = f"models/PPO_{tag}"
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)

#     logs_dir = "logs"
#     if not os.path.exists(logs_dir):
#         os.makedirs(logs_dir)

#     SAVE_EVERY = 10000

#     env = GridworldGymEnv(
#         "12x12_5coin_walls",
#         max_steps=200,
#         randomize_agent_positions=True,
#         randomly_remove_coins=True,
#         hide_agent2=False,
#         normalize_observation=True,
#     )
#     env.reset()

#     model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)

#     for i in range(1, 1000):
#         model.learn(
#             total_timesteps=SAVE_EVERY,
#             reset_num_timesteps=False,
#             tb_log_name=f"PPO_{tag}",
#         )
#         model.save(f"{model_dir}/{SAVE_EVERY*i}")


# #%%
# from gridworld.gym import GridworldGymEnv
# from stable_baselines3 import PPO
# import time

# env = GridworldGymEnv(
#     "12x12_5coin_walls",
#     max_steps=1000,
#     randomize_agent_positions=True,
#     randomly_remove_coins=True,
#     hide_agent2=False,
#     normalize_observation=True,
# )

# model_file = "models/PPO_1603.0400/9042000.zip"
# model = PPO.load(model_file, env=env)

# episodes = 10
# for ep in range(episodes):
#     obs = env.reset()

#     done = False
#     while not done:
#         action, _ = model.predict(obs)
#         obs, reward, done, info = env.step(action)
#         env.render()
#         time.sleep(0.5)

# env.close()

#%%
# from gridworld import Gridworld
# from gridworld.legend import AGENT, AGENT2, WALL

# HIDDEN = 13


# def map_object(i):
#     if i == 0:
#         return " "
#     if i == WALL:
#         return "#"
#     if i == AGENT:
#         return "A"
#     if i == AGENT2:
#         return "B"
#     if i == HIDDEN:
#         return "-"
#     return str(i)


# def print_state(state_arr):
#     for r in state_arr:
#         row = map(lambda i: map_object(i), r)
#         s = "|".join(row)
#         print(s)
#     print()


# def get_hidden(row, col, a_row, a_col):

#     s = slope(row, col, a_row, a_col)

#     if row == a_row and col == a_col:
#         return []

#     # On same row / col
#     if row == a_row:
#         if a_col < col:
#             return [(row, col + 1), (row + 1, col + 1), (row - 1, col + 1)]
#         return [(row, col - 1), (row + 1, col - 1), (row - 1, col - 1)]

#     if col == a_col:
#         if a_row < row:
#             return [(row + 1, col), (row + 1, col + 1), (row + 1, col - 1)]
#         return [(row - 1, col), (row - 1, col + 1), (row - 1, col - 1)]

#     # Top left
#     if col < a_col and row < a_row:
#         if s == 0:
#             return [(row - 1, col - 1)]
#         if s == 1:
#             return [(row - 1, col - 1), (row, col - 1)]
#         if s == -1:
#             return [(row - 1, col), (row - 1, col - 1)]

#     # Bottom left
#     if col < a_col and row > a_row:
#         if s == 0:
#             return [(row + 1, col - 1)]
#         if s == 1:
#             return [(row, col - 1), (row + 1, col - 1)]
#         if s == -1:
#             return [(row + 1, col), (row + 1, col - 1)]

#     # Top right
#     if col > a_col and row < a_row:
#         if s == 0:
#             return [(row - 1, col + 1)]
#         if s == 1:
#             return [(row - 1, col + 1), (row, col + 1)]
#         if s == -1:
#             return [(row - 1, col), (row - 1, col + 1)]

#     # Bottom right
#     if col > a_col and row > a_row:
#         if s == 0:
#             return [(row + 1, col + 1)]
#         if s == 1:
#             return [(row + 1, col + 1), (row, col + 1)]
#         if s == -1:
#             return [(row + 1, col), (row + 1, col + 1)]


# def slope(row, col, a_row, a_col):
#     dx = abs(row - a_row)
#     dy = abs(col - a_col)

#     if dx == dy:
#         return 0

#     if dx > dy:
#         return 1

#     if dx < dy:
#         return -1


# g = Gridworld(
#     "12x12_5coin_walls", randomize_agent_positions=True, randomly_remove_coins=False
# )
# g_arr = g.to_array()
# print(g_arr.shape)

# a_col = g.agent1_position[0]
# a_row = g.agent1_position[1]
# height = g_arr.shape[0]
# width = g_arr.shape[1]


# def drawline(
#     start_row,
#     start_col,
#     vert_step,
#     horiz_step,
#     vert_min,
#     vert_max,
#     horiz_min,
#     horiz_max,
# ):
#     if horiz_step > 0:
#         hs = 1
#     if horiz_step == 0:
#         hs = 0
#     if horiz_step < 0:
#         hs = -1

#     if vert_step > 0:
#         vs = 1
#     if vert_step == 0:
#         vs = 0
#     if vert_step < 0:
#         vs = -1


# print_state(g_arr)

# hide = []
# #  top -> down
# for row in range(1, height - 1):
#     for col in range(1, width - 1):
#         if g_arr[row, col] == WALL:
#             # Hide up
#             if row < a_row and not g_arr[row - 1, col] == WALL:
#                 hide += [(r, col) for r in range(row - 1, -1, -1)]

#             if row > a_row and not g_arr[row + 1, col] == WALL:
#                 hide += [(r, col) for r in range(row + 1, height)]

#             if col < a_col and not g_arr[row, col - 1] == WALL:
#                 hide += [(row, c) for c in range(col - 1, -1, -1)]

#             if col > a_col and not g_arr[row, col + 1] == WALL:
#                 hide += [(row, c) for c in range(col + 1, width)]
# print(hide)
# for h in hide:
#     g_arr[h[0], h[1]] = HIDDEN

# # bottom -> up
# for row in range(a_row - 1, 0, -1):
#     for col in range(1, g_arr.shape[1] - 1):

# # left - right
# for col in range(a_col + 1, g_arr.shape[1] - 1):
#     for row in range(1, g_arr.shape[0] - 1):

# # right -> left
# for col in range(a_col - 1, 0, -1):
#     for row in range(1, g_arr.shape[0] - 1):


#%%
from time import sleep
from tqdm import tqdm

for _ in tqdm(range(1, 1_000_000)):
    sleep(0.01)
