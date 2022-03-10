from tkinter import Grid
from gridworld.gym import GridworldGymEnv
import pickle
import numpy as np
import random


def state_to_string(state):
    s = ""
    for i in state:
        s += f"{int(i):02}"
    return s


def q_values(state):
    ss = state_to_string(state)
    if not ss in q_table:
        # q_table[ss] = np.random.uniform(low=-2, high=0, size=4)
        q_table[ss] = np.zeros(4)

    return q_table[ss]


with open("models/12x12_5coin_walls.1003.0925.pickle", "rb") as f:
    data = pickle.load(f)

q_table = data["q_table"]
params = data["params"]
print(params)

env = GridworldGymEnv(params["env"], max_steps=50, randomize_agent_positions=True)

for i in range(10):
    state = env.gridworld.to_vector()

    steps = 0
    done = False
    while not done:
        action = np.argmax(q_values(state))

        state, reward, done, _ = env.step(action)

        steps += 1
        env.render()

    print(steps)
    env.reset()
