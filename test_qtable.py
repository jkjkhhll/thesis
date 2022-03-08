from envs import CoingameEnv
import pickle
import numpy as np


def state_to_string(state):
    s = ""
    for i in state.flatten():
        s += str(int(i))
    return s


def q_values(state):
    ss = state_to_string(state)
    return q_table[ss]


with open("qtable.pickle", "rb") as f:
    q_table = pickle.load(f)

env = CoingameEnv()
state = env.reset()

done = False
steps = 0
while not done:

    action = np.argmax(q_values(state))

    state, reward, done, _ = env.step(action)

    steps += 1
    env.render()

print(steps)
