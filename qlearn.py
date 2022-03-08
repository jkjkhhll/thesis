import gym
import matplotlib.pyplot as plt
import numpy as np
from envs import CoingameEnv
import random
import pickle
from tqdm import tqdm
from datetime import datetime

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000
RENDER = False
RENDER_EVERY = 500

epsilon = 0.5
START_DECAY = 1
END_DECAY = EPISODES // 2
DECAY_VALUE = epsilon / (END_DECAY - START_DECAY)

env = CoingameEnv()

done = False
steps = 0

q_table = {}

ep_rewards = []
ep_steps = []
aggr_ep_rewards = {"ep": [], "avg": [], "min": [], "max": []}
aggr_ep_steps = {"ep": [], "avg": [], "min": [], "max": []}


def state_to_string(state):
    s = ""
    for i in state.flatten():
        s += str(int(i))
    return s


def q_values(state):
    ss = state_to_string(state)
    if not ss in q_table:
        # q_table[ss] = np.random.uniform(low=-2, high=0, size=4)
        q_table[ss] = np.random.zeros(size=4)

    return q_table[ss]


state = env.reset()

render = False
episode_progress = tqdm(range(EPISODES))
for episode in episode_progress:

    episode_reward = 0

    if episode % RENDER_EVERY == 0 and episode != 0:
        render = True
    else:
        render = False

    steps = 0
    done = False

    while not done:

        if random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_values(state))

        new_state, reward, done, _ = env.step(action)
        episode_reward = +reward

        steps += 1
        if render and RENDER:
            env.render()

        if not done:

            max_future_q = np.max(q_values(new_state))
            current_q = q_values(state)[action]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q
            )
            q_values(state)[action] = new_q

        state = new_state

    if END_DECAY >= episode >= START_DECAY:
        epsilon -= DECAY_VALUE

    ep_rewards.append(episode_reward)
    ep_steps.append(steps)
    if not episode % RENDER_EVERY and not episode == 0:
        average_reward = sum(ep_rewards[-RENDER_EVERY:]) / len(
            ep_rewards[-RENDER_EVERY:]
        )

        aggr_ep_rewards["ep"].append(episode)
        aggr_ep_rewards["avg"].append(average_reward)
        aggr_ep_rewards["min"].append(min(ep_rewards[-RENDER_EVERY:]))
        aggr_ep_rewards["max"].append(max(ep_rewards[-RENDER_EVERY:]))

        average_steps = sum(ep_steps[-RENDER_EVERY:]) / len(ep_steps[-RENDER_EVERY:])

        aggr_ep_steps["ep"].append(episode)
        aggr_ep_steps["avg"].append(average_steps)
        aggr_ep_steps["min"].append(min(ep_steps[-RENDER_EVERY:]))
        aggr_ep_steps["max"].append(max(ep_steps[-RENDER_EVERY:]))

        print(
            f"Ep: {episode}, avg: {average_reward}, min {min(ep_rewards[-RENDER_EVERY:])}, max {max(ep_rewards[-RENDER_EVERY:])}"
        )
        print(
            f"Steps: {episode}, avg: {average_steps}, min {min(ep_steps[-RENDER_EVERY:])}, max {max(ep_steps[-RENDER_EVERY:])}"
        )

# plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"], label="avg")
# plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"], label="min")
# plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"], label="max")


d = datetime.now().strftime("%d%m.%H%M")
with open(f"qtable.{d}.pickle", "wb") as f:
    pickle.dump(q_table, f)

plt.plot(aggr_ep_steps["ep"], aggr_ep_steps["avg"], label="avg")
plt.legend(loc=4)
plt.show()
