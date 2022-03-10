import matplotlib.pyplot as plt
import numpy as np
from gridworld.gym import GridworldGymEnv
import random
import pickle
from tqdm import tqdm
from datetime import datetime

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000
RENDER = True

RENDER_EVERY = 1000
STATS_EVERY = 1000

START_EPSILON = 0.5
START_DECAY = 1
END_DECAY = EPISODES - EPISODES // 4
ENV = "12x12_5coin"

# LEARNING_RATE = 0.1
# DISCOUNT = 0.95
# EPISODES = 50000
# RENDER = True
# RENDER_EVERY = 1000
# START_EPSILON = 1
# START_DECAY = 1
# END_DECAY = EPISODES - EPISODES // 4

epsilon = START_EPSILON
DECAY_VALUE = epsilon / (END_DECAY - START_DECAY)

env = GridworldGymEnv(
    ENV, randomize_agent_positions=True, max_steps=1000, render_delay=0.5
)

# "12x12_5coin", max_steps=1000, render_delay=0.5, wall_hit_cost=0.3, finish_reward=2

done = False
steps = 0

q_table = {}

ep_rewards = []
ep_steps = []

aggr_ep_rewards = {"ep": [], "avg": [], "min": [], "max": []}
aggr_ep_steps = {"ep": [], "avg": [], "min": [], "max": []}


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
        episode_reward += reward

        steps += 1
        if RENDER and episode % RENDER_EVERY == 0 and episode != 0:
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

    # epsilon = epsilon * 0.99

    if epsilon < 0:
        epsilon = 0

    ep_rewards.append(episode_reward)
    ep_steps.append(steps)

    if episode % STATS_EVERY == 0 and episode != 0:
        average_reward = sum(ep_rewards[-STATS_EVERY:]) / len(ep_rewards[-STATS_EVERY:])

        aggr_ep_rewards["ep"].append(episode)
        aggr_ep_rewards["avg"].append(average_reward)
        aggr_ep_rewards["min"].append(min(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards["max"].append(max(ep_rewards[-STATS_EVERY:]))

        average_steps = sum(ep_steps[-STATS_EVERY:]) / len(ep_steps[-STATS_EVERY:])

        aggr_ep_steps["ep"].append(episode)
        aggr_ep_steps["avg"].append(average_steps)
        aggr_ep_steps["min"].append(min(ep_steps[-STATS_EVERY:]))
        aggr_ep_steps["max"].append(max(ep_steps[-STATS_EVERY:]))

        print(
            f"Ep: {episode}, avg: {average_reward}, min {min(ep_rewards[-STATS_EVERY:])}, max {max(ep_rewards[-STATS_EVERY:])}"
        )
        print(
            f"Steps: {episode}, avg: {average_steps}, min {min(ep_steps[-STATS_EVERY:])}, max {max(ep_steps[-STATS_EVERY:])}"
        )
        print(f"Epsilon: {epsilon}")

# plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"], label="avg")
# plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"], label="min")
# plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"], label="max")


d = datetime.now().strftime("%d%m.%H%M")


params = {
    "env": ENV,
    "learning_rate": LEARNING_RATE,
    "discount": DISCOUNT,
    "episodes": EPISODES,
    "start_epsilon": START_EPSILON,
    "start_decay": START_DECAY,
    "end_decay": END_DECAY,
}

stats = {"rewards": aggr_ep_rewards, "steps": aggr_ep_steps}

data = {"params": params, "stats": stats, "q_table": q_table}

with open(f"saved_models/qtable/{ENV}.{d}.pickle", "wb") as f:
    pickle.dump(data, f)

plt.plot(aggr_ep_steps["ep"], aggr_ep_steps["avg"], label="avg")
plt.legend(loc=4)
plt.show()
