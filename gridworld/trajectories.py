from gridworld.agents import Agent
from gridworld import Gridworld
import numpy as np


def generate_trajectory(g, agent: Agent):
    done = False
    g.reset()
    agent.reset()

    print(g.coin_positions)
    action = agent.step(g)
    first_action = action
    step = np.array(g.to_vector() + [action])
    steps = []
    steps.append(step)

    coin = None
    while not done:
        coin, _, _ = g.agent1_action(action)
        if coin:
            break

        action = agent.step(g)
        steps.append(g.to_vector() + [action])

    return np.array(steps), coin + 1, first_action


def generate_trajectories(gridworld_name, n: int, agent: Agent):
    g = Gridworld(gridworld_name, randomize_agent_positions=True)

    trajs = []
    for _ in range(n):
        traj, coin, first_action = generate_trajectory(g, agent)
        trajs.append([traj, np.array([coin, first_action])])

    return np.array(trajs)
