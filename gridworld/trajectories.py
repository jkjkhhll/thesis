from gridworld.agents import Agent
from gridworld import Gridworld
import numpy as np
from tqdm import tqdm
from zipfile import ZipFile
import os
from torch.utils.data import Dataset


def generate_trajectory(g, agent: Agent):
    done = False
    g.reset()
    agent.reset()

    action = agent.step(g)
    first_action = action
    step = np.concatenate((g.to_vector(), [action]))
    steps = []
    steps.append(step)

    coin = None
    while not done:
        coin, _, _ = g.agent1_action(action)

        if coin:
            break

        action = agent.step(g)
        steps.append(np.concatenate((g.to_vector(), [action])))

    return np.array(steps), coin, first_action


def generate_trajectories(
    gridworld_name, n: int, agent: Agent, outfile, progress_bar=False
):
    g = Gridworld(
        gridworld_name,
        randomize_agent_positions=True,
        # hide_agent2=True,
        randomly_remove_coins=True,
    )

    TEMPFILE = "temp.npy"
    labels = []

    if progress_bar:
        n_traj = tqdm(range(n))
    else:
        n_traj = range(n)

    with ZipFile(outfile, "w") as z:
        for i in n_traj:
            traj, coin, first_action = generate_trajectory(g, agent)
            np.save(TEMPFILE, traj)
            z.write(TEMPFILE, arcname=str(i))
            os.remove(TEMPFILE)
            labels.append(np.array([coin, first_action]))

        labels = np.array(labels)
        np.save(TEMPFILE, labels)
        z.write(TEMPFILE, arcname="labels")
        os.remove(TEMPFILE)


class TrajectoryDataset(Dataset):
    def __init__(self, dataset_file):
        self.dataset = np.load(dataset_file)
        self.labels = self.dataset["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.dataset[str(idx)], self.labels[idx]
