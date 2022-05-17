#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gridworld import Gridworld
from gridworld.rendering.console import ConsoleRenderer
from gridworld.trajectories import TrajectoryDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


test_dataset = TrajectoryDataset("saved_trajectories/12x12_5coin_walls_greedy_10k.npz")
train = DataLoader(test_dataset, batch_size=2, shuffle=True, collate_fn=collate_batch)

data, target = iter(train).next()

c = ConsoleRenderer()
for state in data[0]:
    g = Gridworld.from_vector("12x12_5coin_walls", state[:-1])
    c.render(g)

print(target[0])
