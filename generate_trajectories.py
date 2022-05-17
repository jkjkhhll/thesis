#%%
from gridworld.agents.greedy import GreedyAgent
from gridworld.agents.qtable import QTableAgent
from gridworld import imagetools
from gridworld.rendering.pygame import PygameRenderer
from gridworld.trajectories import generate_trajectories
import numpy as np
import random

# a = QTableAgent("saved_models/qtable/12x12_5coin_walls.1203.1043.pickle")
a = QTableAgent("saved_models/qtable/12x12_5coin.1803.1500.pickle")
# a = GreedyAgent()
r = PygameRenderer(delay=5)

#%%
# Generate trajectories

generate_trajectories(
    "12x12_5coin_walls",
    1_000_000,
    a,
    "saved_trajectories/12x12_5coin_walls_qtable_1m.npz",
    progress_bar=True,
)


#%%
# Load and show trajectories

trajs = np.load("saved_trajectories/12x12_5coin_walls_qtable_1m.npz")
labels = trajs["labels"]

for _ in range(5):
    n = random.randint(0, 1_000_000)
    traj = trajs[str(n)]
    print(traj)
    print(labels[n])
    img = imagetools.build_trajectory_image("12x12_5coin_walls", traj)
    r.render_image(img)
