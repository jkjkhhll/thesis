#%%
from gridworld.agents import QTableAgent
from gridworld import imagetools
from gridworld.rendering import PygameRenderer
from gridworld.trajectories import generate_trajectories

a = QTableAgent("saved_models/qtable/12x12_5coin_walls.1003.0925.pickle")
r = PygameRenderer(delay=2)

trajs = generate_trajectories("12x12_5coin_walls", 10, a)

for traj in trajs:
    print(traj)
    img = imagetools.build_trajectory_image("12x12_5coin_walls", traj)
    r.render_image(img)
