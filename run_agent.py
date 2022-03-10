from re import I
from gridworld import Gridworld
from gridworld.agents import QTableAgent
from gridworld.rendering import PygameRenderer

a = QTableAgent("saved_models/qtable/12x12_5coin_walls.1003.0925.pickle")
r = PygameRenderer(delay=0.7)

env = Gridworld("12x12_5coin_walls", randomize_agent_positions=True)
done = False

for _ in range(10):
    env.reset()
    steps = 0
    done = False
    while not done:
        action = a.step(env)
        _, _, done = env.agent1_action(action)
        r.render(env)
        steps += 1
    print(steps)
