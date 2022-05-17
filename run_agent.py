#%%
from gridworld import Gridworld
from gridworld.agents.greedy import GreedyAgent
from gridworld.agents.qtable import QTableAgent
from gridworld.rendering.pygame import PygameRenderer

a = QTableAgent("saved_models/qtable/12x12_5coin.1803.1500.pickle")
# a = GreedyAgent()
r = PygameRenderer(delay=0.5)

env = Gridworld(
    "12x12_5coin_walls",
    # agent2=QTableAgent(
    #     "saved_models/qtable/12x12_5coin.1803.1500.pickle", flip_agents=True
    # ),
    hide_agent2=True,
    randomize_agent_positions=True,
    randomly_remove_coins=True,
)
done = False

for _ in range(10):
    a.reset()
    env.reset()
    steps = 0
    done = False
    while not done:
        action = a.step(env)
        print(action)
        _, _, done = env.agent1_action(action)
        r.render(env)
        steps += 1
    print(steps)
