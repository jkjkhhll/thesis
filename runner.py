from coingame import Coingame
from agents import GreedyAgent
from rendering import PygameRenderer


def run_episode(coingame, agent, renderer):
    renderer.render(coingame)

    a = agent.step(coingame)
    finished = coingame.action(a)
    while not finished:
        a = agent.step(coingame)
        finished = coingame.action(a)
        renderer.render(coingame)


c = Coingame()
a = GreedyAgent()
r = PygameRenderer()

run_episode(c, a, r)
