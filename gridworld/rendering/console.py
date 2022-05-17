from gridworld import Gridworld
from gridworld.rendering import Renderer
from gridworld.legend import WALL, AGENT, AGENT2


class ConsoleRenderer(Renderer):
    def _map_object(self, i):
        if i == 0:
            return " "
        if i == WALL:
            return "#"
        if i == AGENT:
            return "A"
        if i == AGENT2:
            return "B"
        return str(i)

    def render(self, gridworld: Gridworld):
        arr = gridworld.to_array()
        for row in arr:
            row = map(lambda i: self._map_object(i), row)
            print(".".join(row))
        print()

    def close(self):
        pass
