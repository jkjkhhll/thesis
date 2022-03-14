from collections import deque
from gridworld import Gridworld
import numpy as np

def state_to_string(state):
    s = ""
    for i in state:
        s += f"{int(i):02}"
    return s


envname = "12x12_5coin_walls"
g = Gridworld(
    envname,
    hide_agent2=True,
    randomize_agent_positions=True,
    randomly_remove_coins=False,
)


visited = set()
queue = deque()

start_state = g.to_vector().tobytes()
queue.appendleft(start_state)
visited.add(start_state)

parent = {}
parent[start_state] = None


coin_found = False
while not queue.empty():
    current_vec = np.from_buffer(queue.pop, dtype="uint8")
    current_state = Gridworld.from_vector(envname, current_vec)
    if current_state.agent1_position in current_state.coin_positions:

    for action in range(4):
        current_state = 
        parent[current_state] = ( 

