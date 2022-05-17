from collections import deque, defaultdict
from gridworld import Gridworld


def _state_to_string(state):
    s = ""
    for i in state:
        s += f"{int(i):02}"
    return s


def _string_to_state(s):
    return [int(s[i : i + 2]) for i in range(0, len(s), 2)]


def bfs(g: Gridworld, preferred_coin=None, actions_only=True):
    visited = set()
    queue = deque()
    start_state = _state_to_string(g.to_vector())

    queue.appendleft(start_state)
    visited.add(start_state)

    # Parent information is a tuple (parent, action)
    parent = defaultdict(lambda: None)

    found_state = None
    last_action = None

    coin = None
    while not len(queue) == 0:
        current_state_str = queue.pop()
        current_state_vec = _string_to_state(current_state_str)

        visited.add(current_state_str)

        for action in range(4):
            new_state = Gridworld.from_vector(g.name, current_state_vec)
            coin, _, _ = new_state.agent1_action(action)
            new_state_str = _state_to_string(new_state.to_vector())

            if coin:
                if preferred_coin and not coin == preferred_coin:
                    # Found coin but not the preferred one
                    coin = False
                else:
                    last_action = action
                    found_state = current_state_str
                    break

            if not new_state_str in visited:
                queue.appendleft(new_state_str)
                parent[new_state_str] = (current_state_str, action)

        if coin:
            break

    path = deque()

    if actions_only:
        path.appendleft(last_action)
    else:
        path.appendleft((found_state, last_action))

    current = None
    if parent[found_state]:
        current = parent[found_state]

    while current:
        if actions_only:
            path.appendleft(current[1])
        else:
            path.appendleft(current)

        current = parent[current[0]]

    return coin, list(path)
