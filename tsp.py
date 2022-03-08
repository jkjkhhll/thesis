#%%
import itertools
import math


def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def route_length(route):
    length = 0
    for i in range(1, len(route)):
        length += manhattan(route[i - 1], route[i])
    return length


# Brute force
def shortest_route(points):
    shortest_route = None
    shortest_route_length = math.inf

    for route in itertools.permutations(points, len(points)):
        l = route_length(route)
        if l < shortest_route_length:
            shortest_route_length = l
            shortest_route = route

    return (shortest_route, shortest_route_length)
