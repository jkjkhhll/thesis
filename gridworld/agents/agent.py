#%%
from tkinter import Grid
from gridworld.tsp import manhattan, shortest_route
from abc import ABC, abstractmethod

# Agent interface
class Agent:
    @abstractmethod
    def step(self, gridworld):
        pass

    @abstractmethod
    def reset(self):
        pass




        