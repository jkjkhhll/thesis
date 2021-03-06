#%%
from abc import ABC, abstractmethod

# Agent interface
class Agent(ABC):
    @abstractmethod
    def step(self, gridworld):
        pass

    @abstractmethod
    def reset(self):
        pass




        