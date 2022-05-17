from abc import ABC, abstractmethod

# Agent interface
class Renderer(ABC):
    @abstractmethod
    def render(self, gridworld):
        pass

    @abstractmethod
    def close(self):
        pass
