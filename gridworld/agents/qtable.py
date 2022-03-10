from gridworld.agents import Agent
import numpy as np
import pickle


class QTableAgent(Agent):
    def __init__(self, model_file):
        with open(model_file, "rb") as f:
            data = pickle.load(f)

        self.q_table = data["q_table"]

    def step(self, gridworld):
        state = gridworld.to_vector()
        action = np.argmax(self._q_values(state))
        return action

    def reset(self):
        pass

    def _state_to_string(self, state):
        s = ""
        for i in state:
            s += f"{int(i):02}"
        return s

    def _q_values(self, state):
        ss = self._state_to_string(state)
        if not ss in self.q_table:
            self.q_table[ss] = np.zeros(4)

        return self.q_table[ss]
