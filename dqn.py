# Hand-coded DQN

import torch
from torch.autograd import Variable
import random
from gridworld.gym import GridworldGymEnv
from tqdm import tqdm
from collections import deque
import matplotlib as plt

GPU = False

if GPU:
    device = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    device = torch.device("cpu")


class DQN:
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_action),
        )
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, s, y):
        y_pred = self.model(torch.Tensor(s).to(device))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        with torch.no_grad():
            return self.model(torch.Tensor(s))

    def replay(self, memory, replay_size, gamma):
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            td_targets = []
            for state, action, next_state, reward, is_done in replay_data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                td_targets.append(q_values)

            self.update(states, td_targets)


def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()

    return policy_function


def q_learning(
    env,
    estimator,
    n_episodes,
    replay_size,
    print_every=100,
    render=False,
    render_every=100,
    gamma=1.0,
    epsilon=0.1,
    epsilon_decay_start=0,
    epsilon_decay_end=49500,
):
    epsilon_decay_value = epsilon / (epsilon_decay_end - epsilon_decay_start)
    for episode in tqdm(range(n_episodes)):
        policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            total_ep_reward[episode] += reward
            total_ep_steps[episode] += 1

            if render and episode % render_every == 0 and not episode == 0:
                env.render()

            memory.append((state, action, next_state, reward, done))
            if done:
                break

            estimator.replay(memory, replay_size, gamma)
            state = next_state

        if (episode + 1) % print_every == 0 and not episode == 0:
            reward_window = total_ep_reward[
                episode - print_every + 1 : episode + print_every + 2
            ]
            step_window = total_ep_steps[
                episode - print_every + 1 : episode + print_every + 2
            ]

            reward_avg = sum(reward_window) / print_every
            reward_min = max(reward_window)
            reward_max = min(reward_window)

            step_avg = sum(step_window) / print_every
            step_min = min(step_window)
            step_max = max(step_window)

            print(f"Ep: {episode + 1}, epsilon: {epsilon:.4f}")
            print(f"Reward min: {reward_min}, max: {reward_max}, avg: {reward_avg:.4f}")
            print(f"Steps min: {step_min}, max: {step_max}, avg: {step_avg:.4f}")

        if epsilon_decay_start <= episode <= epsilon_decay_end:
            # epsilon = max(epsilon * epsilon_decay, 0.01)
            epsilon = max(epsilon - epsilon_decay_value, 0.01)


env = GridworldGymEnv(
    "12x12_5coin_walls",
    randomize_agent_positions=True,
    randomly_remove_coins=True,
    max_steps=1000,
)
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
n_hidden = 50
lr = 0.001
dqn = DQN(n_state, n_action, n_hidden, lr)

memory = deque(maxlen=10000)

n_episodes = 1000
epsilon = 0.7
decay_end = 950
replay_size = 25

total_ep_reward = [0] * n_episodes
total_ep_steps = [0] * n_episodes

q_learning(
    env,
    dqn,
    n_episodes,
    replay_size,
    gamma=0.9,
    epsilon=epsilon,
    epsilon_decay_end=decay_end,
)

plt.plot(total_ep_reward)
plt.show()
