# Elements of state vector
# 1- x of rocketship
# 2-y of rocketship
# 3-x velocity of rocketship
# 4-y velocity of rocketship
# 5-angle of rocketship
# 6-angular velocity of rocketship
# 7-fuel of rocketship
# 8- x distance from center of landing pad 100 points
# 9- y distance from center of landing pad 100 points
# 10- Contact with ground(0/1)

# --------------------------------
# Actions

#0 → (thrust=False, rotate_left=False, rotate_right=False)    do nothing
#1 → (thrust=True,  rotate_left=False, rotate_right=False)    thrust only
#2 → (thrust=False, rotate_left=True,  rotate_right=False)    rotate left only
#3 → (thrust=False, rotate_left=False, rotate_right=True)     rotate right only
#4 → (thrust=True,  rotate_left=True,  rotate_right=False)    thrust + rotate left
#5 → (thrust=True,  rotate_left=False, rotate_right=True)     thrust + rotate right

# --------------------------------
# Libraries

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lander_env import LanderEnv, STATE_SIZE, NUM_ACTIONS

# ---------------------------------
# Hyperparameters

GAMMA = 0.99
LEARNING_RATE = 1e-3          
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995
BUFFER_SIZE = 100_000
BATCH_SIZE = 64
NUM_EPISODES = 500
MAX_STEPS = 1000
TARGET_UPDATE_EVERY = 500     
TRAIN_EVERY = 1               


# ---------------------------------
# Network

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------
# Agent

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = GAMMA,
        learning_rate: float = LEARNING_RATE,
        epsilon: float = EPSILON_START,
        epsilon_min: float = EPSILON_MIN,
        epsilon_decay: float = EPSILON_DECAY,
        buffer_size: int = BUFFER_SIZE,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Two networks: one we train every step, one used only to compute
        # stable Bellman targets. Without this split, the target shifts
        # under the network every update, which destabilizes training.
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target net is never trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss; less sensitive to reward outliers than MSE

        self.train_step_count = 0

    def act(self, state, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        greedy=True forces pure exploitation (used during evaluation).
        """
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size: int = BATCH_SIZE):
        if len(self.memory) < batch_size:
            return None  # not enough data yet

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s, a) for the actions actually taken, from the policy network
        q_values = self.policy_net(states).gather(1, actions)

        # max_a' Q_target(s', a'), from the target network, no gradient needed
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % TARGET_UPDATE_EVERY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()


# ---------------------------------
# Training loop

def train():
    env = LanderEnv()
    agent = DQNAgent(state_dim=STATE_SIZE, action_dim=NUM_ACTIONS)

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = agent.act(state)
            next_state, reward, done, _info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            if steps % TRAIN_EVERY == 0:
                agent.replay()

        agent.decay_epsilon()
        print(f"Episode {episode + 1:4d} | reward: {total_reward:8.2f} | "
              f"epsilon: {agent.epsilon:.3f} | steps: {steps}")

    return agent, env


def evaluate(agent: DQNAgent, env: LanderEnv, num_episodes: int = 10):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < MAX_STEPS:
            action = agent.act(state, greedy=True)  # no exploration during eval
            next_state, reward, done, _info = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
        total_rewards.append(total_reward)

    print(f"Average Total Reward (Evaluation over {num_episodes} episodes): "
          f"{np.mean(total_rewards):.2f}")
    return total_rewards


if __name__ == "__main__":
    trained_agent, env = train()
    evaluate(trained_agent, env)