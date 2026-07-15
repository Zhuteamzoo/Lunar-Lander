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

from lander_env import LanderEnv, STATE_SIZE, NUM_ACTIONS, MAX_EPISODE_STEPS
from dashboard import TrainingDashboard

# ---------------------------------
# Hyperparameters

GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.996
BUFFER_SIZE = 100_000
BATCH_SIZE = 100
NUM_EPISODES = 1000

# Matches lander_env.py's MAX_EPISODE_STEPS.
MAX_STEPS = MAX_EPISODE_STEPS

TARGET_UPDATE_EVERY = 600

# Was 1 (train every single step). Training every 4th step instead cuts the
# total number of gradient updates ~4x with little loss in data efficiency,
# since consecutive steps are highly correlated anyway -- this is the
# single biggest lever on wall-clock training time.
TRAIN_EVERY = 4

# Successful landings are rare, so a uniformly-sampled batch mostly won't
# contain any. This keeps a second buffer holding only transitions from
# episodes that ended in a landing, and mixes a fixed fraction of every
# training batch from it -- a simpler stand-in for full Prioritized
# Experience Replay, without needing TD-error-based sampling weights.
SUCCESS_BUFFER_SIZE = 20_000
SUCCESS_BATCH_RATIO = 0.25   # fraction of each batch drawn from successes, once available
SUCCESS_MIN_TO_MIX = 20      # don't start mixing until the buffer has at least this many


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
        self.success_memory = deque(maxlen=SUCCESS_BUFFER_SIZE)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.train_step_count = 0

    def act(self, state, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def remember_success(self, transitions):
        """Called once, at the end of an episode that ended in a landing --
        copies that episode's transitions into the dedicated success buffer
        so they get oversampled during training."""
        for t in transitions:
            self.success_memory.append(t)

    def save(self, path: str = "dqn_weights.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size: int = BATCH_SIZE):
        if len(self.memory) < batch_size:
            return None

        # Mix in a fixed fraction from the success buffer, once it has
        # enough transitions to draw from. This guarantees the network
        # sees landing outcomes regularly instead of only when a uniform
        # sample from the (mostly failure-filled) main buffer happens to
        # include one.
        n_success = 0
        if len(self.success_memory) >= SUCCESS_MIN_TO_MIX:
            n_success = min(int(batch_size * SUCCESS_BATCH_RATIO), len(self.success_memory))
        n_regular = batch_size - n_success

        minibatch = random.sample(self.memory, n_regular)
        if n_success > 0:
            minibatch += random.sample(self.success_memory, n_success)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)

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

    def load(self, path: str = "dqn_weights.pth"):
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)


# ---------------------------------
# Training loop

def train():
    env = LanderEnv()
    agent = DQNAgent(state_dim=STATE_SIZE, action_dim=NUM_ACTIONS)
    dashboard = TrainingDashboard()

    first_landing_episode = None

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        info = {}
        episode_transitions = []  # collected so we can push them to the success buffer if this episode lands

        while not done and steps < MAX_STEPS:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            transition = (state, action, reward, next_state, done)
            agent.remember(*transition)
            episode_transitions.append(transition)

            state = next_state
            total_reward += reward
            steps += 1

            if steps % TRAIN_EVERY == 0:
                agent.replay()

        agent.decay_epsilon()

        result = info.get("state", "playing")
        if result == "landed":
            agent.remember_success(episode_transitions)
            if first_landing_episode is None:
                first_landing_episode = episode
                print(f"\n*** FIRST SUCCESSFUL LANDING AT EPISODE {episode + 1}! ***\n")

        print(f"Episode {episode + 1:4d} | reward: {total_reward:8.2f} | "
              f"epsilon: {agent.epsilon:.3f} | steps: {steps} | "
              f"result: {result}")

        dashboard.update(episode + 1, total_reward, result, agent.epsilon, steps)

    agent.save()
    print("Saved trained weights to dqn_weights.pth")
    dashboard.keep_open()
    return agent, env


def evaluate(agent: DQNAgent, env: LanderEnv, num_episodes: int = 10):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0.0 
        done = False
        steps = 0
        while not done and steps < MAX_STEPS:
            action = agent.act(state, greedy=True)
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