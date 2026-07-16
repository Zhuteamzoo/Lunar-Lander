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

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from lander_env import LanderEnv, STATE_SIZE, NUM_ACTIONS, MAX_EPISODE_STEPS
from dashboard import TrainingDashboard

env = LanderEnv()
state = env.reset()
EPSILON_START = 0.99
EPSILON_DECAY = 0.998
EPSILON_MIN = 0.001
state_dim = 6
action_dim = 6
BUFFER_SIZE = 100_000
SUCCESS_BUFFER_SIZE = 20_000
MAX_STEPS = MAX_EPISODE_STEPS
BATCH_SIZE = 500

next_state, reward, done, info = env.step(0)
print(next_state, reward, done, info)

class DQN(nn.module):
    def __init__(self, input_dim: int, output_dim: int):
        super.__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        epsilon: float = EPSILON_START,
        epsilon_min: float = EPSILON_MIN,
        epsilon_decay: float = EPSILON_DECAY,
        buffer_size: int = BUFFER_SIZE
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.memory = deque(maxlen=buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    def act(self,greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, 5)
    def decay_epsilon(self):
        if epsilon > self.epsilon_min:
            epsilon *= self.epsilon_decay
    def remember(self, state, action, reward, next_sate, done):
        self.memory.append((state, action, reward, next_sate, done))
    def save(self, path: str = "dqn_weights.pth"):
        torch.save(self.policy_net.state_dict(), path)
    def replay(self, batch_size: int = BATCH_SIZE):
        if len(self.memory) < batch_size:
            return None
        
        n_success = 0
        

    