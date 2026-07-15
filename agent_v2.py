import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from lander_env import LanderEnv

env = LanderEnv()
state = env.reset()

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
    