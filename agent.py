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

from lander_env import LanderEnv
import random
from collections import deque

buffer = deque(maxlen=100_000)
epsilon = 1.0
epsilon_decay = 0.9995
num_episodes = 100
max_steps = 50
batch_size = 10

env = LanderEnv()

for episode in range(num_episodes):
    state = env.reset()
    for step in range(max_steps):
        if random.random() < epsilon:
            action = random.randint(0, 5)
        else:
            action = 0  # placeholder until the Q-network exists

        next_state, reward, done, info = env.step(action)
        buffer.append((state, action, reward, next_state, done))
        epsilon *= epsilon_decay
        state = next_state

        if len(buffer) >= batch_size:
            sampled_buffer = random.sample(buffer, k=batch_size)

        if done:
            break


