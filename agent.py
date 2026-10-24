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

import os
import pickle
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lander_env import LanderEnv, STATE_SIZE, NUM_ACTIONS, MAX_EPISODE_STEPS, REWARD_SCALE
from dashboard import TrainingDashboard

# ---------------------------------
# Hyperparameters

GAMMA = 0.99
LEARNING_RATE = 0.0003
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.998
BUFFER_SIZE = 100_000
BATCH_SIZE = 200
NUM_EPISODES = 1500
EVAL_EVERY = 50       # run a checkpoint eval every N training episodes
EVAL_EPISODES = 20     # how many greedy episodes per eval
EVAL_SEEDS = [10_000 + i for i in range(EVAL_EPISODES)]

CURRICULUM = [10, 50, 100]
ADVANCE_LANDING_RATE = 0.6
PHASE_EPSILON_BUMP = 0.3
# Matches lander_env.py's MAX_EPISODE_STEPS.

MAX_STEPS = MAX_EPISODE_STEPS

TARGET_UPDATE_EVERY = 500


TRAIN_EVERY = 4

SUCCESS_BUFFER_SIZE = 20_000
SUCCESS_BATCH_RATIO = 0.25  
SUCCESS_MIN_TO_MIX = 10     

HUMAN_DEMO_FILE = "human_demos.pkl"


# ---------------------------------
# Network

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

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

        self.env_step_count = 0

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
        self.last_mean_q = None   # mean Q(s,a) of the last training batch (for the dashboard)

    def act(self, state, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())
    def on_env_step(self):
        """Call once per environment step (not per replay() call) to drive target sync."""
        self.env_step_count += 1
        if self.env_step_count % TARGET_UPDATE_EVERY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def remember_success(self, transitions):
        """Called once, at the end of an episode that ended in a landing --
        copies that episode's transitions into the dedicated success buffer
        so they get oversampled during training."""
        for t in transitions:
            self.success_memory.append(t)

    def load_human_demos(self, path: str = HUMAN_DEMO_FILE, also_into_main_buffer: bool = True):
        """
        Seeds the success buffer with transitions recorded from human play
        (via record_demos.py). Call once, right after constructing the
        agent and before training starts.

        The demo file is a list of episodes; each episode is a list of
        (state, action, reward, next_state, done) tuples using the same
        encoding as LanderEnv, so they slot directly into the replay
        buffers alongside the agent's own experience.

        also_into_main_buffer: also copies transitions into the regular
        replay buffer (not just success_memory), so the network also sees
        the earlier, non-terminal states of a landing approach at normal
        sampling frequency, not just via the success-buffer mixing ratio.
        """
        if not os.path.exists(path):
            print(f"No human demo file found at {path} -- skipping.")
            return 0

        with open(path, "rb") as f:
            episodes = pickle.load(f)

        n_transitions = 0
        skipped = 0
        loaded = 0
        rescaled = 0
        for episode_transitions in episodes:
            if not episode_transitions or len(episode_transitions[0][0]) != self.state_dim:
                skipped += 1
                continue
            # Demos recorded before REWARD_SCALE existed hold raw rewards (landing ~ +430).
            # Scaled rewards never get anywhere near 50, so that's a reliable tell.
            if max(abs(t[2]) for t in episode_transitions) > 50.0:
                episode_transitions = [(s, a, r * REWARD_SCALE, ns, d)
                                       for (s, a, r, ns, d) in episode_transitions]
                rescaled += 1
            loaded += 1
            self.remember_success(episode_transitions)
            if also_into_main_buffer:
                for t in episode_transitions:
                    self.memory.append(t)
            n_transitions += len(episode_transitions)

        if rescaled:
            print(f"Rescaled {rescaled} old demo(s) by REWARD_SCALE={REWARD_SCALE}")
        if skipped:
            print(f"Skipped {skipped} demo(s) with a different state size -- re-record them.")

        print(f"Loaded {len(episodes)} human demo landing(s) "
              f"({n_transitions} transitions) from {path}")
        return loaded

    def save(self, path: str = "dqn_weights.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size: int = BATCH_SIZE):
        if len(self.memory) < batch_size:
            return None
        if len(self.memory)<150:
            current_ratio = 0.4
        else:
            current_ratio = SUCCESS_BATCH_RATIO
        n_success = 0
        if len(self.success_memory) >= SUCCESS_MIN_TO_MIX:
            n_success = min(int(batch_size * current_ratio), len(self.success_memory))
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
        self.last_mean_q = q_values.mean().item()

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step_count += 1

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

    phase = 0
    env.target_points = CURRICULUM[phase]
    print(f"Curriculum phase 1/{len(CURRICULUM)}: target pad = {env.target_points} pts")

    best_eval_score = (-1.0, -float("inf"))   # (landing_rate, mean_reward)
    first_landing_episode = None

    agent.load_human_demos()                  # the ONLY call now
    if len(agent.memory) >= BATCH_SIZE:
        for _ in range(200):
            agent.replay()

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        info = {}
        episode_transitions = []
        ep_losses = []      # loss from every gradient step this episode
        ep_qs = []          # mean Q of every training batch this episode

        while not done and steps < MAX_STEPS:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            # store the TRUE terminal flag, not the time-out flag
            transition = (state, action, reward, next_state, info["terminal"])
            agent.remember(*transition)
            episode_transitions.append(transition)

            state = next_state
            total_reward += reward
            steps += 1
            agent.on_env_step()

            if steps % TRAIN_EVERY == 0:
                loss = agent.replay()
                if loss is not None:
                    ep_losses.append(loss)
                    ep_qs.append(agent.last_mean_q)

        agent.decay_epsilon()

        avg_loss = float(np.mean(ep_losses)) if ep_losses else None
        avg_q = float(np.mean(ep_qs)) if ep_qs else None

        result = info.get("state", "playing")
        on_target = (result == "landed"
                     and env.game.rocket.landed_points == env.target_points)
        if result == "landed" and not on_target:
            result = "landed_wrong_pad"
        if on_target:
            agent.remember_success(episode_transitions)
            if first_landing_episode is None:
                first_landing_episode = episode
                print(f"\n*** FIRST SUCCESSFUL LANDING AT EPISODE {episode + 1}! ***\n")

        print(f"Episode {episode + 1:4d} | reward: {total_reward:8.2f} | "
              f"epsilon: {agent.epsilon:.3f} | steps: {steps} | "
              f"result: {result} | "
              f"loss: {avg_loss if avg_loss is not None else float('nan'):.4f}")
        dashboard.update(episode + 1, total_reward,
                         "crashed" if result == "landed_wrong_pad" else result,
                         agent.epsilon, steps,
                         loss=avg_loss, mean_q=avg_q)

        if (episode + 1) % EVAL_EVERY == 0:
            eval_rewards, eval_landings = evaluate(agent, env)
            landing_rate = eval_landings / len(eval_rewards)
            eval_score = (landing_rate, float(np.mean(eval_rewards)))
            print(f"[checkpoint] episode {episode+1}: landing rate {landing_rate:.0%}, "
                  f"avg reward {eval_score[1]:.1f}")

            if eval_score > best_eval_score:       # landing rate first, reward breaks ties
                best_eval_score = eval_score
                best_path = f"dqn_weights_best_phase{CURRICULUM[phase]}.pth"
                agent.save(best_path)
                print(f"  -> new best, saved {best_path}")

            if phase < len(CURRICULUM) - 1 and landing_rate >= ADVANCE_LANDING_RATE:
                agent.save(f"dqn_weights_phase{CURRICULUM[phase]}.pth")
                phase += 1
                env.target_points = CURRICULUM[phase]
                agent.success_memory.clear()   # old landings were for a different target pad
                best_eval_score = (-1.0, -float("inf"))   # new target, new yardstick
                agent.epsilon = max(agent.epsilon, PHASE_EPSILON_BUMP)
                print(f"*** Curriculum phase {phase + 1}/{len(CURRICULUM)}: "
                      f"target pad = {env.target_points} pts ***")

    agent.save()
    agent.save("dqn_weights_final.pth")    # moved before keep_open(), which likely blocks
    print("Saved trained weights to dqn_weights.pth")
    dashboard.keep_open()
    return agent, env


def evaluate(agent: DQNAgent, env: LanderEnv, num_episodes: int = EVAL_EPISODES, seeded: bool = True):
    """Greedy evaluation on a fixed set of terrains. Returns (rewards, num_landings)."""
    total_rewards = []
    landings = 0
    rng_state = random.getstate()          # so eval doesn't disturb training randomness

    for i in range(num_episodes):
        if seeded:
            random.seed(EVAL_SEEDS[i % len(EVAL_SEEDS)])
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        info = {}
        while not done and steps < MAX_STEPS:
            action = agent.act(state, greedy=True)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        total_rewards.append(total_reward)
        if info.get("state") == "landed" and env.game.rocket.landed_points == env.target_points:
            landings += 1

    random.setstate(rng_state)
    print(f"Evaluation over {num_episodes} episodes: {landings} landings, "
          f"avg reward {np.mean(total_rewards):.2f}")
    return total_rewards, landings


if __name__ == "__main__":
    trained_agent, env = train()
    env.target_points = CURRICULUM[-1]     # judge the final model on the real target
    evaluate(trained_agent, env)