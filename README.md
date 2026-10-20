# Lunar Lander DQN

A custom Lunar Lander game built in pygame, plus a Deep Q-Network (PyTorch) that learns to fly and land the rocket on a target pad. Everything is written from scratch: the physics, the Gym-style environment wrapper, the agent, the training loop, and a live training dashboard.

You can play the game yourself, record your own landings as demonstrations, train an agent, and then watch it fly.

## Features

- **Custom physics game** with gravity, thrust, rotation with damping, limited fuel, and procedurally generated terrain
- **Three landing pads** of different sizes and values (10, 50, and 100 points). The smaller the pad, the harder the landing
- **Double DQN** with a target network, gradient clipping, and a Huber loss
- **Curriculum learning over pad size**: the agent trains on the easy 10-point pad, then moves to the 50-point and 100-point pads once it lands reliably
- **Success replay buffer** that oversamples transitions from successful landings
- **Learning from human demonstrations**: record your own landings and seed the replay buffers with them (DQfD-style)
- **Live matplotlib dashboard** showing reward, outcomes, and run statistics during training
- **Periodic greedy evaluation** on fixed terrain seeds, with best-checkpoint saving

## Project structure

| File | Purpose |
|---|---|
| `lunar_lander.py` | The game itself: physics, terrain generation, landing pads, collision and landing rules. Playable by hand |
| `lander_env.py` | Gym-style wrapper (`reset()` / `step()`): action decoding, state vector, reward function |
| `agent.py` | DQN network, agent (replay buffers, Double DQN update), training loop, and evaluation |
| `dashboard.py` | Live training dashboard |
| `record_demos.py` | Play manually and save successful landings to `human_demos.pkl` |
| `Watch_AI.py` | Watch a trained agent play, greedily and live |

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Zhuteamzoo/Lunar-Lander.git
cd Lunar-Lander
pip install pygame torch numpy matplotlib
```

## Usage

### Play the game yourself

```bash
python lunar_lander.py
```

| Key | Action |
|---|---|
| Up | Thrust |
| Left / Right | Rotate |
| R | New terrain |
| N | New terrain after a landing |

### Record human demonstrations (optional)

```bash
python record_demos.py
```

Fly the rocket onto the target pad. Every safe landing is appended to `human_demos.pkl` using the same state, action, and reward encoding as the environment, so the transitions drop straight into the agent's replay buffers. Crashes are discarded. Press `Q` to quit and save.

If you change the reward function or state vector, re-record your demos. The file stores rewards and states as they were when you recorded them.

### Train the agent

```bash
python agent.py
```

On startup the agent loads `human_demos.pkl` if it exists, pre-trains briefly on it, and then runs the training loop. A dashboard window opens and updates every episode. Closing it does not stop training.

### Watch the trained agent

```bash
python Watch_AI.py
```

The agent plays greedily (epsilon = 0). After a crash the score resets to 0. After a safe landing the score is kept and a new terrain is generated.

`Watch_AI.py` loads the weights file set by `WEIGHTS_PATH` at the top of the script. The state vector includes the position of the **target pad**, so build the environment with the same `target_points` the weights were trained on (for example, `LanderEnv(target_points=10)` for the `phase10` weights).

## How it works

### Game rules

A landing counts as safe when the rocket touches a landing pad with:

- vertical speed below 70 px/s
- horizontal speed below 40 px/s
- tilt within 20 degrees of upright

Touching anything else, or touching a pad too fast or too tilted, is a crash. Leaving the screen is also a crash. The rocket starts with 4000 units of fuel, and thrusting burns 200 per second.

| Pad | Width | Points |
|---|---|---|
| Large | 88 px | 10 |
| Medium | 54 px | 50 |
| Small | 34 px | 100 |

### Actions (6, discrete)

| ID | Action |
|---|---|
| 0 | Do nothing |
| 1 | Thrust |
| 2 | Rotate left |
| 3 | Rotate right |
| 4 | Thrust + rotate left |
| 5 | Thrust + rotate right |

### State (20 values, all normalized)

1. Rocket x and y position
2. x and y velocity
3. Angle and angular velocity
4. Remaining fuel
5. x and y offset to the center of the target pad
6. Ground-contact flag
7. Nine terrain-height samples spanning 200 px either side of the rocket (relative to the rocket)
8. Altitude above the ground directly below

### Reward

The reward combines:

- **Potential-based shaping** that rewards getting closer to the target pad, with a speed penalty that applies as the rocket nears it
- **A small angle penalty** that encourages staying upright
- **A large terminal reward** for a safe landing (scaled by the pad's points)
- **A large terminal penalty** for crashing
- **A time-out penalty** if the episode reaches the step limit without finishing

All of this lives in `lander_env.py`, in `_compute_shaping` and `_compute_reward`.

### Agent and training

- **Network**: two hidden layers of 256 units (ReLU), one Q-value per action
- **Algorithm**: Double DQN with a target network synced every 500 environment steps
- **Exploration**: epsilon-greedy, decaying from 1.0 to 0.05, with a bump back up when the curriculum advances
- **Replay**: a main buffer of 100,000 transitions plus a 20,000-transition success buffer that supplies 25% of each batch once it has enough landings
- **Curriculum**: the target pad goes 10, then 50, then 100 points. The agent advances when its greedy evaluation landing rate reaches 60%
- **Evaluation**: every 50 episodes, 20 greedy episodes on fixed terrain seeds. The best checkpoint per phase is kept, ranked by landing rate first and mean reward second

### Hyperparameters (`agent.py`)

| Name | Value |
|---|---|
| `GAMMA` | 0.99 |
| `LEARNING_RATE` | 0.001 |
| `EPSILON_START` / `EPSILON_MIN` / `EPSILON_DECAY` | 1.0 / 0.05 / 0.995 |
| `BATCH_SIZE` | 200 |
| `BUFFER_SIZE` | 100,000 |
| `NUM_EPISODES` | 1500 |
| `TRAIN_EVERY` | 4 environment steps |
| `TARGET_UPDATE_EVERY` | 500 environment steps |
| `SUCCESS_BATCH_RATIO` | 0.25 |
| `ADVANCE_LANDING_RATE` | 0.6 |
| `MAX_EPISODE_STEPS` | 1000 |

## Output files

| File | Contents |
|---|---|
| `dqn_weights_best_phase{N}.pth` | Best checkpoint while training on the N-point pad |
| `dqn_weights_phase{N}.pth` | Weights at the moment the curriculum advanced past phase N |
| `dqn_weights.pth`, `dqn_weights_final.pth` | Final weights after training |
| `human_demos.pkl` | Recorded human landings |

## Troubleshooting

- **The agent hovers near the pad instead of landing.** Check the reward function. If hovering is cheaper than risking a crash, a DQN will hover. Penalize time spent near the pad and make sure a near-miss crash costs less than a crash far away.
- **The agent crashes far from the pad.** Watch for terrain collisions at high speed. A speed limit that depends on altitude is more effective than a flat speed penalty.
- **Old demos load but training looks wrong.** Demos store the rewards from when they were recorded. Re-record them after changing the reward function.
- **Demos are skipped on load.** The state size changed. Re-record them.