"""
Gym-style wrapper around LanderGame for DQN training.

Save your original game code as `lunar_lander.py` in the same folder as this
file (the one containing LanderGame, Rocket, LandingPad, SCREEN_WIDTH, etc).
This module imports from it directly rather than duplicating game logic, so
your rendering/game code and your RL code stay separate.

Usage:
    env = LanderEnv()
    state = env.reset()
    for _ in range(1000):
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        ...
        if done:
            state = env.reset()
        else:
            state = next_state
"""

import math

from lunar_lander import (
    LanderGame,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    INITIAL_FUEL,
    FPS,
)

# ---- Action space -----------------------------------------------------
# 0: noop            3: right
# 1: thrust          4: thrust + left
# 2: left            5: thrust + right
NUM_ACTIONS = 6

def _decode_action(action: int) -> tuple[bool, bool, bool]:
    """Returns (thrust, rotate_left, rotate_right)."""
    return {
        0: (False, False, False),
        1: (True, False, False),
        2: (False, True, False),
        3: (False, False, True),
        4: (True, True, False),
        5: (True, False, True),
    }[action]


# ---- Normalization constants -------------------------------------------
# Rough bounds for scaling raw physics values into roughly [-1, 1].
# Tune these if you see the agent struggling — e.g. if velocities routinely
# exceed these bounds, widen them.
MAX_SPEED = 250.0          # px/s, for vx and vy
MAX_ANGULAR_VEL = 600.0    # deg/s (matches MAX_ROTATE_SPEED in your game)
MAX_X = float(SCREEN_WIDTH)
MAX_Y = float(SCREEN_HEIGHT)
CONTACT_ALTITUDE_THRESHOLD = 6.0  # px; below this counts as "touching ground"

# ---- State vector (10 elements, per your spec) --------------------------
# 1. x of rocketship
# 2. y of rocketship
# 3. x velocity
# 4. y velocity
# 5. angle (raw degrees, normalized)
# 6. angular velocity
# 7. fuel
# 8. x distance to center of 100-point pad
# 9. y distance to center of 100-point pad
# 10. contact with ground (0/1)
STATE_SIZE = 10
TARGET_PAD_POINTS = 100

MAX_EPISODE_STEPS = 1500   # ~25s of simulated time at 60fps; prevents infinite hovering


class LanderEnv:
    def __init__(self):
        self.game = LanderGame()
        self.dt = 1.0 / FPS
        self.steps_elapsed = 0
        self._prev_shaping = None

    # -- core Gym-like API -------------------------------------------------

    def reset(self) -> list[float]:
        self.game.new_terrain()
        self.steps_elapsed = 0
        self._prev_shaping = self._compute_shaping()
        return self._get_state()

    def step(self, action: int):
        thrust, rotate_left, rotate_right = _decode_action(action)
        fuel_before = self.game.rocket.fuel

        self.game.update(self.dt, thrust, rotate_left, rotate_right)
        self.steps_elapsed += 1

        fuel_used = fuel_before - self.game.rocket.fuel
        reward, done = self._compute_reward(fuel_used)

        truncated = self.steps_elapsed >= MAX_EPISODE_STEPS
        info = {"state": self.game.rocket.state, "truncated": truncated}

        return self._get_state(), reward, (done or truncated), info

    # -- state construction --------------------------------------------

    def _get_target_pad(self):
        for pad in self.game.landing_pads:
            if pad.points == TARGET_PAD_POINTS:
                return pad
        return None  # shouldn't happen given PAD_SPECS, but guard anyway

    def _get_state(self) -> list[float]:
        rocket = self.game.rocket
        pad = self._get_target_pad()

        if pad is not None:
            pad_cx = (pad.x1 + pad.x2) / 2.0
            dx = (pad_cx - rocket.x) / MAX_X
            dy = (pad.y - rocket.y) / MAX_Y
        else:
            dx, dy = 0.0, 0.0

        ground_contact = 1.0 if self.game.altitude() < CONTACT_ALTITUDE_THRESHOLD else 0.0

        return [
            rocket.x / MAX_X,
            rocket.y / MAX_Y,
            rocket.vx / MAX_SPEED,
            rocket.vy / MAX_SPEED,
            _normalize_angle(rocket.angle_deg) / 180.0,
            rocket.angular_velocity / MAX_ANGULAR_VEL,
            rocket.fuel / INITIAL_FUEL,
            dx,
            dy,
            ground_contact,
        ]

    # -- reward ------------------------------------------------------------

    def _compute_shaping(self) -> float:
        """
        Potential function: higher is better. Based purely on distance to
        the 100-point pad, since that's now the only target tracked in the
        state vector.
        """
        rocket = self.game.rocket
        pad = self._get_target_pad()
        if pad is None:
            return 0.0
        pad_cx = (pad.x1 + pad.x2) / 2.0
        dist = math.hypot(pad_cx - rocket.x, pad.y - rocket.y) + 1.0
        return 1.0 / dist

    def _compute_reward(self, fuel_used: float) -> tuple[float, bool]:
        rocket = self.game.rocket

        # Dense shaping: reward getting closer to the 100-point pad
        shaping = self._compute_shaping()
        shaping_reward = 500.0 * (shaping - self._prev_shaping)
        self._prev_shaping = shaping

        # Small per-step costs to discourage stalling and wasting fuel
        time_penalty = -0.03
        fuel_penalty = -0.002 * fuel_used

        # Discourage large tilt even mid-flight, not just at touchdown
        angle_penalty = -0.001 * abs(_normalize_angle(rocket.angle_deg))

        reward = shaping_reward + time_penalty + fuel_penalty + angle_penalty
        done = False

        if rocket.state == "landed":
            reward += rocket.landed_points * 10.0
            done = True
        elif rocket.state == "crashed":
            reward -= 100.0
            done = True

        return reward, done


def _normalize_angle(angle_deg: float) -> float:
    a = angle_deg % 360.0
    if a > 180.0:
        a -= 360.0
    return a