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
MAX_SPEED = 250.0          # px/s, for vx and vy
MAX_ANGULAR_VEL = 600.0    # deg/s (matches MAX_ROTATE_SPEED in your game)
MAX_X = float(SCREEN_WIDTH)
MAX_Y = float(SCREEN_HEIGHT)
MAX_DIAG_DIST = math.hypot(SCREEN_WIDTH, SCREEN_HEIGHT)  # for normalizing distance in shaping
CONTACT_ALTITUDE_THRESHOLD = 6.0  # px; below this counts as "touching ground"

MAX_EPISODE_STEPS = 600

# This used to be one large penalty applied only at the very end (was
# raised from 150 to 2000). With gamma=0.99, a reward 200-300 steps away is
# discounted to roughly 8-15% of its face value from an early-episode
# state's perspective -- so making this number bigger mostly failed to
# change early decisions, it just made the reward scale more lopsided.
# Most of the "don't stall" pressure now comes from the escalating
# per-step time penalty in _compute_reward instead; this is just a small
# top-up applied once, at the very end, if the episode times out.
TRUNCATION_PENALTY = 100.0

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
        if truncated and not done:
            reward -= TRUNCATION_PENALTY

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
        Potential function: higher is better. Combines distance AND
        velocity into one potential, so the reward for closing the gap AND
        slowing down is paid as a *difference* (this step's potential minus
        last step's), not as a flat bonus. That distinction matters a lot:
        a flat per-step bonus for "being close and slow" can be farmed
        forever just by parking in that state, since it pays out every
        step regardless of whether anything is actually improving. A
        potential difference telescopes to (near) zero if the agent just
        sits still, so there's no reward for merely staying put -- only
        for genuinely getting closer and slower than you were a moment ago.
        """
        rocket = self.game.rocket
        pad = self._get_target_pad()
        if pad is None:
            return 0.0
        pad_cx = (pad.x1 + pad.x2) / 2.0
        dist_norm = math.hypot(pad_cx - rocket.x, pad.y - rocket.y) / MAX_DIAG_DIST
        speed_norm = math.hypot(rocket.vx, rocket.vy) / MAX_SPEED
        return -(dist_norm) - (0.6 * speed_norm)

    def _compute_reward(self, fuel_used: float) -> tuple[float, bool]:
        rocket = self.game.rocket

        shaping = self._compute_shaping()
        shaping_reward = 200.0 * (shaping - self._prev_shaping)
        self._prev_shaping = shaping

        step_penalty = 0

        # Escalating urgency: cost per step grows as the episode runs on,
        # instead of a single penalty way out at the truncation point.
        # This is felt immediately, every step, so it isn't neutered by
        # gamma discounting the way a single distant penalty was -- at
        # step 0 this is -0.05, by step 300 (episode end) it's -0.35.
        time_penalty = -0.05 * (1.0 + self.steps_elapsed / 50.0)

        fuel_penalty = -0.05 * fuel_used
        angle_penalty = -0.02 * abs(_normalize_angle(rocket.angle_deg))

        reward = shaping_reward + step_penalty + time_penalty + fuel_penalty + angle_penalty
        done = False

        if rocket.state == "landed":
            reward += rocket.landed_points * 10.0
            done = True
        elif rocket.state == "crashed":
            impact_speed = math.hypot(rocket.vx, rocket.vy)
            reward -= 250.0 + 0.3 * impact_speed
            done = True

        return reward, done


def _normalize_angle(angle_deg: float) -> float:
    a = angle_deg % 360.0
    if a > 180.0:
        a -= 360.0
    return a