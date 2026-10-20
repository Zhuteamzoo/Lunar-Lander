import math
import numpy as np

from lunar_lander import (
    LanderGame,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    INITIAL_FUEL,
    FPS,
    SHIP_BOTTOM_LOCAL_Y
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

MAX_EPISODE_STEPS = 1000

TRUNCATION_PENALTY = 60.0

# --- safe-speed envelope + anti-hover ---
TOUCHDOWN_SPEED = 35.0        # px/s we want at contact (hard limits: 70 vertical / 40 horizontal)
BRAKE_ACCEL = 50.0            # px/s^2, conservative (true net braking is ~150)
SPEED_ENVELOPE_WEIGHT = 0.6
HOVER_RADIUS = 150.0          # px from pad center
HOVER_STEP_PENALTY = 0.2      # per step while lingering near the pad
PAD_CRASH_BASE = 120.0        # crash that touches the target pad ("almost landed")
FAR_CRASH_BASE = 300.0        # crash anywhere else
CRASH_SPEED_WEIGHT = 0.5

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
TARGET_PAD_POINTS = 100
N_TERRAIN_SAMPLES = 9
TERRAIN_SPAN = 200.0
STATE_SIZE = 10 + N_TERRAIN_SAMPLES + 1

class LanderEnv:
    def __init__(self, target_points = TARGET_PAD_POINTS):
        self.game = LanderGame()
        self.dt = 1.0 / FPS
        self.steps_elapsed = 0
        self._prev_shaping = None
        self.target_points = target_points
        self._pre_speed = 0.0

    # -- core Gym-like API -------------------------------------------------

    def reset(self) -> list[float]:
        self.game.new_terrain()
        xs = [p[0] for p in self.game.terrain_points]
        ys = [p[1] for p in self.game.terrain_points]
        self._hx = np.arange(0, SCREEN_WIDTH + 1)
        self._hy = np.interp(self._hx, xs, ys)
        self.steps_elapsed = 0
        self._prev_shaping = self._compute_shaping()
        self._pre_speed = 0.0
        return self._get_state()

    def _ground_y(self,x):
         i = int(min(max(x,0), SCREEN_WIDTH))
         return float(self._hy[i])

    def _terrain_features(self, altitude):
        r = self.game.rocket
        offests = np.linspace(-TERRAIN_SPAN, TERRAIN_SPAN, N_TERRAIN_SAMPLES)
        feats = [(self._ground_y(r.x + o) - r.y) / MAX_Y for o in offests]
        feats.append(altitude / MAX_Y)
        return feats

    def step(self, action: int):
        thrust, rotate_left, rotate_right = _decode_action(action)
        fuel_before = self.game.rocket.fuel

        r = self.game.rocket
        self._pre_speed = math.hypot(r.vx, r.vy)

        self.game.update(self.dt, thrust, rotate_left, rotate_right)
        self.steps_elapsed += 1

        fuel_used = fuel_before - self.game.rocket.fuel
        reward, done = self._compute_reward(fuel_used)

        truncated = self.steps_elapsed >= MAX_EPISODE_STEPS
        if truncated and not done:
            reward -= TRUNCATION_PENALTY

        info = {
            "state": self.game.rocket.state,
            "truncated": truncated,
            "terminal": done,
        }

        return self._get_state(), reward, (done or truncated), info

    # -- state construction --------------------------------------------

    def _get_target_pad(self):
        for pad in self.game.landing_pads:
            if pad.points == self.target_points:
                return pad
        return None  # shouldn't happen given PAD_SPECS, but guard anyway

    def _get_state(self) -> list[float]:
        altitude = self.game.altitude()
        rocket = self.game.rocket
        pad = self._get_target_pad()

        if pad is not None:
            pad_cx = (pad.x1 + pad.x2) / 2.0
            dx = (pad_cx - rocket.x) / MAX_X
            dy = (pad.y - rocket.y) / MAX_Y
        else:
            dx, dy = 0.0, 0.0

        ground_contact = 1.0 if altitude < CONTACT_ALTITUDE_THRESHOLD else 0.0

        state = [
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

        return state + self._terrain_features(altitude)
    

    # -- reward ------------------------------------------------------------

    def _compute_shaping(self) -> float:
        rocket = self.game.rocket
        pad = self._get_target_pad()
        if pad is None:
            return 0.0
        pad_cx = (pad.x1 + pad.x2) / 2.0
        # Rocket centre sits SHIP_BOTTOM_LOCAL_Y above the pad when landed,
        # so distance now actually reaches ~0 at touchdown.
        target_y = pad.y - SHIP_BOTTOM_LOCAL_Y
        dist_norm = math.hypot(pad_cx - rocket.x, target_y - rocket.y) / MAX_DIAG_DIST

        # Speed only costs something when it exceeds what we could brake from
        # at this height above the ground. Free when far up, strict near contact.
        speed = math.hypot(rocket.vx, rocket.vy)
        allowed = TOUCHDOWN_SPEED + math.sqrt(2.0 * BRAKE_ACCEL * self.game.altitude())
        excess = max(0.0, speed - allowed) / MAX_SPEED

        return -dist_norm - SPEED_ENVELOPE_WEIGHT * excess

    def _compute_reward(self, fuel_used: float) -> tuple[float, bool]:
        rocket = self.game.rocket

        shaping = self._compute_shaping()
        shaping_reward = 300.0 * (shaping - self._prev_shaping)
        self._prev_shaping = shaping

        angle_penalty = -0.02 * abs(_normalize_angle(rocket.angle_deg))

        reward = shaping_reward + angle_penalty
        done = False

        pad = self._get_target_pad()

        # Anti-hover: lingering near the pad costs a little every step.
        if pad is not None and rocket.state == "playing":
            pad_cx = (pad.x1 + pad.x2) / 2.0
            if math.hypot(pad_cx - rocket.x, pad.y - rocket.y) < HOVER_RADIUS:
                reward -= HOVER_STEP_PENALTY

        if rocket.state == "landed":
            reward += 400.0 + rocket.landed_points * 3.0
            done = True
        elif rocket.state == "crashed":
            on_pad = pad is not None and pad.x1 <= rocket.x <= pad.x2
            base = PAD_CRASH_BASE if on_pad else FAR_CRASH_BASE
            reward -= base + CRASH_SPEED_WEIGHT * self._pre_speed
            done = True

        return reward, done

def _normalize_angle(angle_deg: float) -> float:
    a = angle_deg % 360.0
    if a > 180.0:
        a -= 360.0 
    return a
