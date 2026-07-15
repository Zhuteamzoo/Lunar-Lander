
import math
import random
import sys
from dataclasses import dataclass

import pygame

# Display
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Physics (pixels, pixels/s, pixels/s²)
GRAVITY = 150.0
THRUST_ACCEL = 300.0
ROTATE_ACCEL = 600.0
MAX_ROTATE_SPEED = 600.0
ROTATION_DAMPING = 0.96
INITIAL_FUEL = 4000.0
FUEL_BURN_PER_SECOND = 200.0

# Landing tolerances
SAFE_VERTICAL_SPEED = 70.0
SAFE_HORIZONTAL_SPEED = 40.0
SAFE_ANGLE_DEVIATION = 20.0

# Ship geometry (~28 px wide, ~40 px tall)
SHIP_TRIANGLE = [(0.0, -22.0), (-14.0, 0.0), (14.0, 0.0)]
SHIP_RECT = [(-10.0, 0.0), (10.0, 0.0), (10.0, 18.0), (-10.0, 18.0)]
THRUSTER_POINT = (0.0, 18.0)
SHIP_BOTTOM_LOCAL_Y = 18.0

# Landing pad sizes (width in px) — total is well under 50% of screen width
PAD_SPECS = [
    {"width": 88, "points": 10},   # biggest, easiest
    {"width": 54, "points": 50},   # medium
    {"width": 34, "points": 100},  # tiny, just wider than the rocket
]


@dataclass
class LandingPad:
    x1: float
    x2: float
    y: float
    points: int


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def rotate_point(x: float, y: float, angle_rad: float) -> tuple[float, float]:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return x * c - y * s, x * s + y * c


def line_intersect(a1, a2, b1, b2) -> bool:
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return False
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = clamp((apx * abx + apy * aby) / ab_len_sq, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def pad_for_segment(a: tuple[float, float], b: tuple[float, float], pads: list[LandingPad]) -> LandingPad | None:
    ax, ay = a
    bx, by = b
    if abs(ay - by) > 0.5:
        return None
    seg_x1, seg_x2 = min(ax, bx), max(ax, bx)
    for pad in pads:
        if abs(ay - pad.y) < 0.5 and seg_x1 >= pad.x1 - 1 and seg_x2 <= pad.x2 + 1:
            return pad
    return None


def place_landing_pads() -> list[LandingPad]:
    margin = 24
    min_gap = 36
    min_y = SCREEN_HEIGHT - 270
    max_y = SCREEN_HEIGHT - 60
    usable_left = margin
    usable_right = SCREEN_WIDTH - margin

    specs = PAD_SPECS.copy()
    random.shuffle(specs)

    pads: list[LandingPad] = []
    occupied: list[tuple[float, float]] = []

    for spec in specs:
        width = spec["width"]
        placed = False
        for _ in range(300):
            x1 = random.uniform(usable_left, usable_right - width)
            x2 = x1 + width
            if any(not (x2 + min_gap < ox1 or x1 > ox2 + min_gap) for ox1, ox2 in occupied):
                continue
            pad_y = random.uniform(min_y, max_y)
            pads.append(LandingPad(x1=x1, x2=x2, y=pad_y, points=spec["points"]))
            occupied.append((x1, x2))
            placed = True
            break
        if not placed:
            # Fallback: evenly space pads if random placement fails
            slot = len(pads)
            span = usable_right - usable_left
            x1 = usable_left + slot * (span / len(PAD_SPECS)) + min_gap
            x1 = clamp(x1, usable_left, usable_right - width)
            pads.append(LandingPad(x1=x1, x2=x1 + width, y=(min_y + max_y) / 2, points=spec["points"]))
            occupied.append((x1, x1 + width))

    pads.sort(key=lambda p: p.x1) 
    return pads


def generate_terrain() -> tuple[list[tuple[float, float]], list[LandingPad]]:
    margin = 20
    step = 26
    min_y = SCREEN_HEIGHT - 270
    max_y = SCREEN_HEIGHT - 60

    pads = place_landing_pads()
    total_pad_width = sum(p.x2 - p.x1 for p in pads)
    assert total_pad_width < SCREEN_WIDTH * 0.5, "Landing pads must cover less than half the terrain"

    points: list[tuple[float, float]] = []
    x = float(margin)
    y = random.uniform(min_y, max_y)
    points.append((0.0, y + random.uniform(-20, 20)))
    points.append((x, y))

    pad_idx = 0
    while x < SCREEN_WIDTH - margin:
        if pad_idx < len(pads) and x >= pads[pad_idx].x1 - step * 0.5:
            pad = pads[pad_idx]
            if points[-1][0] < pad.x1:
                points.append((pad.x1, pad.y))
            else:
                points[-1] = (pad.x1, pad.y)
            points.append((pad.x2, pad.y))
            x = pad.x2
            y = pad.y
            pad_idx += 1
            continue

        x += step
        y = clamp(y + random.randint(-50, 50), min_y, max_y)
        points.append((x, y))

    if points[-1][0] < SCREEN_WIDTH:
        points.append((float(SCREEN_WIDTH), points[-1][1]))

    return points, pads


class Spark:
    __slots__ = ("x", "y", "vx", "vy", "life")

    def __init__(self, x: float, y: float, vx: float, vy: float, life: float) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life

    def update(self, dt: float) -> bool:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        return self.life > 0


class Rocket:
    def __init__(self) -> None:
        self.x = SCREEN_WIDTH * 0.5
        self.y = 90.0
        self.vx = 0.0
        self.vy = 0.0
        self.angle_deg = 0.0
        self.angular_velocity = 0.0
        self.fuel = INITIAL_FUEL
        self.sparks: list[Spark] = []
        self.state = "playing"
        self.landed_points = 0

    def reset_flight(self) -> None:
        self.x = SCREEN_WIDTH * 0.5
        self.y = 90.0
        self.vx = 0.0
        self.vy = 0.0
        self.angle_deg = 0.0
        self.angular_velocity = 0.0
        self.fuel = INITIAL_FUEL
        self.sparks.clear()
        self.state = "playing"
        self.landed_points = 0

    def world_point(self, lx: float, ly: float) -> tuple[float, float]:
        wx, wy = rotate_point(lx, ly, math.radians(self.angle_deg))
        return self.x + wx, self.y + wy

    def ship_edges(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        tip, left, right = SHIP_TRIANGLE
        r_tl, r_tr, r_br, r_bl = SHIP_RECT
        local_segments = [
            (tip, left),
            (tip, right),
            (r_tl, r_bl),
            (r_tr, r_br),
            (r_bl, r_br),
        ]
        return [(self.world_point(a[0], a[1]), self.world_point(b[0], b[1])) for a, b in local_segments]

    def collision_points(self) -> list[tuple[float, float]]:
        local_pts = SHIP_TRIANGLE + SHIP_RECT
        return [self.world_point(lx, ly) for lx, ly in local_pts]

    def spawn_thruster_sparks(self) -> None:
        base_x, base_y = self.world_point(*THRUSTER_POINT)
        angle_rad = math.radians(self.angle_deg)
        ex = -math.sin(angle_rad)
        ey = math.cos(angle_rad)

        for _ in range(random.randint(4, 8)):
            spread = random.uniform(-0.35, 0.35)
            sx = ex * math.cos(spread) - ey * math.sin(spread)
            sy = ex * math.sin(spread) + ey * math.cos(spread)
            speed = random.uniform(90.0, 180.0)
            self.sparks.append(
                Spark(
                    base_x + random.uniform(-2.0, 2.0),
                    base_y + random.uniform(-2.0, 2.0),
                    sx * speed,
                    sy * speed,
                    random.uniform(0.16, 0.34),
                )
            )

    def update(self, dt: float, thrust: bool, rotate_left: bool, rotate_right: bool) -> None:
        self.sparks = [s for s in self.sparks if s.update(dt)]
        if self.state != "playing":
            return

        angular_accel = 0.0
        if rotate_left:
            angular_accel -= ROTATE_ACCEL
        if rotate_right:
            angular_accel += ROTATE_ACCEL

        self.angular_velocity += angular_accel * dt
        self.angular_velocity = clamp(self.angular_velocity, -MAX_ROTATE_SPEED, MAX_ROTATE_SPEED)
        self.angular_velocity *= ROTATION_DAMPING ** (dt * 60.0)
        self.angle_deg = normalize_angle_deg(self.angle_deg + self.angular_velocity * dt)

        if thrust and self.fuel > 0.0:
            angle_rad = math.radians(self.angle_deg)
            self.vx += math.sin(angle_rad) * THRUST_ACCEL * dt
            self.vy += -math.cos(angle_rad) * THRUST_ACCEL * dt
            self.fuel = max(0.0, self.fuel - FUEL_BURN_PER_SECOND * dt)
            self.spawn_thruster_sparks()

        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

        if self.x < -40 or self.x > SCREEN_WIDTH + 40 or self.y < -40 or self.y > SCREEN_HEIGHT + 40:
            self.state = "crashed"

    def draw(self, surface: pygame.Surface) -> None:
        for a, b in self.ship_edges():
            pygame.draw.line(surface, WHITE, a, b, 1)
        for spark in self.sparks:
            if 0 <= int(spark.x) < SCREEN_WIDTH and 0 <= int(spark.y) < SCREEN_HEIGHT:
                surface.set_at((int(spark.x), int(spark.y)), WHITE)


class LanderGame:
    def __init__(self) -> None:
        self.terrain_points: list[tuple[float, float]] = []
        self.landing_pads: list[LandingPad] = []
        self.terrain_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self.rocket = Rocket()
        self.total_score = 0
        self.new_terrain()

    def new_terrain(self) -> None:
        self.terrain_points, self.landing_pads = generate_terrain()
        self.terrain_segments = list(zip(self.terrain_points[:-1], self.terrain_points[1:]))
        self.rocket.reset_flight()

    def y_at_x(self, x: float) -> float | None:
        for (x1, y1), (x2, y2) in self.terrain_segments:
            if x1 <= x <= x2 or x2 <= x <= x1:
                if abs(x2 - x1) < 1e-9:
                    return min(y1, y2)
                t = (x - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)
        return None

    def altitude(self) -> float:
        ground_y = self.y_at_x(self.rocket.x)
        if ground_y is None:
            return 0.0
        bottom_y = max(p[1] for p in self.rocket.collision_points())
        return max(0.0, ground_y - bottom_y)

    def check_collision(self) -> None:
        if self.rocket.state != "playing":
            return

        ship_edges = self.rocket.ship_edges()
        ship_pts = self.rocket.collision_points()
        touched_pad: LandingPad | None = None
        touched = False

        for ta, tb in self.terrain_segments:
            pad = pad_for_segment(ta, tb, self.landing_pads)
            for se in ship_edges:
                if line_intersect(se[0], se[1], ta, tb):
                    touched = True
                    if pad is not None:
                        touched_pad = pad
                    break
            if touched:
                break

            if not touched:
                for px, py in ship_pts:
                    if point_segment_distance(px, py, ta[0], ta[1], tb[0], tb[1]) < 2.5:
                        touched = True
                        if pad is not None:
                            touched_pad = pad
                        break

        if not touched:
            return

        vertical_speed = abs(self.rocket.vy)
        horizontal_speed = abs(self.rocket.vx)
        angle_error = abs(normalize_angle_deg(self.rocket.angle_deg))

        safe = (
            touched_pad is not None
            and vertical_speed < SAFE_VERTICAL_SPEED
            and horizontal_speed < SAFE_HORIZONTAL_SPEED
            and angle_error <= SAFE_ANGLE_DEVIATION
        )

        if safe:
            self.rocket.state = "landed"
            self.rocket.vx = 0.0
            self.rocket.vy = 0.0
            self.rocket.angular_velocity = 0.0
            self.rocket.landed_points = touched_pad.points
            self.total_score += touched_pad.points
            self.rocket.y = touched_pad.y - SHIP_BOTTOM_LOCAL_Y
        else:
            self.rocket.state = "crashed"
            self.rocket.vx = 0.0
            self.rocket.vy = 0.0
            self.rocket.angular_velocity = 0.0

    def update(self, dt: float, thrust: bool, rotate_left: bool, rotate_right: bool) -> None:
        self.rocket.update(dt, thrust, rotate_left, rotate_right)
        self.check_collision()

    def draw_terrain(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        pygame.draw.lines(surface, WHITE, False, self.terrain_points, 1)
        for pad in self.landing_pads:
            pygame.draw.line(surface, WHITE, (pad.x1, pad.y), (pad.x2, pad.y), 2)
            label = font.render(str(pad.points), True, WHITE)
            label_x = (pad.x1 + pad.x2) / 2 - label.get_width() / 2
            surface.blit(label, (label_x, pad.y - 22))


def main() -> None:
    pygame.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Lunar Lander")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    big_font = pygame.font.SysFont("consolas", 34)

    game = LanderGame()
    stars = [(random.randint(0, SCREEN_WIDTH - 1), random.randint(0, SCREEN_HEIGHT - 1)) for _ in range(100)]

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        if dt <= 0.0:
            dt = 1.0 / FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.new_terrain()
                elif event.key == pygame.K_n and game.rocket.state != "playing":
                    game.new_terrain()

        keys = pygame.key.get_pressed()
        game.update(
            dt,
            thrust=keys[pygame.K_UP],
            rotate_left=keys[pygame.K_LEFT],
            rotate_right=keys[pygame.K_RIGHT],
        )

        window.fill(BLACK)
        for sx, sy in stars:
            window.set_at((sx, sy), WHITE)

        game.draw_terrain(window, font)
        game.rocket.draw(window)

        hud_lines = [
            f"ALTITUDE: {game.altitude():6.1f}",
            f"HORIZONTAL SPEED: {abs(game.rocket.vx):5.2f}",
            f"VERTICAL SPEED: {abs(game.rocket.vy):5.2f}",
            f"FUEL: {int(game.rocket.fuel):4d}",
            f"SCORE: {game.total_score}",
            "",
            "UP: thrust   LEFT/RIGHT: rotate",
            "R: new terrain   N: next after landing",
        ]
        y = 12
        for line in hud_lines:
            if line:
                window.blit(font.render(line, True, WHITE), (12, y))
            y += 24

        if game.rocket.state == "landed":
            msg = big_font.render(f"LANDED — +{game.rocket.landed_points} POINTS", True, WHITE)
            window.blit(msg, msg.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20)))
            sub = font.render("Press N for new terrain or R to restart", True, WHITE)
            window.blit(sub, sub.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 24)))
        elif game.rocket.state == "crashed":
            msg = big_font.render("CRASHED — PRESS R TO RESTART", True, WHITE)
            window.blit(msg, msg.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
