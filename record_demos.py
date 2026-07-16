"""
Play the lander manually with the arrow keys. Every episode that ends in a
safe landing is automatically appended to human_demos.pkl, using the exact
same state/action/reward encoding as lander_env.py -- so the transitions
are drop-in compatible with DQNAgent's replay buffers.

Controls: UP = thrust, LEFT/RIGHT = rotate, Q = quit and save.
Crashes are discarded; landings are auto-saved after each episode.
"""

import os
import pickle
import sys

import pygame

from lander_env import LanderEnv
from lunar_lander import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, FPS

DEMO_FILE = "human_demos.pkl"


def _action_from_keys(keys) -> int:
    """Matches the action encoding in lander_env._decode_action."""
    thrust = keys[pygame.K_UP]
    left = keys[pygame.K_LEFT]
    right = keys[pygame.K_RIGHT]
    if thrust and left:
        return 4
    if thrust and right:
        return 5
    if thrust:
        return 1
    if left:
        return 2
    if right:
        return 3
    return 0


def load_demos(path: str = DEMO_FILE):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def save_demos(demos, path: str = DEMO_FILE):
    with open(path, "wb") as f:
        pickle.dump(demos, f)


def main():
    pygame.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Lunar Lander -- Demo Recorder")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    big_font = pygame.font.SysFont("consolas", 30)

    env = LanderEnv()
    demos = load_demos()
    print(f"Loaded {len(demos)} previously saved landing(s) from {DEMO_FILE}")

    state = env.reset()
    episode_transitions = []
    session_landings = 0
    banner = None       # (text, frames_remaining)

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        keys = pygame.key.get_pressed()
        action = _action_from_keys(keys)

        next_state, reward, done, info = env.step(action)
        episode_transitions.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            result = info.get("state", "playing")
            if result == "landed":
                demos.append(episode_transitions)
                save_demos(demos)
                session_landings += 1
                banner = (f"LANDED -- SAVED ({len(demos)} total)", 90)
                print(f"Saved landing #{len(demos)} ({session_landings} this session)")
            else:
                banner = ("CRASHED -- not saved", 60)
                print("Crashed / timed out -- discarded.")

            episode_transitions = []
            state = env.reset()

        window.fill(BLACK)
        env.game.draw_terrain(window, font)
        env.game.rocket.draw(window)

        hud = [
            f"DEMOS SAVED: {len(demos)}  (this session: {session_landings})",
            f"ALTITUDE: {env.game.altitude():6.1f}",
            "UP: thrust   LEFT/RIGHT: rotate   Q: quit",
        ]
        y = 12
        for line in hud:
            window.blit(font.render(line, True, WHITE), (12, y))
            y += 24

        if banner is not None:
            text, frames_left = banner
            msg = big_font.render(text, True, WHITE)
            window.blit(msg, msg.get_rect(center=(SCREEN_WIDTH // 2, 40)))
            frames_left -= 1
            banner = (text, frames_left) if frames_left > 0 else None

        pygame.display.flip()

    pygame.quit()
    print(f"Done. {len(demos)} total landings saved to {DEMO_FILE}.")
    sys.exit()


if __name__ == "__main__":
    main()