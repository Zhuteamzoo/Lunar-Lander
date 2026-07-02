"""
Watch the trained DQN agent play Lunar Lander live.

Runs the policy greedily (epsilon = 0, no exploration). On crash, the score
resets to 0 and a fresh run begins. On a safe landing, the score is kept
and a new terrain is generated so the agent keeps flying.

Requires dqn_weights.pth in the same folder (produced by running agent.py).
"""

import sys

import pygame

from lunar_lander import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, FPS
from lander_env import LanderEnv, STATE_SIZE, NUM_ACTIONS
from agent import DQNAgent

WEIGHTS_PATH = "dqn_weights.pth"


def main():
    pygame.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Lunar Lander — DQN Agent (watching)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)
    big_font = pygame.font.SysFont("consolas", 34)

    env = LanderEnv()
    agent = DQNAgent(state_dim=STATE_SIZE, action_dim=NUM_ACTIONS)
    agent.load(WEIGHTS_PATH)
    agent.policy_net.eval()  # not strictly required (no dropout/batchnorm), but correct habit

    state = env.reset()
    game = env.game  # same LanderGame instance env is driving — used here only for drawing

    result_message = ""
    result_timer = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Greedy action — no exploration, this is the agent's real learned policy
        action = agent.act(state, greedy=True)
        next_state, reward, done, info = env.step(action)
        state = next_state

        if done:
            if info["state"] == "landed":
                result_message = f"LANDED — SCORE: {game.total_score}"
                # keep total_score, just fly again on a new terrain
                state = env.reset()
            else:
                # crashed or timed out — restart the score
                result_message = f"CRASHED — SCORE RESET (was {game.total_score})"
                game.total_score = 0
                state = env.reset()
            result_timer = 1.5  # seconds to show the message before it fades

        # -- render --
        window.fill(BLACK)
        game.draw_terrain(window, font)
        game.rocket.draw(window)

        hud_lines = [
            f"SCORE: {game.total_score}",
            f"ALTITUDE: {game.altitude():6.1f}",
            f"FUEL: {int(game.rocket.fuel):4d}",
            "Agent playing greedily (epsilon = 0)",
        ]
        y = 12
        for line in hud_lines:
            window.blit(font.render(line, True, WHITE), (12, y))
            y += 24

        if result_timer > 0.0:
            msg = big_font.render(result_message, True, WHITE)
            window.blit(msg, msg.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
            result_timer -= dt

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()