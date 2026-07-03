"""
Live training dashboard: reward curve, per-episode outcome overlay, a
stacked band chart of outcome counts over time, and a sidebar with current
run stats. Updates once per episode without blocking training.

Requires matplotlib: pip install matplotlib --break-system-packages
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Outcome -> numeric value for the overlay line, and -> color for the bands
OUTCOME_VALUE = {"playing": 0, "crashed": 500, "landed": 1000}
OUTCOME_COLOR = {"landed": "#4caf50", "crashed": "#ff5252", "playing": "#757575"}
OUTCOME_LABEL = {"landed": "Landed", "crashed": "Crashed", "playing": "Stalled"}

REWARD_COLOR = "#ff4d4d"
STATE_LINE_COLOR = "#4da6ff"
BG_COLOR = "#12141c"
PANEL_COLOR = "#1b1e29"
TEXT_COLOR = "#e8e8f0"
ACCENT_GRID = "#2a2d3a"


class TrainingDashboard:
    def __init__(self, title: str = "DQN Training"):
        plt.style.use("dark_background")

        self.fig = plt.figure(figsize=(13, 7.5), facecolor=BG_COLOR)
        self.fig.canvas.manager.set_window_title(title)

        gs = gridspec.GridSpec(
            2, 4,
            figure=self.fig,
            height_ratios=[2.3, 1.0],
            width_ratios=[1, 1, 1, 0.55],
            hspace=0.35, wspace=0.4,
        )

        self.ax_main = self.fig.add_subplot(gs[0, :3])
        self.ax_state = self.ax_main.twinx()
        self.ax_band = self.fig.add_subplot(gs[1, :3])
        self.ax_info = self.fig.add_subplot(gs[:, 3])
        self.ax_info.axis("off")

        for ax in (self.ax_main, self.ax_band):
            ax.set_facecolor(PANEL_COLOR)
            ax.grid(True, color=ACCENT_GRID, linewidth=0.6)
            for spine in ax.spines.values():
                spine.set_color(ACCENT_GRID)

        self.episodes = []
        self.rewards = []
        self.state_values = []
        self.landed_counts = []
        self.crashed_counts = []
        self.stalled_counts = []
        self._landed = 0
        self._crashed = 0
        self._stalled = 0

        plt.ion()
        self.fig.show()
        self._draw_static_labels()

    def _draw_static_labels(self):
        self.ax_main.set_title(
            "Reward per Episode  (red = reward, blue = outcome)",
            color=TEXT_COLOR, fontsize=12, fontweight="bold", loc="left",
        )
        self.ax_band.set_title(
            "Outcome Totals Over Training", color=TEXT_COLOR, fontsize=11, loc="left"
        )

    def update(self, episode: int, reward: float, result: str, epsilon: float, steps: int):
        result = result if result in OUTCOME_VALUE else "playing"

        if result == "landed":
            self._landed += 1
        elif result == "crashed":
            self._crashed += 1
        else:
            self._stalled += 1

        self.episodes.append(episode)
        self.rewards.append(reward)
        self.state_values.append(OUTCOME_VALUE[result])
        self.landed_counts.append(self._landed)
        self.crashed_counts.append(self._crashed)
        self.stalled_counts.append(self._stalled)

        self._redraw(epsilon, steps, result)

    def _redraw(self, epsilon: float, steps: int, last_result: str):
        self.ax_main.clear()
        self.ax_state.clear()
        self.ax_band.clear()
        self.ax_info.clear()
        self.ax_info.axis("off")

        for ax in (self.ax_main, self.ax_band):
            ax.set_facecolor(PANEL_COLOR)
            ax.grid(True, color=ACCENT_GRID, linewidth=0.6)
            for spine in ax.spines.values():
                spine.set_color(ACCENT_GRID)

        # -- main reward line --
        self.ax_main.plot(self.episodes, self.rewards, color=REWARD_COLOR, linewidth=1.3)
        self.ax_main.set_ylabel("Reward", color=REWARD_COLOR)
        self.ax_main.tick_params(axis="y", colors=REWARD_COLOR)
        self.ax_main.tick_params(axis="x", colors=TEXT_COLOR)

        # -- outcome overlay line (0 / 500 / 1000), semi-transparent --
        self.ax_state.plot(self.episodes, self.state_values, color=STATE_LINE_COLOR,
                            linewidth=1.0, alpha=0.55)
        self.ax_state.set_ylim(-60, 1060)
        self.ax_state.set_yticks([0, 500, 1000])
        self.ax_state.set_yticklabels(["Stall", "Crash", "Land"], color=STATE_LINE_COLOR)
        self.ax_state.tick_params(axis="y", colors=STATE_LINE_COLOR)

        self._draw_static_labels()

        # -- stacked band chart of outcome counts --
        self.ax_band.stackplot(
            self.episodes,
            self.landed_counts, self.crashed_counts, self.stalled_counts,
            colors=[OUTCOME_COLOR["landed"], OUTCOME_COLOR["crashed"], OUTCOME_COLOR["playing"]],
            labels=[OUTCOME_LABEL["landed"], OUTCOME_LABEL["crashed"], OUTCOME_LABEL["playing"]],
            alpha=0.9,
        )
        self.ax_band.set_xlabel("Episode", color=TEXT_COLOR)
        self.ax_band.set_ylabel("Count", color=TEXT_COLOR)
        self.ax_band.tick_params(colors=TEXT_COLOR)
        legend = self.ax_band.legend(loc="upper left", fontsize=8, framealpha=0.25)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        # -- sidebar info panel --
        total = max(1, self._landed + self._crashed + self._stalled)
        land_pct = 100.0 * self._landed / total
        info_lines = [
            ("EPISODE", f"{self.episodes[-1]}"),
            ("EPSILON", f"{epsilon:.3f}"),
            ("STEPS (last ep)", f"{steps}"),
            ("LAST RESULT", OUTCOME_LABEL.get(last_result, last_result).upper()),
            ("", ""),
            ("LANDED", f"{self._landed}  ({land_pct:.1f}%)"),
            ("CRASHED", f"{self._crashed}"),
            ("STALLED", f"{self._stalled}"),
            ("", ""),
            ("LATEST REWARD", f"{self.rewards[-1]:.1f}"),
            ("BEST REWARD", f"{max(self.rewards):.1f}"),
        ]

        y = 0.95
        self.ax_info.text(0.05, 1.0, "RUN STATS", transform=self.ax_info.transAxes,
                           fontsize=12, fontweight="bold", color=TEXT_COLOR, va="top")
        y -= 0.08
        for label, value in info_lines:
            if not label:
                y -= 0.03
                continue
            self.ax_info.text(0.05, y, label, transform=self.ax_info.transAxes,
                               fontsize=8.5, color="#8a8fa3", va="top", family="monospace")
            y -= 0.045
            self.ax_info.text(0.05, y, value, transform=self.ax_info.transAxes,
                               fontsize=12, color=TEXT_COLOR, va="top",
                               family="monospace", fontweight="bold")
            y -= 0.07

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def keep_open(self):
        """Call once training is done, to stop the window from closing immediately."""
        plt.ioff()
        plt.show()