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

INFO_LABELS = [
    "EPISODE", "EPSILON", "STEPS (last ep)", "LAST RESULT",
    "", "LANDED", "CRASHED", "STALLED", "", "LATEST REWARD", "BEST REWARD",
]


class TrainingDashboard:
    def __init__(self, title: str = "DQN Training"):
        plt.style.use("dark_background")

        self.fig = plt.figure(figsize=(13, 7.5), facecolor=BG_COLOR)
        self.fig.canvas.manager.set_window_title(title)

        # -- closed-window guard --------------------------------------
        # Once the user closes the window, `closed` flips True and every
        # subsequent update() call becomes a cheap no-op instead of trying
        # to draw into a destroyed canvas (which is what was crashing
        # training before).
        self.closed = False
        self.fig.canvas.mpl_connect("close_event", self._on_close)

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
        self.steps = []
        self.state_values = []
        self.landed_counts = []
        self.crashed_counts = []
        self.stalled_counts = []
        self._landed = 0
        self._crashed = 0
        self._stalled = 0

        # -- persistent artists, created once ---------------------------
        # Reused and updated via set_data()/set_text() every episode
        # instead of clearing + rebuilding the whole figure each time.
        (self._line_reward,) = self.ax_main.plot([], [], color=REWARD_COLOR, linewidth=1.3)
        (self._line_steps,) = self.ax_main.plot([], [], color="#4caf50", linewidth=1.0, alpha=0.8)
        (self._line_state,) = self.ax_state.plot([], [], color=STATE_LINE_COLOR, linewidth=1.0, alpha=0.55)

        self.ax_state.set_ylim(-60, 1060)
        self.ax_state.set_yticks([0, 500, 1000])
        self.ax_state.set_yticklabels(["Stall", "Crash", "Land"], color=STATE_LINE_COLOR)
        self.ax_state.tick_params(axis="y", colors=STATE_LINE_COLOR)

        self.ax_main.set_ylabel("Reward", color=REWARD_COLOR)
        self.ax_main.tick_params(axis="y", colors=REWARD_COLOR)
        self.ax_main.tick_params(axis="x", colors=TEXT_COLOR)

        self._draw_static_labels()

        self.ax_info.text(0.05, 1.0, "RUN STATS", transform=self.ax_info.transAxes,
                           fontsize=12, fontweight="bold", color=TEXT_COLOR, va="top")

        # Pre-create label/value text artists for the sidebar so we only
        # ever call set_text() on them afterward.
        self._info_label_artists = {}
        self._info_value_artists = {}
        y = 0.95 - 0.08
        for label in INFO_LABELS:
            if not label:
                y -= 0.03
                continue
            self._info_label_artists[label] = self.ax_info.text(
                0.05, y, label, transform=self.ax_info.transAxes,
                fontsize=8.5, color="#8a8fa3", va="top", family="monospace",
            )
            y -= 0.045
            self._info_value_artists[label] = self.ax_info.text(
                0.05, y, "", transform=self.ax_info.transAxes,
                fontsize=12, color=TEXT_COLOR, va="top",
                family="monospace", fontweight="bold",
            )
            y -= 0.07

        plt.ion()
        self.fig.show()

    def _on_close(self, _event):
        self.closed = True

    def _draw_static_labels(self):
        self.ax_main.set_title(
            "Reward per Episode  (red = reward, blue = outcome, green = steps)",
            color=TEXT_COLOR, fontsize=12, fontweight="bold", loc="left",
        )
        self.ax_band.set_title(
            "Outcome Totals Over Training", color=TEXT_COLOR, fontsize=11, loc="left"
        )
        self.ax_band.set_xlabel("Episode", color=TEXT_COLOR)
        self.ax_band.set_ylabel("Count", color=TEXT_COLOR)
        self.ax_band.tick_params(colors=TEXT_COLOR)

    def update(self, episode: int, reward: float, result: str, epsilon: float, steps: int):
        if self.closed:
            # Window is gone -- keep training running, just stop drawing.
            return

        result = result if result in OUTCOME_VALUE else "playing"

        if result == "landed":
            self._landed += 1
        elif result == "crashed":
            self._crashed += 1
        else:
            self._stalled += 1

        self.episodes.append(episode)
        self.rewards.append(reward)
        self.steps.append(steps)
        self.state_values.append(OUTCOME_VALUE[result])
        self.landed_counts.append(self._landed)
        self.crashed_counts.append(self._crashed)
        self.stalled_counts.append(self._stalled)

        try:
            self._redraw(epsilon, steps, result)
        except Exception as e:
            # A focus-switch / window-manager hiccup mid-draw shouldn't be
            # able to kill training. Log once, mark closed if it looks like
            # the window is actually gone, otherwise just skip this frame.
            print(f"[dashboard] draw skipped ({type(e).__name__}: {e})")
            if not plt.fignum_exists(self.fig.number):
                self.closed = True

    def _redraw(self, epsilon: float, steps: int, last_result: str):
        # -- main reward/steps lines (cheap: just new data, no rebuild) --
        self._line_reward.set_data(self.episodes, self.rewards)
        self._line_steps.set_data(self.episodes, self.steps)
        self._line_state.set_data(self.episodes, self.state_values)

        self.ax_main.relim()
        self.ax_main.autoscale_view()

        # -- band chart: stackplot has no incremental update API, so this
        # axis alone gets cleared and rebuilt (cheap relative to the whole
        # figure, since it's one of four axes, not all of them) --
        self.ax_band.clear()
        self.ax_band.set_facecolor(PANEL_COLOR)
        self.ax_band.grid(True, color=ACCENT_GRID, linewidth=0.6)
        for spine in self.ax_band.spines.values():
            spine.set_color(ACCENT_GRID)

        self.ax_band.stackplot(
            self.episodes,
            self.landed_counts, self.crashed_counts, self.stalled_counts,
            colors=[OUTCOME_COLOR["landed"], OUTCOME_COLOR["crashed"], OUTCOME_COLOR["playing"]],
            labels=[OUTCOME_LABEL["landed"], OUTCOME_LABEL["crashed"], OUTCOME_LABEL["playing"]],
            alpha=0.9,
        )
        self._draw_static_labels()
        legend = self.ax_band.legend(loc="upper left", fontsize=8, framealpha=0.25)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)

        # -- sidebar: update existing text artists instead of recreating --
        total = max(1, self._landed + self._crashed + self._stalled)
        land_pct = 100.0 * self._landed / total
        info_values = {
            "EPISODE": f"{self.episodes[-1]}",
            "EPSILON": f"{epsilon:.3f}",
            "STEPS (last ep)": f"{steps}",
            "LAST RESULT": OUTCOME_LABEL.get(last_result, last_result).upper(),
            "LANDED": f"{self._landed}  ({land_pct:.1f}%)",
            "CRASHED": f"{self._crashed}",
            "STALLED": f"{self._stalled}",
            "LATEST REWARD": f"{self.rewards[-1]:.1f}",
            "BEST REWARD": f"{max(self.rewards):.1f}",
        }
        for label, value in info_values.items():
            self._info_value_artists[label].set_text(value)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def keep_open(self):
        """Call once training is done, to stop the window from closing immediately."""
        if self.closed:
            return
        plt.ioff()
        try:
            plt.show()
        except Exception:
            pass