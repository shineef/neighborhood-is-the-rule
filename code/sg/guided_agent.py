"""Random play, but with the cheap priors we measured today. No model, no training.

This composes three findings into one agent, to see whether they add up to
actual level completions rather than just better statistics:

  1. Sample only from `available_actions`. 7 of 25 games are click-only; sampling
     uniformly over 1-6 there makes 5/6 of actions exact no-ops.
  2. Do not click at all in games where clicking never does anything. 11 of 25
     games ignore clicks entirely (every click strategy scored exactly 0.000);
     spending a scarce budget on ACTION6 there is pure waste. Detected online:
     after PROBE_N clicks with zero effect, stop clicking.
  3. Aim clicks at small connected same-colour components rather than uniformly.
     Median 8.1x more real events across the 14 games where clicking works, up
     to 92x on cd82. Falls back to uniform when no small component exists.

The comparison that matters is against the uniform-random baseline measured over
200 seeds: 0.00075 +/- 0.00091, 0.90 of 183 levels. If guided exploration is
worth building on, it should beat that; if it does not, the lift in "events
triggered" does not translate into progress and that is important to know before
building anything bigger on top of it.
"""
import os
import random
from collections import deque
from typing import Any

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

MAX_COMP = int(os.environ.get("GUIDED_MAX_COMP", "40"))
# Number of clicks to spend deciding whether clicking does anything at all, and
# whether to spend them up front.
#
# v1 used PROBE_N=12 and let the probe clicks arrive by chance. That is what made
# guided WORSE than random on ar25/cn04/sk48/m0r0: those games list action 6 in
# available_actions (so clicking is "legal") but clicking never does anything.
# With 6 legal actions only ~1/6 of samples are clicks, so reaching 12 clicks
# takes ~72 actions - roughly half of those games' 145-160 action level-1 budget,
# all of it spent diluting the keyboard actions that actually work.
#
# v2 front-loads instead: click on the first PROBE_N actions, decide immediately,
# then commit. Worst case cost on a click-useless game is exactly PROBE_N wasted
# actions rather than 12 wasted clicks spread over most of the budget.
PROBE_N = int(os.environ.get("GUIDED_PROBE", "30"))
FRONTLOAD = os.environ.get("GUIDED_FRONTLOAD", "0") == "1"   # off by default now that PROBE_N is large: front-loading 30 clicks would burn ~20% of a keyboard game budget up front


def small_component_cells(g, max_size=MAX_COMP):
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    out = []
    for y in range(H):
        for x in range(W):
            if seen[y, x]:
                continue
            col = g[y, x]
            q = deque([(y, x)])
            seen[y, x] = True
            comp = []
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(comp) <= max_size:
                out.extend(comp)
    return out


class GuidedAgent(Agent):
    MAX_ACTIONS = float("inf")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.clicks_tried = 0
        self.clicks_worked = 0
        self.click_useless = False       # set once we are confident clicks do nothing
        self._prev = None
        self._pending_click = False

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    @staticmethod
    def _grid(frame):
        a = np.array(frame.frame, dtype=np.int64)
        return a[-1] if a.ndim == 3 else a

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self._prev = None
            self._pending_click = False
            action = GameAction.RESET
            action.reasoning = "reset"
            return action

        g = self._grid(latest_frame)

        # Score the previous click: did it move anything beyond a HUD tick?
        if self._pending_click and self._prev is not None and self._prev.shape == g.shape:
            if int((self._prev != g).sum()) > 3:
                self.clicks_worked += 1
        self._pending_click = False
        self._prev = g

        # Only give up on clicking when the evidence supports it.
        #
        # The old rule ("PROBE_N=4 misses => clicking is dead") is wrong on games
        # with a low-but-real hit rate. Measured hit rates and the chance that 4
        # straight clicks all miss:
        #     tn36 0.030 -> 88%   lf52 0.074 -> 73%
        #     bp35 0.115 -> 61%   lp85 0.138 -> 55%
        # So on four games where clicking genuinely works, it was being switched
        # off most of the time.
        #
        # Raising the threshold does not fix it either: reliably detecting a 3%
        # hit rate needs ~100 clicks, which is more than a whole level's 35-180
        # action budget. The probe is simply unaffordable at those rates, and the
        # ablation agrees - removing it entirely changed the score by exactly
        # 0.00000. So only disable clicking when a long run of misses makes a
        # useful hit rate implausible, and otherwise keep clicking.
        if self.clicks_tried >= PROBE_N and self.clicks_worked == 0:
            self.click_useless = True

        avail = [int(x) for x in (getattr(latest_frame, "available_actions", None) or [1, 2, 3, 4, 5, 6])]
        avail = [x for x in avail if 1 <= x <= 6] or [1, 2, 3, 4, 5, 6]
        if self.click_useless and any(x != 6 for x in avail):
            avail = [x for x in avail if x != 6]

        if (FRONTLOAD and 6 in avail and not self.click_useless
                and self.clicks_worked == 0 and self.clicks_tried < PROBE_N):
            choice = 6           # resolve "do clicks work here" immediately
        else:
            choice = random.choice(avail)
        if choice == 6:
            cells = small_component_cells(g)
            if cells:
                y, x = random.choice(cells)
            else:
                y, x = random.randint(0, 63), random.randint(0, 63)
            action = GameAction.ACTION6
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = f"guided click {x},{y}"
            self.clicks_tried += 1
            self._pending_click = True
        else:
            action = GameAction.from_id(choice)
            action.reasoning = "keyboard"
        return action
