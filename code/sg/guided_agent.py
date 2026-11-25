"""Random play with three priors. No model, no training.

  1. Sample only from `available_actions`.
  2. Stop clicking in a game where GUIDED_PROBE clicks in a row had no effect.
  3. Aim clicks at small connected components of one color (at most GUIDED_MAX_COMP cells) instead of a
     uniform cell; fall back to a uniform cell when there is no such component.

Env: GUIDED_MAX_COMP=40, GUIDED_PROBE=30, GUIDED_FRONTLOAD=0
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
FRONTLOAD = os.environ.get("GUIDED_FRONTLOAD", "0") == "1"   # 1 spends the probe clicks at the start of a game


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
        self.click_useless = False       # set once clicks are shown to do nothing
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

        # Give up on clicking only after a long run of misses. In some games few clicks land on an active
        # cell (hit rates of 3 to 14% in tn36, lf52, bp35 and lp85), so a handful of misses in a row is not
        # evidence that clicking does nothing.
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
