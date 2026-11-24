"""Guided exploration + a world model trained inside the episode, used to pick actions.

This is the first integration of the two lines. Everything it relies on was
measured first:

  - Dynamics are deterministic and Markov in the frame, so a learned transition
    function is well-posed (300/300 identical replays; 0.0% ambiguity at a large
    enough receptive field).
  - 19 of 25 games reach >=0.80 changed-cell accuracy within 200 transitions,
    and a level allows 35-180 actions - so a model CAN be fitted inside an
    episode rather than amortised into training.
  - One-step exact-frame accuracy is 0.30-0.77 with plain cross-entropy
    (weight=1). Two-step is 0.05-0.52. So one-step lookahead is defensible;
    deep search is not, and is deliberately not attempted here.
  - "Something important happened" is visible as a large frame delta - the same
    signal that separates level completions perfectly (100th percentile in
    12/12 games). So predicted delta size is a usable proxy for progress.

Policy: spend the first PHASE1 actions on the measured click/action heuristics
while recording transitions, fit the model, then score each legal action by the
size of the change the model predicts and take the argmax (epsilon-greedy so it
keeps gathering data).

The honest risk: at 0.30 one-step accuracy on the hardest games the ranking may
be noise, in which case this degrades to guided exploration with extra latency.
That is exactly what the A/B against GuidedAgent measures.
"""
from __future__ import annotations

import os
import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

NCOL = 16
MAX_COMPONENT = int(os.environ.get("WMA_MAX_COMP", "40"))
PROBE_CLICKS = int(os.environ.get("WMA_PROBE", "4"))
EVENT_CELLS = int(os.environ.get("WMA_EVENT", "3"))
PHASE1 = int(os.environ.get("WMA_PHASE1", "120"))     # actions of pure exploration
REFIT_EVERY = int(os.environ.get("WMA_REFIT", "60"))  # refit cadence after phase 1
EPSILON = float(os.environ.get("WMA_EPS", "0.30"))    # keep exploring after phase 1
RF = int(os.environ.get("WMA_RF", "9"))               # small: must train in seconds
WIDTH = int(os.environ.get("WMA_WIDTH", "48"))
EPOCHS = int(os.environ.get("WMA_EPOCHS", "8"))
CAND_CLICKS = int(os.environ.get("WMA_CANDS", "8"))   # click targets scored per step
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.n = nn.GroupNorm(8, w)
        self.c = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, x):
        return x + self.c(F.relu(self.n(x)))


class _WM(nn.Module):
    def __init__(self, rf=RF, width=WIDTH):
        super().__init__()
        n = max(1, (rf - 1) // 2)
        self.stem = nn.Conv2d(NCOL + 7 + 2, width, 3, padding=1)
        self.body = nn.Sequential(*[_Block(width) for _ in range(n - 1)])
        self.head = nn.Conv2d(width, NCOL, 1)

    def forward(self, g, a, xy):
        x = F.one_hot(g, NCOL).permute(0, 3, 1, 2).float()
        ap = F.one_hot(a - 1, 7).float()[:, :, None, None].expand(-1, -1, g.shape[1], g.shape[2])
        B, H, W = g.shape
        cp = torch.zeros(B, 2, H, W, device=g.device)
        ok = xy[:, 0] >= 0
        if ok.any():
            i = torch.nonzero(ok).squeeze(1)
            cx = xy[i, 0].clamp(0, W - 1)
            cy = xy[i, 1].clamp(0, H - 1)
            cp[i, 0, cy, cx] = 1.0
            cp[i, 1, cx, cy] = 1.0
        return self.head(self.body(self.stem(torch.cat([x, ap, cp], 1))))


def _small_components(g, max_size=MAX_COMPONENT):
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    out = []
    for y in range(h):
        for x in range(w):
            if seen[y, x]:
                continue
            col = g[y, x]
            q = deque([(y, x)])
            seen[y, x] = True
            blob = []
            while q:
                cy, cx = q.popleft()
                blob.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(blob) <= max_size:
                out.extend(blob)
    return out


class WMAgent(Agent):
    MAX_ACTIONS = float("inf")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.clicks_tried = 0
        self.clicks_worked = 0
        self.clicks_useless = False
        self._prev = None
        self._pending = None            # (action_id, x, y) awaiting its outcome
        self.buf_g0, self.buf_a, self.buf_xy, self.buf_g1 = [], [], [], []
        self.model = None
        self.n_since_fit = 0
        self.steps = 0

    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state is GameState.WIN

    @staticmethod
    def _grid(frame):
        try:
            a = np.array(frame.frame, dtype=np.int64)
        except Exception:
            return None
        if a.ndim == 3:
            a = a[-1]
        return a if a.shape == (64, 64) else None

    def _fit(self):
        if len(self.buf_g0) < 32:
            return
        g0 = torch.from_numpy(np.array(self.buf_g0)).to(DEV)
        a = torch.from_numpy(np.array(self.buf_a)).to(DEV)
        xy = torch.from_numpy(np.array(self.buf_xy)).to(DEV)
        g1 = torch.from_numpy(np.array(self.buf_g1)).to(DEV)
        if self.model is None:
            torch.manual_seed(0)
            self.model = _WM().to(DEV)
        opt = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        self.model.train()
        n = len(self.buf_g0)
        for _ in range(EPOCHS):
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, 32):
                idx = perm[i:i + 32]
                # weight=1: weighting changed cells wrecks precision and with it
                # exact-frame accuracy, which is what a planner consumes.
                loss = F.cross_entropy(self.model(g0[idx], a[idx], xy[idx]), g1[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        self.model.eval()
        self.n_since_fit = 0

    def _score_candidates(self, g, legal):
        """Predicted change size for each candidate action; bigger = more eventful."""
        cands = []
        for aid in legal:
            if aid == 6:
                cells = _small_components(g)
                random.shuffle(cells)
                for (y, x) in cells[:CAND_CLICKS]:
                    cands.append((6, int(x), int(y)))
            else:
                cands.append((aid, -1, -1))
        if not cands:
            return None
        gg = torch.from_numpy(np.repeat(g[None], len(cands), 0)).to(DEV)
        aa = torch.tensor([c[0] for c in cands], device=DEV)
        xx = torch.tensor([[c[1], c[2]] for c in cands], device=DEV)
        with torch.no_grad():
            pred = self.model(gg, aa, xx).argmax(1)
        delta = (pred != gg).flatten(1).sum(1).cpu().numpy()
        return cands[int(delta.argmax())], delta

    def choose_action(self, frames, latest_frame) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev, self._pending = None, None
            act = GameAction.RESET
            act.reasoning = "reset"
            return act

        g = self._grid(latest_frame)

        # record the outcome of the previous action
        if self._pending is not None and g is not None and self._prev is not None:
            aid, px, py = self._pending
            self.buf_g0.append(self._prev); self.buf_a.append(aid)
            self.buf_xy.append((px, py)); self.buf_g1.append(g)
            self.n_since_fit += 1
            if aid == 6 and int((self._prev != g).sum()) > EVENT_CELLS:
                self.clicks_worked += 1
        self._prev = g
        self._pending = None
        self.steps += 1

        if self.clicks_tried >= PROBE_CLICKS and self.clicks_worked == 0:
            self.clicks_useless = True

        legal = [int(a) for a in (getattr(latest_frame, "available_actions", None) or [])]
        legal = [a for a in legal if 1 <= a <= 6] or [1, 2, 3, 4, 5, 6]
        if self.clicks_useless and any(a != 6 for a in legal):
            legal = [a for a in legal if a != 6]

        if len(self.buf_g0) >= PHASE1 and (self.model is None or self.n_since_fit >= REFIT_EVERY):
            try:
                self._fit()
            except Exception:
                self.model = None

        use_model = (self.model is not None and g is not None
                     and self.steps > PHASE1 and random.random() > EPSILON)
        pick = None
        if use_model:
            try:
                best, _ = self._score_candidates(g, legal)
                pick = best
            except Exception:
                pick = None

        if pick is None:
            if (6 in legal and not self.clicks_useless
                    and self.clicks_worked == 0 and self.clicks_tried < PROBE_CLICKS):
                aid = 6
            else:
                aid = random.choice(legal)
            if aid == 6:
                cells = _small_components(g) if g is not None else []
                if cells:
                    y, x = random.choice(cells)
                else:
                    y, x = random.randint(0, 63), random.randint(0, 63)
                pick = (6, int(x), int(y))
            else:
                pick = (aid, -1, -1)

        aid, px, py = pick
        if aid == 6:
            act = GameAction.ACTION6
            act.set_data({"x": px, "y": py})
            act.reasoning = {"why": "world-model argmax" if use_model else "guided click",
                             "x": px, "y": py}
            self.clicks_tried += 1
        else:
            act = GameAction.from_id(aid)
            act.reasoning = "world-model argmax" if use_model else "guided simple"
        self._pending = (aid, px, py)
        return act
