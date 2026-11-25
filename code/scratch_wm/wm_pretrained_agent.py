"""Agent that picks actions with a network pretrained on the other games.

Per episode:
  1. WMP_PHASE1 actions of guided exploration (sample from the available actions, aim clicks at small
     components, stop clicking once clicks are shown to do nothing), recording every (g0, action, click, g1).
  2. Load the leave-one-game-out checkpoint of this game as the initialization and finetune it on the
     recorded transitions. Refit every WMP_REFIT steps as more data arrives.
  3. Score each legal action by the size of the frame change the network predicts and take the largest
     (epsilon-greedy).

One class covers every arm:
  WMP_PRETRAINED=1  load the checkpoint (0 = random initialization)
  WMP_FINETUNE=1    finetune during the episode (0 = frozen pretrained weights)
  WMP_PHASE1, WMP_REFIT, WMP_EPS, WMP_EPOCHS_FT, WMP_LR, WMP_CANDS, WMP_MODE
The checkpoint must match WMP_RF and WMP_WIDTH; both default to the pretrainer's 9 and 96.
"""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from agents.agent import Agent
from arcengine import GameAction, GameState

# wm_core is a sibling file; the eval harness puts scratch_wm/ on sys.path.
from wm_core import WM, load_ckpt, small_component_cells, NCOL

CKPT_DIR = os.environ.get("WMP_CKPT_DIR",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts"))
USE_PRETRAINED = os.environ.get("WMP_PRETRAINED", "1") == "1"
USE_FINETUNE = os.environ.get("WMP_FINETUNE", "1") == "1"
MAX_COMPONENT = int(os.environ.get("WMP_MAX_COMP", "40"))
PROBE_CLICKS = int(os.environ.get("WMP_PROBE", "4"))
EVENT_CELLS = int(os.environ.get("WMP_EVENT", "3"))
PHASE1 = int(os.environ.get("WMP_PHASE1", "120"))
REFIT_EVERY = int(os.environ.get("WMP_REFIT", "60"))
EPSILON = float(os.environ.get("WMP_EPS", "0.30"))
EPOCHS_FT = int(os.environ.get("WMP_EPOCHS_FT", "8"))
LR = float(os.environ.get("WMP_LR", "1e-3"))
RF = int(os.environ.get("WMP_RF", "9"))
WIDTH = int(os.environ.get("WMP_WIDTH", "96"))
CAND_CLICKS = int(os.environ.get("WMP_CANDS", "8"))
# How the model turns predictions into a choice:
#   argmax  -> the action with the largest predicted frame change
#   filter  -> uniform among the actions whose predicted change exceeds EVENT_CELLS
MODE = os.environ.get("WMP_MODE", "argmax")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WMPretrainedAgent(Agent):
    MAX_ACTIONS = float("inf")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.clicks_tried = 0
        self.clicks_worked = 0
        self.clicks_useless = False
        self._prev = None
        self._pending = None
        self.buf_g0, self.buf_a, self.buf_xy, self.buf_g1 = [], [], [], []
        self.model = None
        self.model_ready = False          # True once a usable model exists
        self.n_since_fit = 0
        self.steps = 0
        self._ckpt_state = None           # cached pretrained weights for re-init
        self.used_model_steps = 0         # diagnostics: how often the model actually drove

    # --- checkpoint handling ------------------------------------------------
    def _prefix(self):
        gid = getattr(self, "game_id", "") or ""
        return gid.split("-")[0]

    def _new_model(self):
        """A fresh WM, initialised from the pretrained checkpoint when available."""
        torch.manual_seed(0)
        m = WM(rf=RF, width=WIDTH).to(DEV)
        if USE_PRETRAINED:
            if self._ckpt_state is None:
                path = os.path.join(CKPT_DIR, f"{self._prefix()}.pt")
                loaded, _ = load_ckpt(path, DEV)   # raises if missing -> caught by caller
                self._ckpt_state = loaded.state_dict()
            m.load_state_dict(self._ckpt_state)
        return m

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

    # --- training -----------------------------------------------------------
    def _fit(self):
        # Frozen pretrained model: build it once, never train.
        if USE_PRETRAINED and not USE_FINETUNE:
            if self.model is None:
                self.model = self._new_model(); self.model.eval()
                self.model_ready = True
            return
        if len(self.buf_g0) < 32:
            return
        g0 = torch.from_numpy(np.array(self.buf_g0)).to(DEV)
        a = torch.from_numpy(np.array(self.buf_a)).to(DEV)
        xy = torch.from_numpy(np.array(self.buf_xy)).to(DEV)
        g1 = torch.from_numpy(np.array(self.buf_g1)).to(DEV)
        if self.model is None:
            self.model = self._new_model()
        opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.model.train()
        n = len(self.buf_g0)
        for _ in range(EPOCHS_FT):
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, 32):
                idx = perm[i:i + 32]
                loss = F.cross_entropy(self.model(g0[idx], a[idx], xy[idx]), g1[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        self.model.eval()
        self.model_ready = True
        self.n_since_fit = 0

    def _score_candidates(self, g, legal):
        cands = []
        for aid in legal:
            if aid == 6:
                cells = small_component_cells(g, MAX_COMPONENT)
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
        if MODE == "filter":
            hits = [i for i in range(len(cands)) if delta[i] > EVENT_CELLS]
            if hits:
                return cands[random.choice(hits)]
        return cands[int(delta.argmax())]

    # --- policy -------------------------------------------------------------
    def choose_action(self, frames, latest_frame) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev, self._pending = None, None
            act = GameAction.RESET
            act.reasoning = "reset"
            return act

        g = self._grid(latest_frame)

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

        # (Re)fit once there is enough phase-1 data. A frozen pretrained model is
        # built immediately on the first call.
        need_refit = self.model is None or self.n_since_fit >= REFIT_EVERY
        if (len(self.buf_g0) >= min(PHASE1, 32) or (USE_PRETRAINED and not USE_FINETUNE)) and need_refit:
            try:
                self._fit()
            except Exception:
                self.model, self.model_ready = None, False

        use_model = (self.model_ready and g is not None
                     and self.steps > PHASE1 and random.random() > EPSILON)
        pick = None
        if use_model:
            try:
                pick = self._score_candidates(g, legal)
                if pick is not None:
                    self.used_model_steps += 1
            except Exception:
                pick = None

        if pick is None:      # guided fallback (exploration / model not ready)
            if (6 in legal and not self.clicks_useless
                    and self.clicks_worked == 0 and self.clicks_tried < PROBE_CLICKS):
                aid = 6
            else:
                aid = random.choice(legal)
            if aid == 6:
                cells = small_component_cells(g, MAX_COMPONENT) if g is not None else []
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
            act.reasoning = {"why": "wm-argmax" if use_model else "guided-click", "x": px, "y": py}
            self.clicks_tried += 1
        else:
            act = GameAction.from_id(aid)
            act.reasoning = "wm-argmax" if use_model else "guided-simple"
        self._pending = (aid, px, py)
        return act
