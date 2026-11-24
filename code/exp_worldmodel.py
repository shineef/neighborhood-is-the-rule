"""Can a network actually learn the dynamics, and how much context does it need?

exp_radius.py established a capacity-independent fact: with a large enough
patch, (patch, action) -> next colour is unambiguous, so the frame is a
sufficient statistic and the dynamics are deterministic. That is a statement
about the *data*. It does not say a learner can find the rule from a realistic
number of samples.

This trains an actual conv next-frame predictor at several receptive fields and
reports held-out accuracy on the cells that CHANGE (the static cells are ~98% of
the grid and make any model look good). The receptive-field sweep is the point:
if accuracy tracks the ambiguity curve we measured, we understand the problem;
if a small-RF net does fine, the ambiguity metric was too pessimistic; if even a
large-RF net fails, the rule is learnable-in-principle but not sample-efficient,
which would matter a lot for the 5x budget.

Receptive field of N stacked 3x3 convs is 2N+1, so depth is set from the target.

Usage: WM_GAMES=cd82-fb555c5d WM_RF=9,17,25,41 python exp_worldmodel.py
"""
import os, sys, random, time, json

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ARC-AGI-3-Agents"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arc_agi import Arcade
from arcengine import GameAction, GameState

GAMES = [g for g in os.environ.get("WM_GAMES", "cd82-fb555c5d,sp80-589a99af").split(",") if g]
RFS = [int(r) for r in os.environ.get("WM_RF", "9,17,25,41").split(",")]
STEPS = int(os.environ.get("WM_STEPS", "4000"))
EPOCHS = int(os.environ.get("WM_EPOCHS", "12"))
# Class-imbalance weight. "auto" sets it to (1-p)/p from the measured fraction of
# changed cells, which is what actually balances the two classes. A fixed 50 was
# tuned on cd82 (1.85% changed) and silently fails on sparser games: lp85 changes
# only 0.10% of cells, so weight 50 still leaves a 20:1 pull toward "no change"
# and the model collapses to the identity map - it scored 0.0015 on changed cells
# with 0.9998 on all cells, i.e. it had learned to copy the input again.
CHANGED_W_ENV = os.environ.get("WM_CHANGED_W", "auto")
# Click-only games (available_actions=[6]) make the action id uninformative:
# every transition has the same action, and what actually happened depends on
# WHERE the click landed - which the original model never received. With
# WM_CLICK=1 the click position is fed in as two extra planes (both (y,x) and
# (x,y) conventions, so we do not have to guess the engine's ordering).
USE_CLICK = os.environ.get("WM_CLICK", "0") == "1"
# A transition whose change is <= EVENT_MIN cells is a counter/HUD tick that
# happens on every action regardless of the click. Those dominate the
# changed-cell metric in this game family, so accuracy is also reported
# restricted to transitions where something bigger than a tick happened.
EVENT_MIN = int(os.environ.get("WM_EVENT_MIN", "3"))
NCOL = 16
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def grid(f):
    a = np.array(f.frame, dtype=np.int64)
    return a[-1] if a.ndim == 3 else a


def rollout(gid, steps, seed):
    rng = random.Random(seed)
    env = Arcade().make(gid)
    f = env.step(GameAction.RESET)
    G0, A, G1, XY = [], [], [], []
    for _ in range(steps):
        if getattr(f, "state", None) == GameState.GAME_OVER:
            f = env.step(GameAction.RESET)
            continue
        # The 5 games on this box declare available_actions=[6] (click-only);
        # sampling 1-6 uniformly makes 5/6 of transitions exact no-ops and
        # dilutes the changed-cell signal ~6x. Restrict to what the frame says
        # the game accepts (falls back to 1-6 if the field is absent).
        avail = [int(x) for x in (getattr(f, "available_actions", None) or [1, 2, 3, 4, 5, 6])]
        avail = [x for x in avail if 1 <= x <= 7] or [1, 2, 3, 4, 5, 6]
        cid = rng.choice(avail)
        a = GameAction.from_id(cid)
        d = {}
        if cid == 6:
            d = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}
            a.set_data(d)
        g0 = grid(f)
        f = env.step(a, data=d)
        g1 = grid(f)
        if g0.shape != (64, 64) or g1.shape != (64, 64):
            continue  # engine returns an empty frame on some level boundaries
        G0.append(g0); A.append(cid); G1.append(g1)
        XY.append((d.get("x", -1), d.get("y", -1)))
    return np.array(G0), np.array(A), np.array(G1), np.array(XY)


class Block(nn.Module):
    """Pre-norm residual 3x3 block. The first version of this experiment used a
    plain 20-deep conv stack and produced a non-monotonic accuracy curve (RF17
    scored worse than RF9), which is the signature of an optimisation failure,
    not of the task getting harder. Residual + GroupNorm removes that confound
    so the sweep measures receptive field rather than trainability."""

    def __init__(self, w):
        super().__init__()
        self.n = nn.GroupNorm(8, w)
        self.c = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, x):
        return x + self.c(F.relu(self.n(x)))


class WM(nn.Module):
    """Receptive field 2N+1 for N 3x3 layers. Action broadcast as extra planes."""

    def __init__(self, rf, width=96):
        super().__init__()
        n = max(1, (rf - 1) // 2)
        self.stem = nn.Conv2d(NCOL + 7 + (2 if USE_CLICK else 0), width, 3, padding=1)
        self.body = nn.Sequential(*[Block(width) for _ in range(n - 1)])
        self.head = nn.Conv2d(width, NCOL, 1)
        self.rf = 2 * n + 1
        self.n = n

    def forward(self, g, a, xy=None):
        x = F.one_hot(g, NCOL).permute(0, 3, 1, 2).float()
        ap = F.one_hot(a - 1, 7).float()[:, :, None, None].expand(-1, -1, g.shape[1], g.shape[2])
        planes = [x, ap]
        if USE_CLICK:
            B, H, W = g.shape
            cp = torch.zeros(B, 2, H, W, device=g.device)
            ok = (xy[:, 0] >= 0)
            if ok.any():
                i = torch.nonzero(ok).squeeze(1)
                cx = xy[i, 0].clamp(0, W - 1); cy = xy[i, 1].clamp(0, H - 1)
                cp[i, 0, cy, cx] = 1.0   # (row=y, col=x)
                cp[i, 1, cx, cy] = 1.0   # (row=x, col=y)
            planes.append(cp)
        return self.head(self.body(self.stem(torch.cat(planes, 1))))


for gid in GAMES:
    print(f"\n{'='*70}\n{gid}   device={DEV}\n{'='*70}", flush=True)
    t0 = time.time()
    G0, A, G1, XY = rollout(gid, STEPS, 11)
    n_tr = int(len(G0) * 0.8)
    changed = (G0 != G1)
    print(f"transitions={len(G0)}  train={n_tr}  changed cells={changed.sum()} "
          f"({100*changed.mean():.2f}% of all cells)", flush=True)

    p = float(changed.mean())
    CHANGED_W = ((1.0 - p) / p) if CHANGED_W_ENV == "auto" else float(CHANGED_W_ENV)
    print(f"  changed-cell loss weight = {CHANGED_W:.1f} "
          f"({'auto (1-p)/p' if CHANGED_W_ENV == 'auto' else 'fixed'})", flush=True)

    g0 = torch.from_numpy(G0).to(DEV); a = torch.from_numpy(A).to(DEV)
    g1 = torch.from_numpy(G1).to(DEV)
    xy = torch.from_numpy(XY).to(DEV)
    per_trans = changed.reshape(len(G0), -1).sum(1)
    n_event = int((per_trans[n_tr:] > EVENT_MIN).sum())
    print(f"  per-transition changed cells: median={np.median(per_trans):.0f} "
          f"max={per_trans.max()}  test transitions with >{EVENT_MIN} changed cells: "
          f"{n_event}/{len(G0)-n_tr}", flush=True)

    # Learning curve mode: hold the architecture fixed and vary how many
    # transitions the model gets. This is the number that decides whether the
    # world-model line survives contact with the 5x budget - a level allows
    # 35-180 actions, so if the model needs thousands of transitions to become
    # accurate, it cannot be learned *during* an episode and the prior has to
    # be amortised into training (plan 5.6).
    SIZES = [int(s) for s in os.environ.get("WM_TRAIN_SIZES", "").split(",") if s.strip()]
    if SIZES:
        rf = int(os.environ.get("WM_FIXED_RF", "25"))
        print(f"  learning curve at RF={rf}")
        print(f"  {'n_train':>8} {'changed-cell acc':>17} {'all-cell acc':>13}")
        for n in SIZES:
            n = min(n, n_tr)
            torch.manual_seed(0)
            m = WM(rf).to(DEV)
            opt = torch.optim.Adam(m.parameters(), lr=float(os.environ.get("WM_LR", "1e-3")))
            bs = 32
            for ep in range(EPOCHS):
                perm = torch.randperm(n, device=DEV)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]
                    per_cell = F.cross_entropy(m(g0[idx], a[idx], xy[idx]), g1[idx], reduction="none")
                    w = torch.where(g0[idx] != g1[idx], CHANGED_W, 1.0)
                    loss = (per_cell * w).sum() / w.sum()
                    opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                hc = tc = ha = ta = 0
                for i in range(n_tr, len(G0), 64):
                    sl = slice(i, min(i + 64, len(G0)))
                    pred = m(g0[sl], a[sl], xy[sl]).argmax(1); truth = g1[sl]
                    ch = (g0[sl] != truth)
                    hc += (pred[ch] == truth[ch]).sum().item(); tc += ch.sum().item()
                    ha += (pred == truth).sum().item(); ta += truth.numel()
            print(f"  {n:>8} {hc/max(tc,1):>17.4f} {ha/ta:>13.4f}", flush=True)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
        continue

    print(f"  {'target RF':>10} {'actual':>7} {'depth':>6} {'changed-cell acc':>17} {'event-cell acc':>15} {'all-cell acc':>13} {'params':>10}")
    for rf in RFS:
        torch.manual_seed(0)
        m = WM(rf).to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=float(os.environ.get("WM_LR", "1e-3")))
        bs = 32
        for ep in range(EPOCHS):
            perm = torch.randperm(n_tr, device=DEV)
            for i in range(0, n_tr, bs):
                idx = perm[i:i + bs]
                logits = m(g0[idx], a[idx], xy[idx])
                # Only ~1.8% of cells change, so plain cross-entropy is minimised
                # by the identity map - the first run scored 0.98 on all cells
                # (exactly the do-nothing baseline) while getting the changed
                # cells wrong, i.e. it had learned to copy the input. Upweight
                # the cells that actually move so the loss measures dynamics.
                per_cell = F.cross_entropy(logits, g1[idx], reduction="none")
                w = torch.where(g0[idx] != g1[idx], CHANGED_W, 1.0)
                loss = (per_cell * w).sum() / w.sum()
                opt.zero_grad(); loss.backward(); opt.step()

        m.eval()
        with torch.no_grad():
            hit_c = tot_c = hit_a = tot_a = hit_e = tot_e = 0
            for i in range(n_tr, len(G0), 64):
                sl = slice(i, min(i + 64, len(G0)))
                pred = m(g0[sl], a[sl], xy[sl]).argmax(1)
                truth = g1[sl]
                ch = (g0[sl] != truth)
                hit_c += (pred[ch] == truth[ch]).sum().item(); tot_c += ch.sum().item()
                hit_a += (pred == truth).sum().item(); tot_a += truth.numel()
                # restrict to transitions where more than a counter tick moved
                ev = ch.reshape(ch.shape[0], -1).sum(1) > EVENT_MIN
                if ev.any():
                    che = ch[ev]
                    hit_e += (pred[ev][che] == truth[ev][che]).sum().item()
                    tot_e += che.sum().item()
        print(f"  {rf:>10} {m.rf:>7} {m.n:>6} {hit_c/max(tot_c,1):>17.4f} "
              f"{(hit_e/tot_e if tot_e else float('nan')):>15.4f} "
              f"{hit_a/tot_a:>13.4f} {sum(p.numel() for p in m.parameters()):>10}", flush=True)

    print(f"  ({time.time()-t0:.0f}s)", flush=True)

print("\nWORLD MODEL SWEEP DONE")
