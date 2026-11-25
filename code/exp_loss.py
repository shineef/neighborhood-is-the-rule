"""Networks trained on one game alone, with a sweep of the loss weight on changing cells.

For each game in LS_GAMES, each view in LS_RFS and each weight in LS_WEIGHTS, a network is trained on the
first 80% of LS_STEPS random moves and scored on the rest:

  exact      share of held-out frames predicted exactly, next to the score of "nothing changes"
  precision  of the cells the network changes, the share that really change
  recall     of the cells that really change, the share the network changes
  fp/frame   wrongly changed cells per frame

The weight "auto" is (1 - p) / p, where p is the share of changing cells.

Env: LS_GAMES=cd82-fb555c5d,sp80-589a99af, LS_STEPS=4000, LS_EPOCHS=25, LS_RFS=25 (comma-separated views),
     LS_WEIGHTS=1,3,10,auto, LS_WIDTH=96, LS_EVENT_MIN=3
"""
import os, sys, random, time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ARC-AGI-3-Agents"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arc_agi import Arcade
from arcengine import GameAction, GameState

GAMES = [g for g in os.environ.get("LS_GAMES", "cd82-fb555c5d,sp80-589a99af").split(",") if g]
STEPS = int(os.environ.get("LS_STEPS", "4000"))
EPOCHS = int(os.environ.get("LS_EPOCHS", "25"))
RFS = [int(r) for r in os.environ.get("LS_RFS", os.environ.get("LS_RF", "25")).split(",")]
WEIGHTS = os.environ.get("LS_WEIGHTS", "1,3,10,auto").split(",")
WIDTH = int(os.environ.get("LS_WIDTH", "96"))
EVENT_MIN = int(os.environ.get("LS_EVENT_MIN", "3"))
RF_ENV = int(os.environ.get("LS_RF", "25"))
NCOL = 16
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def grid(f):
    a = np.array(f.frame, dtype=np.int64)
    return a[-1] if a.ndim == 3 else a


def collect(gid, steps, seed):
    rng = random.Random(seed)
    env = Arcade().make(gid)
    f = env.step(GameAction.RESET)
    G0, A, G1, XY = [], [], [], []
    for _ in range(steps):
        if getattr(f, "state", None) == GameState.GAME_OVER:
            f = env.step(GameAction.RESET)
            continue
        avail = [int(x) for x in (getattr(f, "available_actions", None) or [1, 2, 3, 4, 5, 6])]
        avail = [x for x in avail if 1 <= x <= 6] or [1, 2, 3, 4, 5, 6]
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
            continue
        G0.append(g0); A.append(cid); G1.append(g1)
        XY.append((d.get("x", -1), d.get("y", -1)))
    return np.array(G0), np.array(A), np.array(G1), np.array(XY)


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.n = nn.GroupNorm(8, w); self.c = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, x):
        return x + self.c(F.relu(self.n(x)))


class WM(nn.Module):
    def __init__(self, rf, width=96):
        super().__init__()
        n = max(1, (rf - 1) // 2)
        self.stem = nn.Conv2d(NCOL + 7 + 2, width, 3, padding=1)
        self.body = nn.Sequential(*[Block(width) for _ in range(n - 1)])
        self.head = nn.Conv2d(width, NCOL, 1)

    def forward(self, g, a, xy):
        x = F.one_hot(g, NCOL).permute(0, 3, 1, 2).float()
        ap = F.one_hot(a - 1, 7).float()[:, :, None, None].expand(-1, -1, g.shape[1], g.shape[2])
        B, H, W = g.shape
        cp = torch.zeros(B, 2, H, W, device=g.device)
        ok = xy[:, 0] >= 0
        if ok.any():
            i = torch.nonzero(ok).squeeze(1)
            cx = xy[i, 0].clamp(0, W - 1); cy = xy[i, 1].clamp(0, H - 1)
            cp[i, 0, cy, cx] = 1.0
            cp[i, 1, cx, cy] = 1.0
        return self.head(self.body(self.stem(torch.cat([x, ap, cp], 1))))


for gid in GAMES:
    print(f"\n{'='*78}\n{gid}   device={DEV}\n{'='*78}", flush=True)
    t0 = time.time()
    G0, A, G1, XY = collect(gid, STEPS, 11)
    p = float((G0 != G1).mean())
    auto = (1 - p) / max(p, 1e-9)
    n_tr = int(len(G0) * 0.8)
    g0 = torch.from_numpy(G0).to(DEV); a = torch.from_numpy(A).to(DEV)
    g1 = torch.from_numpy(G1).to(DEV); xy = torch.from_numpy(XY).to(DEV)

    st_exact = float(np.mean([np.array_equal(G0[i], G1[i]) for i in range(n_tr, len(G0))]))
    print(f"transitions={len(G0)}  changed={100*p:.2f}%  auto weight={auto:.0f}  "
          f"static-exact baseline={st_exact:.3f}", flush=True)
    ev_idx = [i for i in range(n_tr, len(G0)) if int((G0[i] != G1[i]).sum()) > EVENT_MIN]
    print(f"  event transitions in test set: {len(ev_idx)}/{len(G0)-n_tr}")
    print(f"  {'RF':>4} {'weight':>7} {'exact':>8} {'vs static':>10} {'EVENT-ex':>9} {'precision':>10} {'recall':>8} {'fp/frame':>9}")

    for RF in RFS:
     for wspec in WEIGHTS:
         W = auto if wspec == "auto" else float(wspec)
         torch.manual_seed(0)
         m = WM(RF, width=WIDTH).to(DEV)
         opt = torch.optim.Adam(m.parameters(), lr=1e-3)
         for ep in range(EPOCHS):
             perm = torch.randperm(n_tr, device=DEV)
             for i in range(0, n_tr, 32):
                 idx = perm[i:i + 32]
                 pc = F.cross_entropy(m(g0[idx], a[idx], xy[idx]), g1[idx], reduction="none")
                 w = torch.where(g0[idx] != g1[idx], W, 1.0)
                 (pc * w).sum().div(w.sum()).backward()
                 opt.step(); opt.zero_grad()
         m.eval()
         TP = FP = FN = 0
         ex = nf = 0
         ex_ev = nf_ev = 0
         with torch.no_grad():
             for i in range(n_tr, len(G0), 64):
                 sl = slice(i, min(i + 64, len(G0)))
                 pred = m(g0[sl], a[sl], xy[sl]).argmax(1)
                 truth = g1[sl]; s = g0[sl]
                 tc = (s != truth); pcm = (s != pred)
                 TP += int((tc & pcm).sum()); FP += int((~tc & pcm).sum())
                 FN += int((tc & ~pcm).sum())
                 hit = (pred == truth).all(dim=(1, 2))
                 ex += int(hit.sum()); nf += truth.shape[0]
                 # Exact-frame restricted to transitions where something actually
                 # happened. Plain exact-frame breaks down on games that mostly do
                 # nothing: lp85's do-nothing baseline is 0.982, so a model that
                 # learned NOTHING scores 0.982 and looks excellent. Counting only
                 # transitions with a real event is the only figure that still
                 # discriminates there.
                 ev = tc.flatten(1).sum(1) > EVENT_MIN
                 ex_ev += int((hit & ev).sum()); nf_ev += int(ev.sum())
         prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1)
         lbl = f"{W:.0f}" + ("(auto)" if wspec == "auto" else "")
         print(f"  {RF:>4} {lbl:>7} {ex/nf:>8.3f} {ex/nf - st_exact:>+10.3f} "
               f"{(ex_ev/nf_ev if nf_ev else float('nan')):>9.3f} {prec:>10.4f} "
               f"{rec:>8.4f} {FP/nf:>9.1f}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

print("\nLOSS SWEEP DONE")
