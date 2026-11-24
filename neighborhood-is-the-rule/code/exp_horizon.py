"""How far can the world model actually roll out before it drifts?

Everything reported so far is ONE-STEP accuracy: given a true frame and an
action, predict the next frame. Median 0.979 across 25 games sounds strong, but
planning does not consume one-step predictions - it feeds the model its own
output, repeatedly. Errors compound, and a model that is 98% right per step can
be useless at depth 20.

**This is the assumption the entire planning layer rests on and it has never
been tested.** Measured here: train as before, then roll the model forward from
a real state using a real action sequence, feeding predictions back in, and
compare against the true trajectory at each depth.

Two metrics, because they fail differently:
  exact   - fraction of rollouts still bit-identical to the truth at depth k.
            This is what a planner needs: one wrong cell can invalidate a plan.
  cells   - mean cell agreement at depth k. Degrades gracefully and shows
            whether the model is drifting slowly or collapsing.

Also reports the same for the trivial "assume nothing changes" model, since on
these grids that is a strong baseline and any honest comparison needs it.
"""
import os, sys, random, time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ARC-AGI-3-Agents"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arc_agi import Arcade
from arcengine import GameAction, GameState

GAMES = [g for g in os.environ.get("HZ_GAMES", "cd82-fb555c5d,sp80-589a99af,tr87-cd924810").split(",") if g]
STEPS = int(os.environ.get("HZ_STEPS", "4000"))
EPOCHS = int(os.environ.get("HZ_EPOCHS", "25"))
RF = int(os.environ.get("HZ_RF", "25"))
DEPTHS = [int(d) for d in os.environ.get("HZ_DEPTHS", "1,2,3,5,10,20,40").split(",")]
NROLL = int(os.environ.get("HZ_ROLLOUTS", "60"))
SS = float(os.environ.get("HZ_SCHED_SAMPLE", "0"))   # closed-loop training rate
NCOL = 16
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def grid(f):
    a = np.array(f.frame, dtype=np.int64)
    return a[-1] if a.ndim == 3 else a


def collect(gid, steps, seed, need_chain=0):
    """Returns transitions, and (if need_chain) contiguous chains for rollout."""
    rng = random.Random(seed)
    env = Arcade().make(gid)
    f = env.step(GameAction.RESET)
    G0, A, G1, XY = [], [], [], []
    chains = []
    cur = []
    for _ in range(steps):
        if getattr(f, "state", None) == GameState.GAME_OVER:
            f = env.step(GameAction.RESET)
            if len(cur) >= need_chain and need_chain:
                chains.append(cur)
            cur = []
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
            if need_chain and len(cur) >= need_chain:
                chains.append(cur)
            cur = []
            continue
        G0.append(g0); A.append(cid); G1.append(g1)
        XY.append((d.get("x", -1), d.get("y", -1)))
        cur.append((g0, cid, (d.get("x", -1), d.get("y", -1)), g1))
    if need_chain and len(cur) >= need_chain:
        chains.append(cur)
    return np.array(G0), np.array(A), np.array(G1), np.array(XY), chains


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
    print(f"\n{'='*72}\n{gid}   device={DEV}\n{'='*72}", flush=True)
    t0 = time.time()
    maxd = max(DEPTHS)
    G0, A, G1, XY, _ = collect(gid, STEPS, 11)
    _, _, _, _, chains = collect(gid, STEPS, 99, need_chain=maxd + 1)
    p = float((G0 != G1).mean())
    # weight=1 (plain cross-entropy). The (1-p)/p weight this used to carry was
    # a leftover workaround: it maximised recall on changed cells while driving
    # precision to 0.16 and 386 false-positive cells per frame, so the model
    # never produced an exact frame and lost to the do-nothing baseline. The
    # weight sweep (exp_loss.py) shows weight=1 is the best operating point by a
    # wide margin - cd82 exact 0.343 vs static 0.221, sp80 0.598 vs 0.000.
    W = float(os.environ.get("HZ_WEIGHT", "1"))
    print(f"train transitions={len(G0)}  changed={100*p:.2f}%  weight={W:.0f}  "
          f"eval chains={len(chains)}", flush=True)
    if not chains:
        print("  no contiguous chain long enough - skipping")
        continue

    g0 = torch.from_numpy(G0).to(DEV); a = torch.from_numpy(A).to(DEV)
    g1 = torch.from_numpy(G1).to(DEV); xy = torch.from_numpy(XY).to(DEV)
    n_tr = int(len(G0) * 0.8)

    torch.manual_seed(0)
    m = WM(RF).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        perm = torch.randperm(n_tr, device=DEV)
        for i in range(0, n_tr, 32):
            idx = perm[i:i + 32]
            src = g0[idx]
            # Scheduled sampling. Teacher forcing trains only on TRUE frames, but
            # at plan time the model consumes its OWN output, so it never sees the
            # slightly-wrong inputs it will actually face - which is exactly why
            # exact-frame accuracy collapses by depth 3-5. With probability SS we
            # feed back the model's own prediction for one step first.
            if SS > 0 and ep >= EPOCHS // 3 and random.random() < SS:
                with torch.no_grad():
                    src = m(src, a[idx], xy[idx]).argmax(1)
            pc = F.cross_entropy(m(src, a[idx], xy[idx]), g1[idx], reduction="none")
            w = torch.where(g0[idx] != g1[idx], W, 1.0)
            loss = (pc * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()

    # closed-loop rollout: feed the model its own prediction
    starts = []
    rng = random.Random(5)
    for _ in range(NROLL):
        ch = rng.choice(chains)
        if len(ch) <= maxd:
            continue
        s = rng.randint(0, len(ch) - maxd - 1)
        starts.append(ch[s:s + maxd])
    if not starts:
        print("  not enough chain material")
        continue

    exact = {d: 0 for d in DEPTHS}
    cells = {d: [] for d in DEPTHS}
    s_exact = {d: 0 for d in DEPTHS}
    s_cells = {d: [] for d in DEPTHS}
    with torch.no_grad():
        for seq in starts:
            cur = torch.from_numpy(seq[0][0][None]).to(DEV)
            static = seq[0][0]
            for k, (tg0, cid, (cx, cy), tg1) in enumerate(seq, start=1):
                av = torch.tensor([cid], device=DEV)
                axy = torch.tensor([[cx, cy]], device=DEV)
                cur = m(cur, av, axy).argmax(1)
                if k in exact:
                    pred = cur[0].cpu().numpy()
                    exact[k] += int(np.array_equal(pred, tg1))
                    cells[k].append(float((pred == tg1).mean()))
                    s_exact[k] += int(np.array_equal(static, tg1))
                    s_cells[k].append(float((static == tg1).mean()))
    n = len(starts)
    print(f"  {'depth':>6} {'exact':>8} {'cells':>8}   {'static exact':>13} {'static cells':>13}")
    for d in DEPTHS:
        print(f"  {d:>6} {exact[d]/n:>8.3f} {np.mean(cells[d]):>8.4f}   "
              f"{s_exact[d]/n:>13.3f} {np.mean(s_cells[d]):>13.4f}", flush=True)
    print(f"  ({time.time()-t0:.0f}s, {n} rollouts)", flush=True)

print("\nHORIZON DONE")
