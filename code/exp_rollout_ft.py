"""How far ahead can the BEST world model we have roll out?

E2.2 / E30 measured closed-loop rollouts only for models trained from scratch on one
game. The strongest model the project produced is the cross-game pretrained one
finetuned on the held-out game's 300 transitions (E9: 0.45 one-step exact vs 0.35
from scratch). This measures its closed-loop exact-frame accuracy at depth k, next to
scratch and the identity ("nothing changes") baseline, and records the one-step
exact on the standard 600-transition test slice so the compounding law
exact(k) ~ exact(1)^k can be checked game by game.

Arms (same WM class, RF, width, finetune budget as exp_meta.py):
  scratch   finetune from random init on the held-out 300 transitions (FT_EPOCHS)
  pretrain  RO_PRE_STEPS Adam steps on the pooled other 24 games, then the same finetune
Rollouts: RO_NROLL windows drawn from contiguous chains collected at seed 99 (as in
exp_horizon.py); exact match of the whole frame at depths RO_DEPTHS.

Env: RO_SEED, RO_HELD, RO_RF=9, RO_WIDTH=96, RO_PRE_STEPS=4000, RO_FT_EPOCHS=20,
     RO_NROLL=100, RO_DEPTHS=1,2,3,5,10, RO_OUT=rollout_ft.jsonl,
     RO_CLICK_T=1 (0 leaves the second click plane empty, see WM.forward)
"""
import os, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wmcache

NCOL = 16
SEED = int(os.environ.get("RO_SEED", "0"))
RF = int(os.environ.get("RO_RF", "9"))
WIDTH = int(os.environ.get("RO_WIDTH", "96"))
PRE_STEPS = int(os.environ.get("RO_PRE_STEPS", "4000"))
FT_EPOCHS = int(os.environ.get("RO_FT_EPOCHS", "20"))
TEST_N = 300
NROLL = int(os.environ.get("RO_NROLL", "100"))
DEPTHS = [int(d) for d in os.environ.get("RO_DEPTHS", "1,2,3,5,10").split(",")]
EVENT_MIN = 3
HELD = [g for g in os.environ.get("RO_HELD", "").split(",") if g]
OUT = os.environ.get("RO_OUT", "rollout_ft.jsonl")
CLICK_T = os.environ.get("RO_CLICK_T", "1") == "1"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Block(nn.Module):
    def __init__(self, w):
        super().__init__(); self.n = nn.GroupNorm(8, w); self.c = nn.Conv2d(w, w, 3, padding=1)
    def forward(self, x): return x + self.c(F.relu(self.n(x)))


class WM(nn.Module):
    def __init__(self, rf=RF, width=WIDTH):
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
            cp[i, 0, cy, cx] = 1.0          # the clicked cell: frames are indexed [row y, column x]
            if CLICK_T:
                cp[i, 1, cx, cy] = 1.0      # the same mark at the transposed cell. Every network in the paper
                                            # has it: it was added before the engine's coordinate order was known.
        return self.head(self.body(self.stem(torch.cat([x, ap, cp], 1))))


def T(d):
    return tuple(torch.from_numpy(np.asarray(x)) for x in d)


def batch(d, idx):
    return tuple(x[idx].to(DEV) for x in d)


def sgd_steps(model, data, steps, gen, lr=1e-3, bs=32):
    opt = torch.optim.Adam(model.parameters(), lr=lr); model.train(); n = len(data[0])
    for _ in range(steps):
        b0, ba, b1, bxy = batch(data, torch.randint(0, n, (min(bs, n),), generator=gen))
        loss = F.cross_entropy(model(b0, ba, bxy), b1)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); return model


def finetune(model, data, epochs, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr); model.train(); n = len(data[0])
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            b0, ba, b1, bxy = batch(data, perm[i:i + 32])
            loss = F.cross_entropy(model(b0, ba, bxy), b1)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); return model


@torch.no_grad()
def one_step(model, d):
    g0, a, g1, xy = d
    hit = same = ev_hit = ev_n = 0
    for i in range(0, len(g0), 64):
        sl = slice(i, i + 64)
        p = model(g0[sl].to(DEV), a[sl].to(DEV), xy[sl].to(DEV)).argmax(1).cpu()
        h = (p == g1[sl]).all(dim=(1, 2)); s = (g0[sl] == g1[sl]).all(dim=(1, 2))
        ev = (g0[sl] != g1[sl]).flatten(1).sum(1) > EVENT_MIN
        hit += int(h.sum()); same += int(s.sum()); ev_hit += int((h & ev).sum()); ev_n += int(ev.sum())
    n = len(g0)
    return dict(exact=hit / n, static=same / n, event_exact=ev_hit / max(ev_n, 1), n_event=ev_n)


@torch.no_grad()
def rollouts(model, windows):
    maxd = max(DEPTHS)
    ex = {k: 0 for k in DEPTHS}; st = {k: 0 for k in DEPTHS}
    for (g0, a, g1, xy) in windows:
        cur = torch.from_numpy(g0[:1]).to(DEV)
        for k in range(1, maxd + 1):
            cur = model(cur, torch.from_numpy(a[k - 1:k]).to(DEV), torch.from_numpy(xy[k - 1:k]).to(DEV)).argmax(1)
            if k in ex:
                ex[k] += int((cur[0].cpu().numpy() == g1[k - 1]).all())
                st[k] += int((g0[0] == g1[k - 1]).all())
    n = len(windows)
    return {str(k): dict(exact=ex[k] / n, static=st[k] / n) for k in DEPTHS}, n


ALL = wmcache.all_games()
if not HELD:
    HELD = ALL
CFG = dict(seed=SEED, rf=RF, width=WIDTH, pre_steps=PRE_STEPS, ft_epochs=FT_EPOCHS, nroll=NROLL, depths=DEPTHS, click_t=CLICK_T)
print("CONFIG " + json.dumps(CFG), flush=True)
maxd = max(DEPTHS)
for held in HELD:
    t0 = time.time()
    others = [g for g in ALL if g != held]
    pooled = T(tuple(np.concatenate(z) for z in zip(*[wmcache.collect(g, 3000, 11) for g in others])))
    d = wmcache.collect(held, TEST_N + 600, 23)
    tr = T(tuple(x[:TEST_N] for x in d)); te = T(tuple(x[TEST_N:] for x in d))
    chains = [c for c in wmcache.collect_chains(held, 4000, 99, maxd + 1) if len(c[0]) > maxd]
    rng = random.Random(5)
    windows = []
    for _ in range(NROLL if chains else 0):
        c = rng.choice(chains); s = rng.randint(0, len(c[0]) - maxd - 1)
        windows.append(tuple(np.asarray(x[s:s + maxd]) for x in c))
    for arm in ("scratch", "pretrain"):
        torch.manual_seed(SEED)
        m = WM().to(DEV)
        if arm == "pretrain":
            sgd_steps(m, pooled, PRE_STEPS, torch.Generator().manual_seed(SEED))
        finetune(m, tr, FT_EPOCHS)
        r1 = one_step(m, te)
        ro, n = rollouts(m, windows) if windows else ({}, 0)
        rec = dict(game=held, arm=arm, **CFG, one_step=r1, rollout=ro, n_windows=n)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{held:<16} {arm:<9} s{SEED} 1-step {r1['exact']:.3f} (static {r1['static']:.3f}) "
              + " ".join(f"d{k}={v['exact']:.2f}/{v['static']:.2f}" for k, v in ro.items()), flush=True)
    print(f"{held} ({time.time()-t0:.0f}s)", flush=True)
print("\nROLLOUT FT DONE")
