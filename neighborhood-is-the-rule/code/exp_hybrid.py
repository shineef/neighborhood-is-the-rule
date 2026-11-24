"""Memorisation vs learning vs both, on identical data and identical rollout windows.

Held-out game protocol as exp_meta.py / exp_rollout_ft.py: collect(900, 23) -> first 300 train,
last 600 test; CNN = WM(RF) pretrained HY_PRE_STEPS Adam steps on the pooled other 24 games
(collect(3000, 11)) and finetuned HY_FT_EPOCHS on the 300 (the 'pretrain' arm). Rollout windows are
built exactly as in exp_rollout_ft.py (chains at seed 99, random.Random(5)), so every predictor below
is scored on the same 100 windows as the CNN rollouts there.

Predictors
  lookup_k{k}      memorised k x k windows, unseen -> no change
  lookup_backoff   largest seen window among HY_KS (most specific context first), else no change
  cnn              the finetuned network alone
  hybrid_k13       memorised 13 x 13 where seen, network elsewhere
  hybrid_backoff   backoff lookup where any window was seen, network elsewhere

Env: HY_SEED, HY_HELD, HY_RF=9, HY_KS=7,13,25,41, HY_PRE_STEPS=4000, HY_FT_EPOCHS=20,
     HY_NROLL=100, HY_DEPTHS=1,2,3,5,10, HY_OUT=hybrid.jsonl
"""
import os, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wmcache
from lookup_core import WindowLookup

NCOL = 16
SEED = int(os.environ.get("HY_SEED", "0"))
RF = int(os.environ.get("HY_RF", "9"))
WIDTH = 96
KS = [int(k) for k in os.environ.get("HY_KS", "7,13,25,41").split(",")]
PRE_STEPS = int(os.environ.get("HY_PRE_STEPS", "4000"))
FT_EPOCHS = int(os.environ.get("HY_FT_EPOCHS", "20"))
NROLL = int(os.environ.get("HY_NROLL", "100"))
DEPTHS = [int(d) for d in os.environ.get("HY_DEPTHS", "1,2,3,5,10").split(",")]
HELD = [g for g in os.environ.get("HY_HELD", "").split(",") if g]
OUT = os.environ.get("HY_OUT", "hybrid.jsonl")
TEST_N, EVENT_MIN = 300, 3
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
            cp[i, 0, cy, cx] = 1.0; cp[i, 1, cx, cy] = 1.0
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
def cnn_predict(model, G, A, XY):
    out = []
    for i in range(0, len(A), 64):
        out.append(model(torch.as_tensor(np.asarray(G[i:i + 64]), device=DEV).long(),
                         torch.as_tensor(np.asarray(A[i:i + 64]), device=DEV).long(),
                         torch.as_tensor(np.asarray(XY[i:i + 64]), device=DEV).long()).argmax(1).cpu().numpy())
    return np.concatenate(out)


def predictors(model, L):
    def backoff(G, A, XY, fallback=None):
        G = np.asarray(G)
        P = G.copy() if fallback is None else fallback(G, A, XY)
        filled = np.zeros(G.shape, bool)
        for k in sorted(L, reverse=True):           # most specific context first
            p, seen = L[k].predict(G, A, XY)
            m = seen & ~filled
            P[m] = p[m]; filled |= seen
        return P

    def hybrid13(G, A, XY):
        p, seen = L[13].predict(G, A, XY)
        c = cnn_predict(model, G, A, XY); c[seen] = p[seen]
        return c

    preds = {f"lookup_k{k}": (lambda k: (lambda G, A, XY: L[k].predict(G, A, XY)[0]))(k) for k in sorted(L)}
    preds["lookup_backoff"] = lambda G, A, XY: backoff(G, A, XY)
    preds["cnn"] = lambda G, A, XY: cnn_predict(model, G, A, XY)
    if 13 in L:
        preds["hybrid_k13"] = hybrid13
    preds["hybrid_backoff"] = lambda G, A, XY: backoff(G, A, XY, fallback=lambda g, a, xy: cnn_predict(model, g, a, xy))
    return preds


def one_step(P, G0, G1):
    hit = (P == G1).all(axis=(1, 2)); same = (G0 == G1).all(axis=(1, 2))
    chg = G0 != G1; ev = chg.reshape(len(G0), -1).sum(1) > EVENT_MIN
    pchg = P != G0; tp = (chg & pchg).sum()
    return dict(exact=float(hit.mean()), static=float(same.mean()),
                event_exact=float(hit[ev].mean()) if ev.any() else None, n_event=int(ev.sum()),
                precision=float(tp / max(pchg.sum(), 1)), recall=float(tp / max(chg.sum(), 1)))


def rollouts(pred, Wg0, Wa, Wg1, Wxy):
    cur = Wg0[:, 0].copy(); res = {}
    for k in range(1, max(DEPTHS) + 1):
        cur = pred(cur, Wa[:, k - 1], Wxy[:, k - 1])
        if k in DEPTHS:
            res[str(k)] = dict(exact=float((cur == Wg1[:, k - 1]).all(axis=(1, 2)).mean()),
                               static=float((Wg0[:, 0] == Wg1[:, k - 1]).all(axis=(1, 2)).mean()))
    return res


ALL = wmcache.all_games()
if not HELD:
    HELD = ALL
CFG = dict(seed=SEED, rf=RF, ks=KS, pre_steps=PRE_STEPS, ft_epochs=FT_EPOCHS, nroll=NROLL, depths=DEPTHS)
print("CONFIG " + json.dumps(CFG), flush=True)
maxd = max(DEPTHS)
for held in HELD:
    t0 = time.time()
    others = [g for g in ALL if g != held]
    pooled = T(tuple(np.concatenate(z) for z in zip(*[wmcache.collect(g, 3000, 11) for g in others])))
    d = wmcache.collect(held, TEST_N + 600, 23)
    tr = tuple(np.asarray(x[:TEST_N]) for x in d); te = tuple(np.asarray(x[TEST_N:]) for x in d)
    # rollout windows: identical construction to exp_rollout_ft.py
    chains = [c for c in wmcache.collect_chains(held, 4000, 99, maxd + 1) if len(c[0]) > maxd]
    rng = random.Random(5); win = []
    for _ in range(NROLL if chains else 0):
        c = rng.choice(chains); s = rng.randint(0, len(c[0]) - maxd - 1)
        win.append(tuple(np.asarray(x[s:s + maxd]) for x in c))
    W = tuple(np.stack(z) for z in zip(*win)) if win else None   # (N, maxd, ...) for G0, A, G1, XY

    torch.manual_seed(SEED)
    m = WM().to(DEV)
    sgd_steps(m, pooled, PRE_STEPS, torch.Generator().manual_seed(SEED))
    finetune(m, T(tr), FT_EPOCHS)
    L = {k: WindowLookup(k).fit(tr[0], tr[1], tr[2], tr[3]) for k in KS}
    rec = dict(game=held, **CFG, n_windows=len(win), pred={})
    for name, f in predictors(m, L).items():
        r = dict(one_step=one_step(f(te[0], te[1], te[3]), te[0], te[2]))
        if W is not None:
            r["rollout"] = rollouts(f, W[0], W[1], W[2], W[3])
        rec["pred"][name] = r
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"{held:<16} s{SEED} " + "  ".join(f"{n}={v['one_step']['exact']:.3f}" for n, v in rec["pred"].items())
          + f"  ({time.time()-t0:.0f}s)", flush=True)
print("\nHYBRID DONE")
