"""Scheduled sampling for the next-frame network, on consecutive moves.

A network trained with teacher forcing only ever sees true frames, but a planner feeds it its own
predictions. Scheduled sampling (Bengio et al., 2015) closes that gap during training: with some
probability the network is given its own prediction of frame t+1 instead of the true one and has to
produce frame t+2 from it.

Arms, trained from the same initialisation, with the same batches and the same sampled batches:
  tf        teacher forcing: (g_t, a_t) -> g_{t+1}
  ss_next   scheduled sampling: in a sampled batch the input is the network's own prediction of g_{t+1},
            the action is the NEXT one, a_{t+1}, and the target is g_{t+2}. Moves whose successor is not
            the next recorded move (reset, dropped frame, end of the training slice) stay teacher-forced.
  ss_same   own prediction of g_{t+1} with the SAME action a_t, fitted to g_{t+1}. This is what
            exp_horizon.py does with HZ_SCHED_SAMPLE > 0; it tells the network that repeating an action
            leaves the frame as it is. Kept only to show its effect next to the correct pairing.

Data and rollout windows follow exp_horizon.py: SC_STEPS random moves (seed 11), first 80% for training;
SC_NROLL windows drawn with random.Random(5) from unbroken runs of a second random play (seed 99).

Env: SC_GAMES (ids or four-letter prefixes), SC_SEED, SC_ARMS=tf,ss_next,ss_same, SC_RATE=0.15, SC_RF=25,
     SC_WIDTH=96, SC_STEPS=4000, SC_EPOCHS=25, SC_DEPTHS=1,2,3,5,10,20, SC_NROLL=60, SC_OUT=sched.jsonl,
     SC_PREP=1 (only collect and cache the data, then exit)
"""
import os, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wmcache

NCOL = 16
GAMES = [g for g in os.environ.get("SC_GAMES", "sp80,cd82,tr87,ar25,m0r0,re86,sc25,tu93").split(",") if g]
SEED = int(os.environ.get("SC_SEED", "0"))
ARMS = [a for a in os.environ.get("SC_ARMS", "tf,ss_next,ss_same").split(",") if a]
RATE = float(os.environ.get("SC_RATE", "0.15"))
RF = int(os.environ.get("SC_RF", "25"))
WIDTH = int(os.environ.get("SC_WIDTH", "96"))
STEPS = int(os.environ.get("SC_STEPS", "4000"))
EPOCHS = int(os.environ.get("SC_EPOCHS", "25"))
DEPTHS = [int(d) for d in os.environ.get("SC_DEPTHS", "1,2,3,5,10,20").split(",")]
NROLL = int(os.environ.get("SC_NROLL", "60"))
OUT = os.environ.get("SC_OUT", "sched.jsonl")
PREP = os.environ.get("SC_PREP", "0") == "1"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAXD = max(DEPTHS)


class Block(nn.Module):
    def __init__(self, w):
        super().__init__(); self.n = nn.GroupNorm(8, w); self.c = nn.Conv2d(w, w, 3, padding=1)
    def forward(self, x): return x + self.c(F.relu(self.n(x)))


class WM(nn.Module):
    """Same network as exp_horizon.py / exp_rollout_ft.py."""
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


def resolve(g):
    if "-" in g:
        return g
    hit = [x for x in wmcache.all_games() if x.split("-")[0] == g]
    if not hit:
        raise SystemExit(f"unknown game {g}")
    return hit[0]


def load(gid):
    G0, A, G1, XY = wmcache.collect(gid, STEPS, 11)
    chains = [c for c in wmcache.collect_chains(gid, STEPS, 99, MAXD + 1) if len(c[0]) > MAXD]
    return (G0, A, G1, XY), chains


def train(arm, data, n_tr, nxt_ok):
    """One network. All arms draw the same initial weights, the same batches and the same sampled batches."""
    g0, a, g1, xy = data
    torch.manual_seed(SEED)
    coin = random.Random(1000 + SEED)
    m = WM().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n_sampled = n_two_step = 0
    for ep in range(EPOCHS):
        perm = torch.randperm(n_tr, device=DEV)
        for i in range(0, n_tr, 32):
            idx = perm[i:i + 32]
            sampled = ep >= EPOCHS // 3 and coin.random() < RATE      # drawn for every arm, used by two
            src, act, pos, tgt = g0[idx], a[idx], xy[idx], g1[idx]
            if sampled and arm == "ss_same":
                with torch.no_grad():
                    src = m(src, act, pos).argmax(1)
                n_sampled += 1; n_two_step += len(idx)
            elif sampled and arm == "ss_next":
                two = nxt_ok[idx]                                         # successor is the next recorded move
                if two.any():
                    j = idx[two]
                    with torch.no_grad():
                        pred = m(g0[j], a[j], xy[j]).argmax(1)
                    src = src.clone(); act = act.clone(); pos = pos.clone(); tgt = tgt.clone()
                    src[two] = pred; act[two] = a[j + 1]; pos[two] = xy[j + 1]; tgt[two] = g1[j + 1]
                n_sampled += 1; n_two_step += int(two.sum())
            loss = F.cross_entropy(m(src, act, pos), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m, n_sampled, n_two_step


@torch.no_grad()
def one_step(m, data, lo):
    g0, a, g1, xy = data
    hit = same = 0
    for i in range(lo, len(g0), 64):
        sl = slice(i, i + 64)
        hit += int((m(g0[sl], a[sl], xy[sl]).argmax(1) == g1[sl]).all(dim=(1, 2)).sum())
        same += int((g0[sl] == g1[sl]).all(dim=(1, 2)).sum())
    n = len(g0) - lo
    return hit / max(n, 1), same / max(n, 1)


@torch.no_grad()
def rollouts(m, W):
    Wg0, Wa, Wg1, Wxy = W                                               # (N, MAXD, ...)
    cur = Wg0[:, 0]
    res = {}
    for k in range(1, MAXD + 1):
        cur = m(cur, Wa[:, k - 1], Wxy[:, k - 1]).argmax(1)
        if k in DEPTHS:
            res[str(k)] = dict(exact=float((cur == Wg1[:, k - 1]).all(dim=(1, 2)).float().mean()),
                               static=float((Wg0[:, 0] == Wg1[:, k - 1]).all(dim=(1, 2)).float().mean()))
    return res


GIDS = [resolve(g) for g in GAMES]
if PREP:
    for gid in GIDS:
        t0 = time.time(); d, ch = load(gid)
        print(f"{gid} cached: {len(d[0])} moves, {len(ch)} runs of more than {MAXD} moves ({time.time()-t0:.0f}s)", flush=True)
    raise SystemExit(0)

CFG = dict(seed=SEED, rate=RATE, rf=RF, width=WIDTH, steps=STEPS, epochs=EPOCHS, depths=DEPTHS, nroll=NROLL)
print("CONFIG " + json.dumps(CFG) + f" arms={ARMS} device={DEV}", flush=True)
for gid in GIDS:
    (G0, A, G1, XY), chains = load(gid)
    if not chains:
        print(f"{gid}: no unbroken run longer than {MAXD} moves - skipped", flush=True); continue
    n_tr = int(len(G0) * 0.8)
    data = tuple(torch.from_numpy(np.asarray(x)).to(DEV) for x in (G0, A, G1, XY))
    # move i can be chained to move i+1 if the frame it produced is the frame move i+1 started from
    ok = np.zeros(len(G0), bool)
    ok[:-1] = (G1[:-1] == G0[1:]).all(axis=(1, 2))
    ok[n_tr - 1:] = False                                               # the successor must be a training move too
    nxt_ok = torch.from_numpy(ok).to(DEV)
    rng = random.Random(5); win = []                                    # windows exactly as in exp_horizon.py
    for _ in range(NROLL):
        c = rng.choice(chains); s = rng.randint(0, len(c[0]) - MAXD - 1)
        win.append(tuple(np.asarray(x[s:s + MAXD]) for x in c))
    W = tuple(torch.from_numpy(np.stack(z)).to(DEV) for z in zip(*win))
    for arm in ARMS:
        t0 = time.time()
        m, n_s, n_2 = train(arm, data, n_tr, nxt_ok)
        te, te_static = one_step(m, data, n_tr)
        ro = rollouts(m, W)
        rec = dict(game=gid, arm=arm, **CFG, n_train=n_tr, chainable=float(ok[:n_tr].mean()), sampled_batches=n_s,
                   two_step_samples=n_2, test_exact=te, test_static=te_static, rollout=ro, secs=round(time.time() - t0))
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{gid:<16} {arm:<8} s{SEED} test {te:.3f} (static {te_static:.3f}) "
              + " ".join(f"d{k}={v['exact']:.2f}/{v['static']:.2f}" for k, v in ro.items())
              + f"  [{n_s} sampled batches, {n_2} samples] ({time.time()-t0:.0f}s)", flush=True)
    del data, W
print("\nSCHED DONE")
