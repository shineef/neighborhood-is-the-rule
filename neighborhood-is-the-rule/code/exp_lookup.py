"""How far does pure memorisation of local rules go on a held-out game?

E1 (exp_radius.py) showed the next colour of a cell is a deterministic function of
the action and a wide enough window around it (0% ambiguity at 7-41 cells), and
exp_learnability.py memorised 3x3 keys, which were too narrow. This memorises
WIDE windows (k x k colours + click marker) from the same 300 local transitions
the CNN finetunes on (collect(held, TEST_N+600, 23): first TEST_N train, last
600 test, the exp_pretrain.py / exp_meta.py protocol) and predicts "no change"
for any key it has never seen. It reports the same metrics as exp_pretrain.py
(exact, static, EVENT-exact) plus coverage, so it can sit in the same table as
scratch / pretrain / finetune.

It also runs a data curve on a separate split: collect(held, CURVE_MAX+600, 31),
last 600 test, first n train for n in LK_CURVE.

Env: LK_HELD (default all 25), LK_K (window widths, default 3,7,13,25,41),
     LK_TEST_N=300, LK_CURVE=50,100,300,1000,3000 (empty to skip), LK_ROLL=1 for
     closed-loop rollouts on contiguous runs of the test slice.
Output: one JSON line per (game, protocol, n, k) to LK_OUT (default lookup_out.jsonl).
"""
import os, sys, json, glob, random, time
import numpy as np
import torch
import torch.nn.functional as F

TEST_N = int(os.environ.get("LK_TEST_N", "300"))
KS = [int(k) for k in os.environ.get("LK_K", "3,7,13,25,41").split(",") if k]
CURVE = [int(n) for n in os.environ.get("LK_CURVE", "50,100,300,1000,3000").split(",") if n]
EVENT_MIN = int(os.environ.get("LK_EVENT_MIN", "3"))
HELD = [g for g in os.environ.get("LK_HELD", "").split(",") if g]
OUT = os.environ.get("LK_OUT", "lookup_out.jsonl")
ROLL = os.environ.get("LK_ROLL", "1") == "1"
DEPTHS = [1, 2, 3, 5, 10]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# data: shared cache, identical protocol and keys to exp_pretrain.collect
from wmcache import collect, all_games


# ---------------------------------------------------------------- window memoriser
class WindowLookup:
    """(k x k window of colours + click marker, action) -> next colour of the centre.
    Windows are hashed exactly with two random integer convolutions in float64.
    Unseen keys predict 'no change'."""

    def __init__(self, k, seed=0):
        self.k = k
        g = torch.Generator().manual_seed(1234 + seed)
        self.W = torch.randint(1, 1 << 20, (2, 2, k, k), generator=g).double().to(DEV)

    def keys(self, G, A, XY):
        out = []
        for i in range(0, len(A), 64):
            s = torch.as_tensor(G[i:i + 64], device=DEV).double() + 1.0
            B = s.shape[0]
            click = torch.zeros_like(s)
            xy = torch.as_tensor(XY[i:i + 64], device=DEV).long()
            m = xy[:, 0] >= 0
            if m.any():
                bi = torch.nonzero(m).squeeze(1)
                click[bi, xy[m, 1].clamp(0, 63), xy[m, 0].clamp(0, 63)] = 1.0
            h = F.conv2d(torch.stack([s, click], 1), self.W, padding=self.k // 2)
            h1, h2 = h[:, 0].long() % (1 << 27), h[:, 1].long() % (1 << 28)
            act = torch.as_tensor(A[i:i + 64], device=DEV).long().view(B, 1, 1).expand(B, 64, 64)
            out.append(((h1 << 31) | (h2 << 3) | (act & 7)).cpu().numpy().reshape(B, -1))
        return np.concatenate(out)

    def fit(self, G0, A, G1, XY):
        k = self.keys(G0, A, XY).ravel()
        c = G1.reshape(-1).astype(np.int64)
        u, cnt = np.unique((k << 4) | c, return_counts=True)
        uk, uc = u >> 4, u & 15
        order = np.lexsort((-cnt, uk))
        uk, uc = uk[order], uc[order]
        first = np.ones(len(uk), bool); first[1:] = uk[1:] != uk[:-1]
        self.kk, self.cc = uk[first], uc[first]
        return self

    def predict(self, G0, A, XY):
        key = self.keys(G0, A, XY)
        pos = np.clip(np.searchsorted(self.kk, key), 0, len(self.kk) - 1)
        seen = self.kk[pos] == key
        p = G0.reshape(len(G0), -1).copy()
        p[seen] = self.cc[pos[seen]]
        return p.reshape(G0.shape), seen.reshape(G0.shape)


def metrics(G0, G1, P, seen):
    hit = (P == G1).all(axis=(1, 2))
    same = (G0 == G1).all(axis=(1, 2))
    chg = G0 != G1
    ev = chg.reshape(len(G0), -1).sum(1) > EVENT_MIN
    pchg = P != G0
    tp = (chg & pchg).sum()
    return dict(exact=float(hit.mean()), static=float(same.mean()),
                event_exact=float(hit[ev].mean()) if ev.any() else None, n_event=int(ev.sum()),
                precision=float(tp / max(pchg.sum(), 1)), recall=float(tp / max(chg.sum(), 1)),
                coverage=float(seen.mean()), coverage_changed=float(seen[chg].mean()) if chg.any() else None)


def rollouts(L, G0, A, G1, XY):
    """Closed loop on consecutive test transitions that continue each other."""
    cont = np.array([(G1[i] == G0[i + 1]).all() for i in range(len(A) - 1)] + [False])
    res = {}
    for k in DEPTHS:
        st = [t for t in range(len(A) - k + 1) if cont[t:t + k - 1].all()]
        if not st:
            continue
        st = np.array(st[:300])
        cur = G0[st].copy()
        for j in range(k):
            cur, _ = L.predict(cur, A[st + j], XY[st + j])
        res[k] = dict(exact=float((cur == G1[st + k - 1]).all(axis=(1, 2)).mean()),
                      static=float((G0[st] == G1[st + k - 1]).all(axis=(1, 2)).mean()), n=int(len(st)))
    return res


def emit(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


ALL = all_games()
if not HELD:
    HELD = ALL
print("CONFIG " + json.dumps(dict(test_n=TEST_N, ks=KS, curve=CURVE, event_min=EVENT_MIN)), flush=True)
for held in HELD:
    t0 = time.time()
    d = collect(held, TEST_N + 600, 23)
    tr = tuple(x[:TEST_N] for x in d); te = tuple(x[TEST_N:] for x in d)
    for k in KS:
        L = WindowLookup(k).fit(tr[0], tr[1], tr[2], tr[3])
        P, seen = L.predict(te[0], te[1], te[3])
        r = metrics(te[0], te[2], P, seen)
        if ROLL:
            r["rollout"] = rollouts(L, te[0], te[1], te[2], te[3])
        emit(dict(game=held, protocol="standard", n=TEST_N, k=k, **r))
        print(f"{held:<16} std n={TEST_N:<5} k={k:<3} exact={r['exact']:.3f} static={r['static']:.3f} "
              f"ev={r['event_exact']} cov_chg={r['coverage_changed']}", flush=True)
    if CURVE:
        mx = max(CURVE)
        dc = collect(held, mx + 600, 31)
        te_c = tuple(x[-600:] for x in dc)
        for n in CURVE:
            trc = tuple(x[:min(n, len(dc[0]) - 600)] for x in dc)
            for k in KS:
                L = WindowLookup(k).fit(trc[0], trc[1], trc[2], trc[3])
                P, seen = L.predict(te_c[0], te_c[1], te_c[3])
                r = metrics(te_c[0], te_c[2], P, seen)
                emit(dict(game=held, protocol="curve", n=int(len(trc[0])), k=k, **r))
        print(f"{held:<16} curve done", flush=True)
    print(f"{held} ({time.time()-t0:.0f}s)", flush=True)
print("\nLOOKUP DONE")
