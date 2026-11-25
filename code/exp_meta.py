"""Network from scratch, pretrained on the other games, and with a Reptile initialization.

One game is held out at a time. The three arms share the architecture, the finetuning budget and the
evaluation:

    scratch   trained only on the held-out game's MT_TEST_N moves
    pretrain  MT_PRE_STEPS Adam steps on the pooled moves of the 24 other games, then the same finetuning
    reptile   Reptile initialization, then the same finetuning: MT_OUTER times, sample a game, take MT_INNER
              SGD steps on it (batch MT_INNER_BS, learning rate MT_INNER_LR) and move the initialization
              toward the adapted weights,  theta <- theta + MT_EPS * (theta_adapted - theta)

The held-out game is never in the pool.

Env: MT_SEED=0, MT_HELD (default all games), MT_RF=9, MT_WIDTH=96, MT_PER_GAME=3000, MT_TEST_N=300,
     MT_FT_EPOCHS=20, MT_EVENT_MIN=3, MT_OUTER=400, MT_INNER=10, MT_INNER_BS=32, MT_EPS=0.2, MT_INNER_LR=1e-3,
     MT_PRE_STEPS (default MT_OUTER * MT_INNER), MT_MEM_FRAC=0, MT_CACHE (folder of cached random play)
"""
import os, sys, random, time, json, glob

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ARC-AGI-3-Agents"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arc_agi import Arcade
from arcengine import GameAction, GameState

NCOL = 16
RF = int(os.environ.get("MT_RF", "9"))
WIDTH = int(os.environ.get("MT_WIDTH", "96"))
PER_GAME = int(os.environ.get("MT_PER_GAME", "3000"))
TEST_N = int(os.environ.get("MT_TEST_N", "300"))      # what a level's budget affords
FT_EPOCHS = int(os.environ.get("MT_FT_EPOCHS", "20"))  # finetune epochs, same for all arms
EVENT_MIN = int(os.environ.get("MT_EVENT_MIN", "3"))
HELD = [g for g in os.environ.get("MT_HELD", "").split(",") if g]
SEED = int(os.environ.get("MT_SEED", "0"))   # init + reptile sampling

# Reptile hyperparameters. INNER is deliberately close to what a level affords -
# the init is being tuned for adaptation at that specific budget, not in general.
OUTER = int(os.environ.get("MT_OUTER", "400"))     # meta-iterations
INNER = int(os.environ.get("MT_INNER", "10"))      # SGD steps per sampled game
INNER_BS = int(os.environ.get("MT_INNER_BS", "32"))
EPS = float(os.environ.get("MT_EPS", "0.2"))       # outer step size
INNER_LR = float(os.environ.get("MT_INNER_LR", "1e-3"))
# Ordinary pretraining gets the SAME number of gradient steps as reptile, so the
# comparison is compute-matched rather than epoch-matched.
PRE_STEPS = int(os.environ.get("MT_PRE_STEPS", str(OUTER * INNER)))

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optional cap on this process's share of GPU memory, for shared machines.
MEM_FRAC = float(os.environ.get("MT_MEM_FRAC", "0"))
if MEM_FRAC > 0 and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
CACHE = os.environ.get("MT_CACHE", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pt_cache"))


def grid(f):
    a = np.array(f.frame, dtype=np.int64)
    return a[-1] if a.ndim == 3 else a


def collect(gid, steps, seed):
    """Same on-disk cache as exp_pretrain.py - identical (gid, steps, seed) keys,
    so the two experiments share collected data instead of re-stepping the env."""
    if CACHE:
        os.makedirs(CACHE, exist_ok=True)
        path = os.path.join(CACHE, f"{gid.split('-')[0]}_{steps}_{seed}.npz")
        if os.path.exists(path):
            try:
                z = np.load(path)
                return (z["G0"], z["A"], z["G1"], z["XY"])
            except Exception:
                pass
    rng = random.Random(seed)
    env = Arcade().make(gid)
    f = env.step(GameAction.RESET)
    G0, A, G1, XY = [], [], [], []
    for _ in range(steps):
        if getattr(f, "state", None) == GameState.GAME_OVER:
            f = env.step(GameAction.RESET); continue
        avail = [int(x) for x in (getattr(f, "available_actions", None) or [1, 2, 3, 4, 5, 6])]
        avail = [x for x in avail if 1 <= x <= 6] or [1, 2, 3, 4, 5, 6]
        cid = rng.choice(avail)
        a = GameAction.from_id(cid); d = {}
        if cid == 6:
            d = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}; a.set_data(d)
        g0 = grid(f); f = env.step(a, data=d); g1 = grid(f)
        if g0.shape != (64, 64) or g1.shape != (64, 64):
            continue
        G0.append(g0); A.append(cid); G1.append(g1); XY.append((d.get("x", -1), d.get("y", -1)))
    out = (np.array(G0), np.array(A), np.array(G1), np.array(XY))
    if CACHE:
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, G0=out[0], A=out[1], G1=out[2], XY=out[3])
        os.replace(tmp, path)
    return out


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


def as_cpu_tensors(d):
    G0, A, G1, XY = d
    return (torch.from_numpy(G0), torch.from_numpy(A),
            torch.from_numpy(G1), torch.from_numpy(XY))


def batch(data, idx):
    g0, a, g1, xy = data
    return (g0[idx].to(DEV), a[idx].to(DEV), g1[idx].to(DEV), xy[idx].to(DEV))


def sgd_steps(model, data, steps, lr, bs, gen):
    """Plain SGD-with-Adam for `steps` minibatches drawn from `data`."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = len(data[0])
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), generator=gen)
        b0, ba, b1, bxy = batch(data, idx)
        loss = F.cross_entropy(model(b0, ba, bxy), b1)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def finetune(model, data, epochs, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = len(data[0])
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            b0, ba, b1, bxy = batch(data, perm[i:i + 32])
            loss = F.cross_entropy(model(b0, ba, bxy), b1)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def evaluate(model, data):
    g0, a, g1, xy = data
    ex = nf = ex_ev = nf_ev = st_ex = 0
    with torch.no_grad():
        for i in range(0, len(g0), 64):
            sl = slice(i, min(i + 64, len(g0)))
            pred = model(g0[sl].to(DEV), a[sl].to(DEV), xy[sl].to(DEV)).argmax(1)
            truth = g1[sl].to(DEV); src = g0[sl].to(DEV)
            hit = (pred == truth).all(dim=(1, 2))
            same = (src == truth).all(dim=(1, 2))
            ev = (src != truth).flatten(1).sum(1) > EVENT_MIN
            ex += int(hit.sum()); nf += truth.shape[0]; st_ex += int(same.sum())
            ex_ev += int((hit & ev).sum()); nf_ev += int(ev.sum())
    return dict(exact=ex / max(nf, 1), static=st_ex / max(nf, 1),
                event_exact=ex_ev / max(nf_ev, 1), n_event=nf_ev)


def reptile(pool_by_game, outer, inner, eps, seed=SEED):
    """theta <- theta + eps * (theta_adapted - theta), one sampled game per step."""
    torch.manual_seed(seed)
    meta = WM().to(DEV)
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    games = list(pool_by_game)
    for _ in range(outer):
        g = rng.choice(games)
        adapted = WM().to(DEV)
        adapted.load_state_dict(meta.state_dict())
        sgd_steps(adapted, pool_by_game[g], inner, INNER_LR, INNER_BS, gen)
        with torch.no_grad():
            for pm, pa in zip(meta.parameters(), adapted.parameters()):
                pm.add_(eps * (pa.detach() - pm))
            for bm, ba in zip(meta.buffers(), adapted.buffers()):
                bm.copy_(ba)
    meta.eval()
    return meta


ALL = sorted(json.load(open(m))["game_id"] for m in
             glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "environment_files", "*", "*", "metadata.json")))
if not HELD:
    HELD = ALL

CFG = dict(rf=RF, width=WIDTH, per_game=PER_GAME, test_n=TEST_N,
           ft_epochs=FT_EPOCHS, outer=OUTER, inner=INNER, eps=EPS,
           pre_steps=PRE_STEPS, inner_lr=INNER_LR, seed=SEED)
print("CONFIG " + json.dumps(CFG, sort_keys=True), flush=True)
print(f"{'held-out game':<20} {'regime':<10} {'exact':>8} {'static':>8} {'EVENT-ex':>9} {'n_ev':>6}")

for held in HELD:
    t0 = time.time()
    others = [g for g in ALL if g != held]
    pool_by_game = {g: as_cpu_tensors(collect(g, PER_GAME, 11)) for g in others}
    pooled = tuple(torch.cat([pool_by_game[g][i] for g in others]) for i in range(4))

    d = collect(held, TEST_N + 600, 23)
    tr = as_cpu_tensors(tuple(x[:TEST_N] for x in d))
    te = as_cpu_tensors(tuple(x[TEST_N:] for x in d))

    torch.manual_seed(SEED)
    m = WM().to(DEV); finetune(m, tr, FT_EPOCHS)
    r = evaluate(m, te)
    print(f"{held:<20} {'scratch':<10} {r['exact']:>8.3f} {r['static']:>8.3f} "
          f"{r['event_exact']:>9.3f} {r['n_event']:>6}", flush=True)

    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    m = WM().to(DEV); sgd_steps(m, pooled, PRE_STEPS, 1e-3, 32, gen)
    finetune(m, tr, FT_EPOCHS)
    r = evaluate(m, te)
    print(f"{held:<20} {'pretrain':<10} {r['exact']:>8.3f} {r['static']:>8.3f} "
          f"{r['event_exact']:>9.3f} {r['n_event']:>6}", flush=True)

    m = reptile(pool_by_game, OUTER, INNER, EPS)
    finetune(m, tr, FT_EPOCHS)
    r = evaluate(m, te)
    print(f"{held:<20} {'reptile':<10} {r['exact']:>8.3f} {r['static']:>8.3f} "
          f"{r['event_exact']:>9.3f} {r['n_event']:>6}  ({time.time()-t0:.0f}s)", flush=True)

print("\nMETA DONE")
