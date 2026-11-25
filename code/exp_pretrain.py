"""Pretraining across games: transfer without finetuning, number of games, moves per game.

One game is held out. Three arms, same architecture:

  scratch   trained only on the held-out game's PT_TEST_N moves
  pretrain  trained on the other games and scored on the held-out game without any finetuning
  finetune  pretrained, then finetuned on the held-out game's PT_TEST_N moves

PT_N_GAMES limits the pool to that many of the 24 other games (chosen with PT_SAMPLE_SEED); PT_PER_GAME sets
the number of moves taken from each. The held-out game is never in the pool.

Env: PT_HELD (default all games), PT_RF=9, PT_WIDTH=96, PT_EPOCHS=12, PT_PER_GAME=3000, PT_TEST_N=300,
     PT_EVENT_MIN=3, PT_N_GAMES=0 (all 24), PT_SAMPLE_SEED=7, PT_CACHE (folder of cached random play)
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
RF = int(os.environ.get("PT_RF", "9"))
WIDTH = int(os.environ.get("PT_WIDTH", "96"))
EPOCHS = int(os.environ.get("PT_EPOCHS", "12"))
PER_GAME = int(os.environ.get("PT_PER_GAME", "3000"))   # transitions per training game
TEST_N = int(os.environ.get("PT_TEST_N", "300"))        # transitions the held-out game gets
EVENT_MIN = int(os.environ.get("PT_EVENT_MIN", "3"))
HELD = [g for g in os.environ.get("PT_HELD", "").split(",") if g]
N_GAMES = int(os.environ.get("PT_N_GAMES", "0"))   # 0 = use all 24
SAMPLE_SEED = int(os.environ.get("PT_SAMPLE_SEED", "7"))  # which games land in the pool
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def grid(f):
    a = np.array(f.frame, dtype=np.int64)
    return a[-1] if a.ndim == 3 else a


CACHE = os.environ.get("PT_CACHE", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pt_cache"))


def collect(gid, steps, seed):
    """Cached on disk. The training pool is rebuilt for every held-out game, but
    collection is deterministic in (gid, steps, seed), so all 25 held-out runs
    were replaying the identical 24-game pool from scratch - 19.2M env steps at
    PT_PER_GAME=8000 where 0.77M distinct ones exist. Env stepping is
    single-threaded CPU, so the GPU sat idle through it."""
    if CACHE:
        os.makedirs(CACHE, exist_ok=True)
        path = os.path.join(CACHE, f"{gid.split('-')[0]}_{steps}_{seed}.npz")
        if os.path.exists(path):
            try:
                z = np.load(path)
                return (z["G0"], z["A"], z["G1"], z["XY"])
            except Exception:
                pass                      # corrupt/partial file: just recollect
    out = _collect_raw(gid, steps, seed)
    if CACHE:
        # atomic: concurrent workers share this dir. Write through a file object -
        # np.savez_compressed(str) appends ".npz" and the rename would miss.
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, G0=out[0], A=out[1], G1=out[2], XY=out[3])
        os.replace(tmp, path)
    return out


def _collect_raw(gid, steps, seed):
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
    return (np.array(G0), np.array(A), np.array(G1), np.array(XY))


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


def to_dev(d):
    """Keep the pool in CPU memory and ship batches to the GPU on demand.

    Moving it wholesale OOM'd at PT_PER_GAME=8000: 24 games x 8000 int64 64x64
    frames is ~6.3 GB for G0 alone, doubled for G1, on a 23.5 GB card. The pool
    only ever gets touched 32 rows at a time, so residency bought nothing - and
    the ceiling would have blocked the PT_N_GAMES=24 arm too."""
    G0, A, G1, XY = d
    return (torch.from_numpy(G0), torch.from_numpy(A),
            torch.from_numpy(G1), torch.from_numpy(XY))


def train(model, data, epochs, lr=1e-3):
    g0, a, g1, xy = data
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = len(g0)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            b0 = g0[idx].to(DEV, non_blocking=True); ba = a[idx].to(DEV, non_blocking=True)
            b1 = g1[idx].to(DEV, non_blocking=True); bxy = xy[idx].to(DEV, non_blocking=True)
            loss = F.cross_entropy(model(b0, ba, bxy), b1)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def evaluate(model, data):
    g0, a, g1, xy = data
    ex = nf = ex_ev = nf_ev = 0
    st_ex = st_ev = 0
    with torch.no_grad():
        for i in range(0, len(g0), 64):
            sl = slice(i, min(i + 64, len(g0)))
            pred = model(g0[sl].to(DEV), a[sl].to(DEV), xy[sl].to(DEV)).argmax(1)
            truth = g1[sl].to(DEV); src = g0[sl].to(DEV)
            hit = (pred == truth).all(dim=(1, 2))
            same = (src == truth).all(dim=(1, 2))
            ev = (src != truth).flatten(1).sum(1) > EVENT_MIN
            ex += int(hit.sum()); nf += truth.shape[0]
            st_ex += int(same.sum())
            ex_ev += int((hit & ev).sum()); nf_ev += int(ev.sum())
            st_ev += int((same & ev).sum())
    return dict(exact=ex / max(nf, 1), static=st_ex / max(nf, 1),
                event_exact=ex_ev / max(nf_ev, 1), n_event=nf_ev)


ALL = sorted(json.load(open(m))["game_id"] for m in
             glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "environment_files", "*", "*", "metadata.json")))
if not HELD:
    HELD = ALL

# Stamp every run with the config that produced it. Two arms of the PT_N_GAMES
# sweep were launched on different machines with PT_EPOCHS 12 vs 20, and the
# curve looked like a real diversity effect until the scratch baseline - which
# does not touch the pretraining pool and must be identical - came out 0.354 vs
# 0.294. Without the stamp, that mismatch is only findable by remembering which
# shell each arm was started from.
CFG = dict(rf=RF, width=WIDTH, epochs=EPOCHS, per_game=PER_GAME,
           test_n=TEST_N, n_games=N_GAMES, event_min=EVENT_MIN,
           sample_seed=SAMPLE_SEED)
print("CONFIG " + json.dumps(CFG, sort_keys=True), flush=True)
print(f"{'held-out game':<20} {'regime':<10} {'exact':>8} {'static':>8} {'EVENT-ex':>9} {'n_ev':>6}")
for held in HELD:
    t0 = time.time()
    others = [g for g in ALL if g != held]
    # How many GAMES to pretrain on, as opposed to how many steps per game.
    # Scaling steps saturates almost immediately (1500 -> 6000 per game moved
    # finetune exact 0.4526 -> 0.4608), so the interesting axis is diversity.
    # Subsample deterministically per held-out game so the comparison is paired.
    if N_GAMES and N_GAMES < len(others):
        others = random.Random(SAMPLE_SEED).sample(others, N_GAMES)
    # training pool from the other games
    pool = []
    for g in others:
        pool.append(collect(g, PER_GAME, 11))
    G0 = np.concatenate([p[0] for p in pool]); A = np.concatenate([p[1] for p in pool])
    G1 = np.concatenate([p[2] for p in pool]); XY = np.concatenate([p[3] for p in pool])
    pool_t = to_dev((G0, A, G1, XY))

    # held-out game: small train slice + test slice
    d = collect(held, TEST_N + 600, 23)
    tr = to_dev(tuple(x[:TEST_N] for x in d))
    te = to_dev(tuple(x[TEST_N:] for x in d))

    torch.manual_seed(0); m_s = WM().to(DEV); train(m_s, tr, EPOCHS)
    r = evaluate(m_s, te)
    print(f"{held:<20} {'scratch':<10} {r['exact']:>8.3f} {r['static']:>8.3f} "
          f"{r['event_exact']:>9.3f} {r['n_event']:>6}", flush=True)

    torch.manual_seed(0); m_p = WM().to(DEV); train(m_p, pool_t, max(1, EPOCHS // 4))
    r = evaluate(m_p, te)
    print(f"{held:<20} {'pretrain':<10} {r['exact']:>8.3f} {r['static']:>8.3f} "
          f"{r['event_exact']:>9.3f} {r['n_event']:>6}", flush=True)

    train(m_p, tr, EPOCHS)
    r = evaluate(m_p, te)
    print(f"{held:<20} {'finetune':<10} {r['exact']:>8.3f} {r['static']:>8.3f} "
          f"{r['event_exact']:>9.3f} {r['n_event']:>6}  ({time.time()-t0:.0f}s)", flush=True)

print("\nPRETRAIN DONE")
