"""Shared, cached random-play collection. Identical protocol and cache keys to
exp_pretrain.collect / exp_meta.collect, so all scripts read the same pt_cache files.
Also collects contiguous chains for closed-loop rollouts, as in exp_horizon.collect."""
import os, random
import numpy as np
from arc_agi import Arcade
from arcengine import GameAction, GameState

CACHE = os.environ.get("PT_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pt_cache"))


def grid(f):
    a = np.array(f.frame, dtype=np.int64)
    return a[-1] if a.ndim == 3 else a


def _step_random(rng, f):
    avail = [int(x) for x in (getattr(f, "available_actions", None) or [1, 2, 3, 4, 5, 6])]
    avail = [x for x in avail if 1 <= x <= 6] or [1, 2, 3, 4, 5, 6]
    cid = rng.choice(avail)
    a = GameAction.from_id(cid); d = {}
    if cid == 6:
        d = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}; a.set_data(d)
    return cid, a, d


def _save(path, **arrs):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrs)
    os.replace(tmp, path)


def collect(gid, steps, seed):
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
        cid, a, d = _step_random(rng, f)
        g0 = grid(f); f = env.step(a, data=d); g1 = grid(f)
        if g0.shape != (64, 64) or g1.shape != (64, 64):
            continue
        G0.append(g0); A.append(cid); G1.append(g1); XY.append((d.get("x", -1), d.get("y", -1)))
    out = (np.array(G0), np.array(A), np.array(G1), np.array(XY))
    _save(path, G0=out[0], A=out[1], G1=out[2], XY=out[3])
    return out


def collect_chains(gid, steps, seed, min_len):
    """Contiguous runs of transitions (no reset, no dropped frame inside), as in
    exp_horizon.collect(need_chain=...). Cached as flat arrays plus chain ids."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{gid.split('-')[0]}_chains_{steps}_{seed}_{min_len}.npz")
    if os.path.exists(path):
        z = np.load(path)
        G0, A, G1, XY, C = z["G0"], z["A"], z["G1"], z["XY"], z["C"]
        return [(G0[C == c], A[C == c], G1[C == c], XY[C == c]) for c in np.unique(C)]
    rng = random.Random(seed)
    env = Arcade().make(gid)
    f = env.step(GameAction.RESET)
    chains, cur = [], []

    def close():
        if len(cur) >= min_len:
            chains.append(list(cur))
        cur.clear()

    for _ in range(steps):
        if getattr(f, "state", None) == GameState.GAME_OVER:
            f = env.step(GameAction.RESET); close(); continue
        cid, a, d = _step_random(rng, f)
        g0 = grid(f); f = env.step(a, data=d); g1 = grid(f)
        if g0.shape != (64, 64) or g1.shape != (64, 64):
            close(); continue
        cur.append((g0, cid, (d.get("x", -1), d.get("y", -1)), g1))
    close()
    if chains:
        G0 = np.array([t[0] for ch in chains for t in ch]); A = np.array([t[1] for ch in chains for t in ch])
        XY = np.array([t[2] for ch in chains for t in ch]); G1 = np.array([t[3] for ch in chains for t in ch])
        C = np.concatenate([np.full(len(ch), i) for i, ch in enumerate(chains)])
        _save(path, G0=G0, A=A, G1=G1, XY=XY, C=C)
    return [(np.array([t[0] for t in ch]), np.array([t[1] for t in ch]),
             np.array([t[3] for t in ch]), np.array([t[2] for t in ch])) for ch in chains]


def all_games(code_dir=None):
    import glob, json
    code_dir = code_dir or os.path.dirname(os.path.abspath(__file__))
    return sorted(json.load(open(m))["game_id"] for m in
                  glob.glob(os.path.join(code_dir, "environment_files", "*", "*", "metadata.json")))
