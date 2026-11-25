"""How wide a neighborhood the next color of a cell depends on.

For each radius r in EXP_RADII, every key (neighborhood of side 2r + 1, action) seen in EXP_STEPS random
moves is checked for conflicts: the same key followed by different colors. The conflict rate does not depend
on any learner; it is a lower bound on the error of every predictor keyed on that neighborhood.

Env: EXP_GAMES=sp80-589a99af,cd82-fb555c5d, EXP_STEPS=600, EXP_RADII=1,2,3,5
"""
import os, sys, random, time
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ARC-AGI-3-Agents"))
import numpy as np
from arc_agi import Arcade
from arcengine import GameAction, GameState

GAMES = [g for g in os.environ.get("EXP_GAMES", "sp80-589a99af,cd82-fb555c5d").split(",") if g]
STEPS = int(os.environ.get("EXP_STEPS", "600"))
RADII = [int(r) for r in os.environ.get("EXP_RADII", "1,2,3,5").split(",")]


def grid(f):
    a = np.array(f.frame, dtype=np.int16)
    return a[-1] if a.ndim == 3 else a


def rollout(gid, steps, seed):
    rng = random.Random(seed)
    arc = Arcade()
    env = arc.make(gid)
    f = env.step(GameAction.RESET)
    out = []
    for _ in range(steps):
        if getattr(f, "state", None) == GameState.GAME_OVER:
            f = env.step(GameAction.RESET)
            continue
        cid = rng.choice([1, 2, 3, 4, 5, 6])
        a = GameAction.from_id(cid)
        d = {}
        if cid == 6:
            d = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}
            a.set_data(d)
        g0 = grid(f)
        f = env.step(a, data=d)
        out.append((g0, cid, grid(f)))
    return out


def conflict_rate(trans, r):
    """Fraction of cell observations whose (patch, action) key is ambiguous.

    Only cells that actually change are counted: the overwhelming majority of
    cells are static, and including them drowns the signal in trivially-easy
    predictions.
    """
    seen = {}
    amb = set()
    n_changed = 0
    for g0, cid, g1 in trans:
        p = np.pad(g0, r, mode="constant", constant_values=-1)
        ys, xs = np.nonzero(g0 != g1)
        n_changed += len(ys)
        for y, x in zip(ys, xs):
            k = (p[y:y + 2 * r + 1, x:x + 2 * r + 1].tobytes(), cid)
            v = int(g1[y, x])
            if k in seen:
                if seen[k] != v:
                    amb.add(k)
            else:
                seen[k] = v
    # how many changed-cell observations fall on an ambiguous key
    bad = 0
    for g0, cid, g1 in trans:
        p = np.pad(g0, r, mode="constant", constant_values=-1)
        ys, xs = np.nonzero(g0 != g1)
        for y, x in zip(ys, xs):
            k = (p[y:y + 2 * r + 1, x:x + 2 * r + 1].tobytes(), cid)
            if k in amb:
                bad += 1
    return bad / max(n_changed, 1), len(seen), n_changed


for gid in GAMES:
    t0 = time.time()
    trans = rollout(gid, STEPS, 7)
    print(f"\n{gid}: {len(trans)} transitions", flush=True)
    print(f"  {'radius':>7} {'patch':>7} {'ambiguous changed-cells':>24} {'distinct keys':>14}")
    for r in RADII:
        rate, keys, nch = conflict_rate(trans, r)
        print(f"  {r:>7} {2*r+1}x{2*r+1:<4} {rate:>23.1%} {keys:>14}", flush=True)
    print(f"  (changed-cell observations: {nch}; {time.time()-t0:.0f}s)", flush=True)

print("\nRADIUS SWEEP DONE")
