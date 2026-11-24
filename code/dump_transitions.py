"""Dump real transitions in a form a strong model can actually read and reason over.

The point is to measure the CEILING of program synthesis on these games. The
local code model (Qwen3-Coder-Next) manages 0.384 on cd82 and 0.000 on three
other games, but that could mean either "these dynamics are not expressible as a
short program" or "this model is not good enough". Those two have opposite
implications, and only a much stronger writer can tell them apart.

Prints only the cells that changed, plus a small window of context, because the
full 64x64 hex dump is mostly unchanging background and buries the signal.
"""
import os, sys, random, json
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ARC-AGI-3-Agents"))
import numpy as np
from arc_agi import Arcade
from arcengine import GameAction, GameState

GAME = os.environ.get("DT_GAME", "cd82-fb555c5d")
N = int(os.environ.get("DT_N", "12"))
MIN_CELLS = int(os.environ.get("DT_MIN", "1"))
WIN = int(os.environ.get("DT_WIN", "9"))
SEED = int(os.environ.get("DT_SEED", "7"))

ACT = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "INTERACT", 6: "CLICK", 7: "UNDO"}


def grid(f):
    a = np.array(f.frame, dtype=np.int64)
    return a[-1] if a.ndim == 3 else a


rng = random.Random(SEED)
env = Arcade().make(GAME)
f = env.step(GameAction.RESET)
g0 = grid(f)
print(f"GAME {GAME}   available_actions={list(getattr(f,'available_actions',[]) or [])}")
print(f"initial frame: shape={g0.shape}  colour histogram={dict(Counter(g0.ravel().tolist()).most_common())}")
print("\n=== INITIAL FRAME (hex, full) ===")
for r in range(64):
    print(f"{r:2d} " + "".join(f"{v:x}" for v in g0[r]))

shown = 0
prev = g0
steps = 0
while shown < N and steps < 4000:
    steps += 1
    if getattr(f, "state", None) == GameState.GAME_OVER:
        f = env.step(GameAction.RESET); prev = grid(f); continue
    avail = [int(x) for x in (getattr(f, "available_actions", None) or [1, 2, 3, 4, 5, 6])]
    avail = [x for x in avail if 1 <= x <= 6] or [1, 2, 3, 4, 5, 6]
    cid = rng.choice(avail)
    a = GameAction.from_id(cid)
    d = {}
    if cid == 6:
        d = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}
        a.set_data(d)
    f = env.step(a, data=d)
    cur = grid(f)
    if cur.shape != prev.shape:
        prev = cur; continue
    diff = (prev != cur)
    n = int(diff.sum())
    if n < MIN_CELLS:
        prev = cur; continue

    shown += 1
    ys, xs = np.nonzero(diff)
    act = ACT[cid] + (f" x={d['x']} y={d['y']}" if cid == 6 else "")
    print(f"\n=== TRANSITION {shown}: action={act}  |  {n} cells changed ===")
    print(f"    bbox rows {ys.min()}-{ys.max()}  cols {xs.min()}-{xs.max()}")
    if n <= 40:
        print("    per-cell (row,col): before -> after")
        for y, x in zip(ys.tolist(), xs.tolist()):
            print(f"      ({y:2d},{x:2d}): {prev[y,x]:x} -> {cur[y,x]:x}")
    else:
        y0, y1 = max(0, ys.min() - 2), min(64, ys.max() + 3)
        x0, x1 = max(0, xs.min() - 2), min(64, xs.max() + 3)
        print(f"    too many to list; window rows {y0}:{y1} cols {x0}:{x1}")
        print("    BEFORE:")
        for r in range(y0, y1):
            print(f"    {r:2d} " + "".join(f"{v:x}" for v in prev[r, x0:x1]))
        print("    AFTER:")
        for r in range(y0, y1):
            print(f"    {r:2d} " + "".join(f"{v:x}" for v in cur[r, x0:x1]))
    prev = cur

print("\nDUMP DONE")
