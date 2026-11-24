"""Is the dynamics of these games learnable from few observations?

Every candidate line in the plan - meta-RL, program synthesis, model-based
planning - rests on the same unverified premise: that you can predict the next
frame after seeing a modest number of transitions. This measures that directly,
before we commit weeks to any of them.

Two parts:

  A. Determinism check. Replay an identical action sequence twice from reset and
     compare frames step by step. The static scan said only 2 of 25 games import
     `random`, but that is a proxy; this tests the actual behaviour.

  B. Local-rule learnability. Model the dynamics as a per-cell rule:
        (3x3 neighbourhood of colours at t, action) -> colour of centre at t+1
     Fit it by pure memorisation from N transitions, then score held-out
     transitions. Memorisation is deliberately the dumbest possible learner - if
     it works, a real model will do better; if the learning curve is flat, the
     dynamics are not local and that tells us something important.

  Reported against two baselines:
     - "static": predict the frame is unchanged (these games mostly do nothing)
     - "coverage": fraction of held-out cells whose (neighbourhood, action) key
       was ever seen in training. Accuracy on unseen keys is guesswork, so
       coverage is what actually limits few-shot prediction.

Usage: EXP_GAMES=sp80-589a99af,cd82-fb555c5d python exp_learnability.py
"""
import os, sys, json, random, time
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ARC-AGI-3-Agents"))
import numpy as np
from env import load_dataset
from arc_agi import Arcade
from arcengine import GameAction, GameState

GAMES = [g for g in os.environ.get("EXP_GAMES", "sp80-589a99af,cd82-fb555c5d").split(",") if g]
STEPS = int(os.environ.get("EXP_STEPS", "3000"))
SEED = int(os.environ.get("EXP_SEED", "0"))


def grid(frame):
    a = np.array(frame.frame, dtype=np.int16)
    if a.ndim == 3:
        a = a[-1]
    return a


def rollout(gid, steps, seed, record=True):
    """Random play, resetting on death. Returns [(grid_t, action_key, grid_t1)]."""
    rng = random.Random(seed)
    arc = Arcade()
    env = arc.make(gid)
    f = env.step(GameAction.RESET)
    out = []
    acts = []
    for _ in range(steps):
        if getattr(f, "state", None) == GameState.GAME_OVER:
            f = env.step(GameAction.RESET)
            acts.append(("RESET", None))
            continue
        cid = rng.choice([1, 2, 3, 4, 5, 6])
        a = GameAction.from_id(cid)
        data = {}
        if cid == 6:
            data = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}
            a.set_data(data)
        g0 = grid(f)
        f = env.step(a, data=data)
        acts.append((cid, tuple(sorted(data.items())) if data else None))
        if record:
            out.append((g0, cid, grid(f)))
    return out, acts


def replay(gid, acts):
    """Re-execute a recorded action sequence and return the frame hashes."""
    arc = Arcade()
    env = arc.make(gid)
    f = env.step(GameAction.RESET)
    hs = []
    for cid, data in acts:
        if cid == "RESET":
            f = env.step(GameAction.RESET)
        else:
            a = GameAction.from_id(cid)
            d = dict(data) if data else {}
            if d:
                a.set_data(d)
            f = env.step(a, data=d)
        hs.append(hash(grid(f).tobytes()))
    return hs


def neighbourhood_keys(g, cid):
    """3x3 patch (edge-padded) + action id, for every cell."""
    p = np.pad(g, 1, mode="constant", constant_values=-1)
    H, W = g.shape
    keys = np.empty((H, W), dtype=object)
    for y in range(H):
        for x in range(W):
            keys[y, x] = (p[y:y + 3, x:x + 3].tobytes(), cid)
    return keys


def evaluate(transitions, n_train):
    table = {}
    conflicts = 0
    for g0, cid, g1 in transitions[:n_train]:
        keys = neighbourhood_keys(g0, cid)
        for y in range(g0.shape[0]):
            for x in range(g0.shape[1]):
                k = keys[y, x]
                v = int(g1[y, x])
                if k in table and table[k] != v:
                    conflicts += 1
                    table[k] = v
                else:
                    table.setdefault(k, v)

    hit = tot = covered = static_ok = 0
    for g0, cid, g1 in transitions[n_train:]:
        keys = neighbourhood_keys(g0, cid)
        for y in range(g0.shape[0]):
            for x in range(g0.shape[1]):
                tot += 1
                truth = int(g1[y, x])
                if int(g0[y, x]) == truth:
                    static_ok += 1
                k = keys[y, x]
                if k in table:
                    covered += 1
                    if table[k] == truth:
                        hit += 1
    return dict(acc=hit / tot, coverage=covered / tot,
                static=static_ok / tot, conflicts=conflicts, table=len(table))


for gid in GAMES:
    print(f"\n{'='*66}\n{gid}\n{'='*66}", flush=True)

    # --- A. determinism -------------------------------------------------
    t0 = time.time()
    trans, acts = rollout(gid, min(STEPS, 400), SEED)
    h1 = replay(gid, acts)
    h2 = replay(gid, acts)
    same = sum(1 for a, b in zip(h1, h2) if a == b)
    print(f"A. determinism: {same}/{len(h1)} steps identical across two replays "
          f"-> {'DETERMINISTIC' if same == len(h1) else 'NON-DETERMINISTIC'}", flush=True)

    # --- B. learnability ------------------------------------------------
    trans, _ = rollout(gid, STEPS, SEED + 1)
    changed = sum(1 for g0, _, g1 in trans if not np.array_equal(g0, g1))
    print(f"B. collected {len(trans)} transitions, {changed} of them changed the "
          f"screen ({100*changed/max(len(trans),1):.1f}%)", flush=True)

    print(f"   {'train':>6} {'cell acc':>9} {'coverage':>9} {'static':>8} {'conflicts':>10} {'table':>9}")
    for n in [50, 100, 250, 500, 1000, 2000]:
        if n >= len(trans):
            break
        r = evaluate(trans, n)
        print(f"   {n:>6} {r['acc']:>9.4f} {r['coverage']:>9.4f} {r['static']:>8.4f} "
              f"{r['conflicts']:>10} {r['table']:>9}", flush=True)
    print(f"   ({time.time()-t0:.0f}s)", flush=True)

print("\nEXPERIMENT DONE")
