"""Step functions written by a language model.

Per game:
  1. collect transitions by random play
  2. show the model PS_SHOW of them as text grids with the action taken
  3. ask for a Python function  step(grid, action) -> grid
  4. run it on PS_TEST held-out transitions and score exact frames and cell accuracy
  5. report the best of PS_K samples, and how many ran without an error

The reference is the identity program (`return grid`), which already gets about 98% of the cells right
because most cells never change.

Env: PS_MODEL=Qwen/Qwen3-Coder-Next, PS_GAMES=cd82-fb555c5d, PS_SHOW=6, PS_TEST=60, PS_K=4, PS_MAXNEW=4096,
     PS_CROP=24, PS_EVENT_MIN=3
"""
import os, sys, re, json, random, time, traceback

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ARC-AGI-3-Agents"))
import numpy as np
import torch
from arc_agi import Arcade
from arcengine import GameAction, GameState

MODEL = os.environ.get("PS_MODEL", "Qwen/Qwen3-Coder-Next")
GAMES = [g for g in os.environ.get("PS_GAMES", "cd82-fb555c5d").split(",") if g]
NSHOW = int(os.environ.get("PS_SHOW", "6"))      # transitions shown in the prompt
NTEST = int(os.environ.get("PS_TEST", "60"))     # held-out transitions to score on
KSAMP = int(os.environ.get("PS_K", "4"))         # programs sampled per game
# Functions with docstrings often run past 1400 tokens; a lower limit cuts them off in mid-string.
MAXNEW = int(os.environ.get("PS_MAXNEW", "4096"))
CROP = int(os.environ.get("PS_CROP", "24"))
EVENT_MIN = int(os.environ.get("PS_EVENT_MIN", "3"))      # crop size around changes, for prompt size


def grid_of(f):
    a = np.array(f.frame, dtype=np.int64)
    return a[-1] if a.ndim == 3 else a


def rollout(gid, steps, seed):
    rng = random.Random(seed)
    env = Arcade().make(gid)
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
        g0 = grid_of(f)
        f = env.step(a, data=d)
        g1 = grid_of(f)
        if not np.array_equal(g0, g1):        # only informative transitions
            out.append((g0, cid, d, g1))
    return out


def as_text(g, crop=None):
    """Compact text rendering. Full 64x64 blows up the context, so optionally
    crop to a window; the model is told the offset so coordinates stay honest."""
    if crop is None:
        sub, oy, ox = g, 0, 0
    else:
        sub, oy, ox = crop
    rows = ["".join(f"{v:x}" for v in row) for row in sub]
    return f"(origin row={oy} col={ox}, {sub.shape[0]}x{sub.shape[1]})\n" + "\n".join(rows)


def crop_around_change(g0, g1, size):
    ys, xs = np.nonzero(g0 != g1)
    if len(ys) == 0:
        return (g0[:size, :size], 0, 0)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = max(0, min(64 - size, cy - size // 2))
    x0 = max(0, min(64 - size, cx - size // 2))
    return (g0[y0:y0 + size, x0:x0 + size], y0, x0)


ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "INTERACT", 6: "CLICK"}


def build_prompt(samples):
    parts = [
        "You are reverse-engineering the rules of a grid puzzle game.",
        "The screen is a 64x64 grid of colour indices written as single hex digits (0-f).",
        "Below are real observed transitions: the grid before an action, the action, and the grid after.",
        "Each grid is cropped around the region that changed; the origin is given.",
        "",
    ]
    for i, (g0, cid, d, g1) in enumerate(samples):
        c = crop_around_change(g0, g1, CROP)
        c1 = (g1[c[1]:c[1] + c[0].shape[0], c[2]:c[2] + c[0].shape[1]], c[1], c[2])
        act = ACTION_NAMES[cid] + (f" at x={d['x']} y={d['y']}" if cid == 6 else "")
        parts += [f"--- transition {i+1}: action = {act}", "BEFORE:", as_text(None, c),
                  "AFTER:", as_text(None, c1), ""]
    parts += [
        "Write a single Python function that reproduces this game's dynamics:",
        "",
        "    def step(grid, action, x=None, y=None):",
        "        # grid: numpy int array, shape (64,64), values 0-15",
        "        # action: one of 'UP','DOWN','LEFT','RIGHT','INTERACT','CLICK'",
        "        # x,y: cell coordinates, only for CLICK",
        "        # return: the next 64x64 numpy array",
        "",
        "Rules: use only numpy and the standard library. Do not print anything.",
        "Infer the general rule, do not hard-code these specific grids.",
        "Reply with ONLY the code inside a ```python block.",
    ]
    return "\n".join(parts)


def extract_code(text):
    """Pull runnable code out of the reply.

    The first version just fell back to the whole reply when there was no
    ```python fence, so prose got compiled and 3 of 4 samples died with
    "invalid syntax (line 1)" - which looked like the model failing when it was
    actually the harness failing. Try progressively looser strategies and, as a
    last resort, start at the first `def step`."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:python)?\s*(.*)", text, re.S)      # unterminated fence
    if m and "def step" in m.group(1):
        return m.group(1).strip()
    i = text.find("def step")
    if i >= 0:
        return text[i:].strip()
    return text.strip()


def score_program(code, tests):
    """Run the candidate on held-out transitions. Returns cell/frame accuracy on
    the cells that actually change, plus whether it executed at all."""
    ns = {"np": np, "numpy": np}
    try:
        exec(compile(code, "<prog>", "exec"), ns)
        fn = ns.get("step")
        if fn is None:
            return dict(ok=False, err="no step()", ran=0, cell=0.0, frame=0.0, cell_game=0.0, frame_game=0.0)
    except Exception as e:
        return dict(ok=False, err=f"compile: {e}", ran=0, cell=0.0, frame=0.0, cell_game=0.0, frame_game=0.0)

    hit = tot = frames = 0
    hit_g = tot_g = frames_g = 0
    ran = 0
    for g0, cid, d, g1 in tests:
        try:
            pred = fn(g0.copy(), ACTION_NAMES[cid],
                      **({"x": d["x"], "y": d["y"]} if cid == 6 else {}))
            pred = np.asarray(pred)
            if pred.shape != g1.shape:
                continue
        except Exception:
            continue
        ran += 1
        ch = (g0 != g1)
        hit += int((pred[ch] == g1[ch]).sum()); tot += int(ch.sum())
        frames += int(np.array_equal(pred, g1))

        # Scores on real events. A transition that changes only a few cells is a counter tick, and the
        # ticking cells are not in a fixed place from game to game, so events are filtered by their size.
        if int(ch.sum()) > EVENT_MIN:
            hit_g += int((pred[ch] == g1[ch]).sum()); tot_g += int(ch.sum())
            frames_g += int(np.array_equal(pred, g1))
    return dict(ok=True, err="", ran=ran,
                cell=hit / max(tot, 1), frame=frames / max(len(tests), 1),
                cell_game=hit_g / max(tot_g, 1), frame_game=frames_g / max(len(tests), 1))


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"loading {MODEL} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    for gid in GAMES:
        print(f"\n{'='*70}\n{gid}  |  {MODEL}\n{'='*70}", flush=True)
        trans = rollout(gid, 1500, 3)
        if len(trans) < NSHOW + NTEST:
            print(f"only {len(trans)} changing transitions, need {NSHOW+NTEST}; skipping")
            continue
        random.Random(0).shuffle(trans)
        shown, tests = trans[:NSHOW], trans[NSHOW:NSHOW + NTEST]

        ident = score_program("def step(grid, action, x=None, y=None):\n    return grid", tests)
        print(f"identity baseline: changed-cell acc={ident['cell']:.4f} "
              f"exact-frame={ident['frame']:.4f}   (anything at or below this learned nothing)",
              flush=True)

        prompt = build_prompt(shown)
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok([text], return_tensors="pt").to(model.device)
        print(f"prompt tokens: {enc.input_ids.shape[1]}", flush=True)

        best = None
        for k in range(KSAMP):
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=True,
                                     temperature=0.7, top_p=0.9)
            gen = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
            code = extract_code(gen)
            r = score_program(code, tests)
            print(f"  sample {k}: ok={r['ok']} ran={r.get('ran',0)}/{len(tests)} "
                  f"cell={r['cell']:.4f} frame={r['frame']:.4f} | GAME-AREA cell={r['cell_game']:.4f} "
                  f"frame={r['frame_game']:.4f} {r['err'][:50]}", flush=True)
            if best is None or (r["ok"] and r["cell_game"] > best[0]["cell_game"]):
                best = (r, code)

        r, code = best
        print(f"\nBEST game-area: changed-cell={r['cell_game']:.4f} exact-frame={r['frame_game']:.4f}  "
              f"(incl HUD: cell={r['cell']:.4f} frame={r['frame']:.4f})", flush=True)
        os.makedirs("progsyn", exist_ok=True)
        with open(f"progsyn/{gid}.json", "w") as f:
            json.dump({"game": gid, "model": MODEL, "best": r, "identity": ident,
                       "code": code}, f, indent=2)

    print("\nPROGSYN DONE")


if __name__ == "__main__":
    main()
