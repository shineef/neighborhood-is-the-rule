"""Paired evaluation of the agent with the pretrained network against the random and the guided agent.

Scoring follows sg/scoring.py and the games run through env.py. All arms play the same seeds. One game per
invocation, so games can run in parallel.

Env:
  EW_GAMES=cd82,vc33,...   (default: all 25)
  EW_SEEDS=40              seeds per arm
  EW_ARMS=random,guided,wm_pretrained   (wm_scratch, wm_frozen also available)
  EW_OUT=results/eval.jsonl
"""
import os, sys, json, random, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "ARC-AGI-3-Agents"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sg"))
sys.path.insert(0, HERE)

import numpy as np
from scoring import game_score
from env import ARCEnvironment, load_dataset
from random_agent import RandomAgent
from guided_agent import GuidedAgent

GAMES = [g for g in os.environ.get("EW_GAMES", "").split(",") if g]
SEEDS = int(os.environ.get("EW_SEEDS", "40"))
SEED0 = int(os.environ.get("EW_SEED0", "1000"))
OUT = os.environ.get("EW_OUT", os.path.join(HERE, "results", "eval.jsonl"))
WANT = [a for a in os.environ.get("EW_ARMS", "random,guided,wm_pretrained").split(",") if a]
os.makedirs(os.path.dirname(OUT), exist_ok=True)

ds = load_dataset(os.path.join(ROOT, "environment_files"))
if not GAMES:
    GAMES = sorted(ds)

# The arms with a network use one class with different flags. The module reads the flags at import, so
# each arm imports it inside a factory that sets them first.
def make_wm_class(pretrained, finetune, mode="argmax"):
    os.environ["WMP_PRETRAINED"] = "1" if pretrained else "0"
    os.environ["WMP_FINETUNE"] = "1" if finetune else "0"
    os.environ["WMP_MODE"] = mode
    import importlib
    import wm_pretrained_agent as m
    importlib.reload(m)
    return m.WMPretrainedAgent


def get_arm(name):
    if name == "random":
        return RandomAgent
    if name == "guided":
        return GuidedAgent
    if name == "wm_pretrained":
        return make_wm_class(True, True, "argmax")
    if name == "wm_filter":
        return make_wm_class(True, True, "filter")
    if name == "wm_frozen":
        return make_wm_class(True, False, "argmax")
    if name == "wm_scratch":
        return make_wm_class(False, True, "argmax")
    raise ValueError(name)


def play(gid, AgentCls, seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    env = ARCEnvironment(ds[gid], level_id_str=gid)
    obs = env.reset()
    agent = AgentCls(card_id="ew", game_id=gid, agent_name="ew",
                     ROOT_URL="", record=False, arc_env=env.arc_env)
    agent.frames = [obs]
    agent.action_counter = 0
    done = False
    info = {}
    while not done:
        a = agent.choose_action(agent.frames, obs)
        obs, r, done, info = env.step(a)
        agent.action_counter += 1
        agent.append_frame(obs)
        if agent.is_done(agent.frames, obs):
            break
    recs = info.get("level_records", [])
    total = info.get("total_levels", len(recs)) or 1
    done_lv = sum(1 for r in recs if r.get("completed"))
    used = getattr(agent, "used_model_steps", None)
    return game_score(recs), done_lv, total, used


with open(OUT, "a") as fh:
    for gid in GAMES:
        res = {"game": gid, "seeds": SEEDS}
        for name in WANT:
            cls = get_arm(name)
            t0 = time.time()
            scores, levels, used = [], [], []
            for s in range(SEEDS):
                sc, lv, tot, um = play(gid, cls, SEED0 + s)
                scores.append(sc); levels.append(lv)
                if um is not None:
                    used.append(um)
            res[name] = {
                "score_mean": float(np.mean(scores)),
                "score_se": float(np.std(scores, ddof=1) / len(scores) ** 0.5) if len(scores) > 1 else 0.0,
                "levels_mean": float(np.mean(levels)),
                "levels_max": int(max(levels)),
                "win_rate": float(np.mean([l > 0 for l in levels])),
                "total_levels": tot,
                "sec": round(time.time() - t0, 1),
                "scores": [round(float(s), 4) for s in scores],  # per-seed, for paired tests
            }
            ms = ""
            if used:
                res[name]["model_steps_mean"] = float(np.mean(used))
                ms = f" mdl={res[name]['model_steps_mean']:.0f}"
            print(f"{gid:<18} {name:<14} score={res[name]['score_mean']:.4f}"
                  f" +/-{res[name]['score_se']:.4f} win%={100*res[name]['win_rate']:.0f}"
                  f" lv={res[name]['levels_mean']:.2f}{ms} ({res[name]['sec']}s)", flush=True)
        fh.write(json.dumps(res) + "\n"); fh.flush()

print("EVAL_WM DONE", flush=True)
