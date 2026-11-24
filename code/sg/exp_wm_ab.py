"""Where does guided exploration's 2.90x actually come from?

Aggregate result (200 seeds each): guided scores 0.00206 vs random 0.00075,
Welch t = +10.36, and completes 2.60 of 183 levels vs 0.90. That is a real
effect, but an aggregate hides the two things worth knowing:

  1. Is the lift broad, or is it one or two games carrying everything? If it is
     concentrated, the "median 8.1x click lift" story is weaker than it looks
     and we should not generalise it to the private set.
  2. Does guided crack any of the 14 games where random NEVER completed a level
     in 40000 steps? Those are the games with no positive examples at all, so
     anything that bootstraps off random data has no starting point there.
     Breaking even one of them is worth more than incremental gains elsewhere.

Runs both agents on the same games with the same seeds, so the comparison is
paired and the difference is attributable to the policy rather than to seed
luck. One game per invocation so it parallelises across workers.
"""
import os, sys, json, random, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "..", "ARC-AGI-3-Agents"))
sys.path.append(os.path.join(ROOT, ".."))
sys.path.append(ROOT)

import numpy as np
from env import ARCEnvironment, load_dataset
from random_agent import RandomAgent
from guided_agent import GuidedAgent
from wm_agent import WMAgent

GAMES = [g for g in os.environ.get("BD_GAMES", "").split(",") if g]
SEEDS = int(os.environ.get("BD_SEEDS", "40"))
OUT = os.environ.get("BD_OUT", os.path.join(ROOT, "..", "breakdown.jsonl"))

ds = load_dataset(os.path.join(ROOT, "..", "environment_files"))
if not GAMES:
    GAMES = sorted(ds)


def play(gid, AgentCls, seed):
    random.seed(seed)
    np.random.seed(seed)
    env = ARCEnvironment(ds[gid], level_id_str=gid)
    obs = env.reset()
    agent = AgentCls(card_id="bd", game_id=gid, agent_name="bd",
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
    numer = 0.0
    done_lv = 0
    for rec in recs:
        if rec["completed"] and rec["agent_steps"] > 0:
            numer += rec["level"] * min((rec["human_steps"] / rec["agent_steps"]) ** 2, 1.15)
            done_lv += 1
    denom = sum(range(1, total + 1))
    return (numer / denom if denom else 0.0), done_lv, total


with open(OUT, "a") as fh:
    for gid in GAMES:
        t0 = time.time()
        res = {"game": gid}
        for name, cls in (("guided", GuidedAgent), ("wm", WMAgent)):
            scores, levels = [], []
            for s in range(SEEDS):
                sc, lv, tot = play(gid, cls, 1000 + s)
                scores.append(sc); levels.append(lv)
            res[name] = {
                "score_mean": float(np.mean(scores)),
                "score_se": float(np.std(scores, ddof=1) / len(scores) ** 0.5) if len(scores) > 1 else 0.0,
                "levels_mean": float(np.mean(levels)),
                "levels_max": int(max(levels)),
                "win_rate": float(np.mean([l > 0 for l in levels])),
                "total_levels": tot,
            }
        fh.write(json.dumps(res) + "\n")
        fh.flush()
        r, g = res["guided"], res["wm"]
        print(f"{gid:<20} guided lv={r['levels_mean']:.2f} win%={100*r['win_rate']:>5.1f} | "
              f"guided lv={g['levels_mean']:.2f} win%={100*g['win_rate']:>5.1f} "
              f"| score {r['score_mean']:.5f} -> {g['score_mean']:.5f}  ({time.time()-t0:.0f}s)",
              flush=True)

print("BREAKDOWN DONE")
