"""Summaries for Sec. 5 and for the click planes. Run from the repository root: python code/sum_sec5.py
  1. scheduled sampling against plain training   results/sched_s*.jsonl      (exp_sched.py)
  2. networks without the second click plane     results/clickab_s*.jsonl    (exp_rollout_ft.py, RO_CLICK_T=0)
     against the networks with both planes       results/rollout_ft*.jsonl
  3. the compounding law, game by game           results/hybrid_s*.jsonl     (exp_hybrid.py)
Set RES to read another folder; SCHED_GLOB and CLICK_GLOB to read other files for 1 and 2.
"""
import json, glob, os, statistics as S

RES = os.environ.get("RES", "results")


def load(p):
    return [json.loads(l) for f in sorted(glob.glob(f"{RES}/{p}")) for l in open(f) if l.strip()]


def m(v):
    v = list(v)
    return S.mean(v) if v else float("nan")


# ------------------------------------------------------------------ 1. scheduled sampling
R = load(os.environ.get("SCHED_GLOB", "sched_s*.jsonl"))
if R:
    arms = [a for a in ("tf", "ss_next", "ss_same") if any(r["arm"] == a for r in R)]
    by = {a: {(r["game"], r["seed"]): r for r in R if r["arm"] == a} for a in arms}
    D = sorted({int(d) for r in R for d in r["rollout"]})
    print(f"== 1. SCHEDULED SAMPLING (rate {R[0]['rate']}, view {R[0]['rf']}, {R[0]['steps']} moves, {R[0]['epochs']} epochs)")
    for a in arms:
        print(f"   {a:<8} runs {len(by[a]):>2}  games {len({k[0] for k in by[a]})}  seeds {sorted({k[1] for k in by[a]})}")
    for other in arms[1:]:
        both = sorted(set(by["tf"]) & set(by[other]))
        if not both:
            continue
        print(f" -- {other} against tf, {len(both)} paired runs (same game, seed, initial weights, batches)")
        print("    depth    tf     " + f"{other:<8}" + " diff    better/worse/equal   diff by seed")
        rows = [("test", lambda r: r["test_exact"])] + [(f"d{d}", (lambda d: lambda r: r["rollout"][str(d)]["exact"])(d)) for d in D]
        for name, f in rows:
            a0 = [f(by["tf"][k]) for k in both]; a1 = [f(by[other][k]) for k in both]
            df = [y - x for x, y in zip(a0, a1)]
            seeds = sorted({k[1] for k in both})
            bs = " ".join(f"{m(f(by[other][k]) - f(by['tf'][k]) for k in both if k[1] == s):+.3f}" for s in seeds)
            print(f"    {name:<6} {m(a0):.3f}   {m(a1):.3f}   {m(df):+.3f}   "
                  f"{sum(x > 1e-9 for x in df):>2}/{sum(x < -1e-9 for x in df):>2}/{sum(abs(x) <= 1e-9 for x in df):>2}"
                  f"            {bs}")
        print(f"    static d1 {m(by['tf'][k]['rollout']['1']['static'] for k in both):.3f}; "
              f"two-step samples per run {m(by[other][k]['two_step_samples'] for k in both):.0f}; "
              f"chainable moves {m(by[other][k]['chainable'] for k in both):.3f}")
        print("    by game (mean over seeds), d1 d2 d3 d5:")
        for g in sorted({k[0] for k in both}):
            ks = [k for k in both if k[0] == g]
            cell = lambda arm, d: m(by[arm][k]["rollout"][str(d)]["exact"] for k in ks)
            print(f"      {g.split('-')[0]}  tf " + " ".join(f"{cell('tf', d):.2f}" for d in D[:4])
                  + f"   {other} " + " ".join(f"{cell(other, d):.2f}" for d in D[:4]))

# ------------------------------------------------------------------ 2. click plane
A = load(os.environ.get("CLICK_GLOB", "clickab_s*.jsonl"))
B = [r for r in load("rollout_ft*.jsonl") if r.get("rf", 9) == 9 and r.get("click_t", True)]
if A:
    print("\n== 2. CLICK PLANES: both planes (paper) against the correct plane only")
    D = ["1", "2", "3", "5", "10"]
    for arm in ("scratch", "pretrain"):
        a = {(r["game"], r["seed"]): r for r in A if r["arm"] == arm}
        b = {(r["game"], r["seed"]): r for r in B if r["arm"] == arm}
        both = sorted(set(a) & set(b))
        if not both:
            print(f"   {arm}: no paired runs yet (ablation {len(a)}, paper {len(b)})"); continue
        seeds = sorted({k[1] for k in both})
        full = [s for s in seeds if sum(k[1] == s for k in both) == 25]
        print(f" -- {arm}: {len(both)} paired runs, complete seeds {full}")
        for name, f in (("exact", lambda r: r["one_step"]["exact"]), ("events", lambda r: r["one_step"]["event_exact"])):
            pb = [m(f(b[k]) for k in both if k[1] == s) for s in seeds]
            pa = [m(f(a[k]) for k in both if k[1] == s) for s in seeds]
            print(f"    {name:<7} both planes {m(pb):.4f} ({' '.join(f'{x:.4f}' for x in pb)})   one plane {m(pa):.4f} ({' '.join(f'{x:.4f}' for x in pa)})   diff {m(pa) - m(pb):+.4f}")
        ok = [k for k in both if a[k]["rollout"] and b[k]["rollout"]]
        print("    rollouts d=1 2 3 5 10: both " + " ".join(f"{m(b[k]['rollout'][d]['exact'] for k in ok):.3f}" for d in D)
              + "   one " + " ".join(f"{m(a[k]['rollout'][d]['exact'] for k in ok):.3f}" for d in D))
        gd = []
        for g in sorted({k[0] for k in both}):
            ks = [k for k in both if k[0] == g]
            gd.append((m(a[k]["one_step"]["exact"] for k in ks) - m(b[k]["one_step"]["exact"] for k in ks), g.split("-")[0],
                       m(b[k]["one_step"]["exact"] for k in ks), m(a[k]["one_step"]["exact"] for k in ks)))
        print(f"    games where one plane is better/worse/equal: {sum(x[0] > 1e-9 for x in gd)}/{sum(x[0] < -1e-9 for x in gd)}/{sum(abs(x[0]) <= 1e-9 for x in gd)}")
        print("    largest per-game changes (one plane - both): " + "  ".join(f"{g} {d:+.3f} ({x:.3f}->{y:.3f})" for d, g, x, y in sorted(gd, key=lambda t: -abs(t[0]))[:8]))

# ------------------------------------------------------------------ 3. compounding law, game by game
H = load("hybrid_s*.jsonl")
if H:
    print("\n== 3. COMPOUNDING LAW game by game: measured exact after k steps | mean over runs of (exact after 1 step)^k")
    names = [n for n in ("lookup_k7", "lookup_k13", "lookup_k25", "lookup_k41", "lookup_backoff", "cnn", "hybrid_k13", "hybrid_backoff") if n in H[0]["pred"]]
    D = ["2", "3", "5", "10"]
    print("    predictor        d1      " + "   ".join(f"d{d}: meas  law " for d in D) + "  (pooled d1)^2")
    for n in names:
        rs = [r["pred"][n]["rollout"] for r in H if "rollout" in r["pred"][n]]
        d1 = m(x["1"]["exact"] for x in rs)
        print(f"    {n:<15} {d1:.3f}   " + "   ".join(f"{m(x[d]['exact'] for x in rs):.3f} {m(x['1']['exact'] ** int(d) for x in rs):.3f}  " for d in D)
              + f"   {d1 ** 2:.3f}")
