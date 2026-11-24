"""Compact summary of the network, table and rollout results. Run from the repository root:
    python code/summarize.py > results/summary.txt
Reads results/meta_s*.log, results/lookup.jsonl and results/rollout_ft_s*.jsonl (set RES to change the folder).
"""
import glob, json, re, os, statistics as st
import numpy as np

R = os.environ.get("RES", "results")
out = []
P = out.append


def mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else float("nan")


# ---------------------------------------------------------------- meta logs
meta = {}  # seed -> game -> regime -> dict
for f in sorted(glob.glob(f"{R}/meta_s*.log")):
    s = int(re.search(r"meta_s(\d+)", f).group(1))
    for line in open(f):
        m = re.match(r"^(\S+-\S+)\s+(scratch|pretrain|reptile)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", line)
        if m:
            g = m.group(1).split("-")[0]
            meta.setdefault(s, {}).setdefault(g, {})[m.group(2)] = dict(
                exact=float(m.group(3)), static=float(m.group(4)), ev=float(m.group(5)), n_ev=int(m.group(6)))

P("== META (exp_meta.py; RF9, 300 local, 600 test) ==")
P("seed ngames  scratch  pretrain  reptile | rep-pre  SE    t    W/L | pre-scr  | ev: scr pre rep")
diffs, prescr = [], []
for s in sorted(meta):
    gs = [g for g in meta[s] if all(r in meta[s][g] for r in ("scratch", "pretrain", "reptile"))]
    if not gs:
        continue
    sc = [meta[s][g]["scratch"]["exact"] for g in gs]; pr = [meta[s][g]["pretrain"]["exact"] for g in gs]
    rp = [meta[s][g]["reptile"]["exact"] for g in gs]
    d = np.array(rp) - np.array(pr); se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    ev = [mean([meta[s][g][r]["ev"] for g in gs]) for r in ("scratch", "pretrain", "reptile")]
    P(f"{s:>4} {len(gs):>6}  {mean(sc):.4f}   {mean(pr):.4f}   {mean(rp):.4f} | {d.mean():+.4f} {se:.4f} {d.mean()/se if se else 0:+.2f} {int((d>0).sum())}/{int((d<0).sum())} | {mean(pr)-mean(sc):+.4f} | {ev[0]:.3f} {ev[1]:.3f} {ev[2]:.3f}")
    if len(gs) == 25:
        diffs.append(d.mean()); prescr.append(mean(pr) - mean(sc))
if len(diffs) >= 2:
    a = np.array(diffs); b = np.array(prescr)
    P(f"complete seeds={len(a)}  reptile-pretrain mean {a.mean():+.4f} SD {a.std(ddof=1):.4f} t {a.mean()/(a.std(ddof=1)/np.sqrt(len(a))):+.2f} pos {int((a>0).sum())}/{len(a)}")
    P(f"                 pretrain-scratch mean {b.mean():+.4f} SD {b.std(ddof=1):.4f} t {b.mean()/(b.std(ddof=1)/np.sqrt(len(b))):+.2f}")

# per-game seed means
pg = {}
for s in meta:
    for g, r in meta[s].items():
        for reg, v in r.items():
            pg.setdefault(g, {}).setdefault(reg, []).append(v["exact"])
        pg[g]["static"] = [r[next(iter(r))]["static"]]
        pg[g]["n_ev"] = [r[next(iter(r))]["n_ev"]]

# ---------------------------------------------------------------- lookup
LK = [json.loads(l) for l in open(f"{R}/lookup.jsonl")] if os.path.exists(f"{R}/lookup.jsonl") else []
if LK:
    P("\n== LOOKUP standard (300 train / 600 test) ==")
    P("  k   exact  static  ev_exact  prec   recall  cov_all  cov_chg  #exact>static")
    for k in sorted({r["k"] for r in LK}):
        rs = [r for r in LK if r["protocol"] == "standard" and r["k"] == k]
        P(f"{k:>3}  {mean([r['exact'] for r in rs]):.4f} {mean([r['static'] for r in rs]):.4f}  {mean([r['event_exact'] for r in rs]):.4f}   "
          f"{mean([r['precision'] for r in rs]):.3f}  {mean([r['recall'] for r in rs]):.3f}   {mean([r['coverage'] for r in rs]):.4f}   {mean([r['coverage_changed'] for r in rs]):.3f}    "
          f"{sum(r['exact'] > r['static'] for r in rs)}/{len(rs)}")
    P("best-k per game (oracle over k): mean exact %.4f" % mean([max(r["exact"] for r in LK if r["protocol"] == "standard" and r["game"] == g)
                                                              for g in {r["game"] for r in LK}]))
    P("\n== LOOKUP curve (seed 31; last 600 test) mean exact / mean cov_chg ==")
    ks = sorted({r["k"] for r in LK}); ns = sorted({r["n"] for r in LK if r["protocol"] == "curve"})
    P("   n  " + "  ".join(f"k={k:<2} ex  cov" for k in ks))
    for n in ns:
        row = []
        for k in ks:
            rs = [r for r in LK if r["protocol"] == "curve" and r["n"] == n and r["k"] == k]
            row.append(f"{mean([r['exact'] for r in rs]):.3f} {mean([r['coverage_changed'] for r in rs]):.3f}")
        P(f"{n:>5}  " + "   ".join(row))
    P("\n== LOOKUP rollouts (standard split) mean exact / static ==")
    for k in ks:
        rs = [r for r in LK if r["protocol"] == "standard" and r["k"] == k and r.get("rollout")]
        ds = sorted({int(d) for r in rs for d in r["rollout"]})
        P(f"k={k:<3}" + "  ".join(f"d{d}: {mean([r['rollout'][str(d)]['exact'] for r in rs if str(d) in r['rollout']]):.3f}/"
                                  f"{mean([r['rollout'][str(d)]['static'] for r in rs if str(d) in r['rollout']]):.3f}" for d in ds))

# ---------------------------------------------------------------- rollout_ft
RO = []
for f in glob.glob(f"{R}/rollout_ft*.jsonl"):
    RO += [json.loads(l) for l in open(f)]
if RO:
    P("\n== ROLLOUT after finetune (closed loop, chains seed 99) ==")
    for arm in ("scratch", "pretrain"):
        rs = [r for r in RO if r["arm"] == arm and r["rollout"]]
        if not rs:
            continue
        ds = sorted({int(d) for r in rs for d in r["rollout"]})
        P(f"{arm:<9} n={len(rs)} 1step {mean([r['one_step']['exact'] for r in rs]):.3f}  " +
          "  ".join(f"d{d}: {mean([r['rollout'][str(d)]['exact'] for r in rs]):.3f}/{mean([r['rollout'][str(d)]['static'] for r in rs]):.3f}" for d in ds))
        # compounding: predicted exact(1)^k vs measured, averaged over runs
        P("   compound check  " + "  ".join(
            f"d{d}: meas {mean([r['rollout'][str(d)]['exact'] for r in rs]):.3f} pred {mean([r['rollout']['1']['exact']**d for r in rs]):.3f}" for d in ds))
        P(f"   games beating static at d5: {sum(r['rollout'].get('5',{}).get('exact',0) > r['rollout'].get('5',{}).get('static',1) for r in rs)}/{len(rs)}")

# ---------------------------------------------------------------- per-game table
if pg:
    P("\n== PER GAME (seed means) ==")
    P("game  n_ev static scratch pretrain reptile  | lookup k7  k13  k41 | covchg k7")
    for g in sorted(pg):
        lk = {r["k"]: r for r in LK if r["protocol"] == "standard" and r["game"].split("-")[0] == g}
        P(f"{g:<5} {pg[g]['n_ev'][0]:>4} {pg[g]['static'][0]:.3f}  {mean(pg[g].get('scratch',[])):.3f}   {mean(pg[g].get('pretrain',[])):.3f}   {mean(pg[g].get('reptile',[])):.3f}  | "
          + " ".join(f"{lk[k]['exact']:.3f}" if k in lk else "  -  " for k in (7, 13, 41))
          + f" | {lk[7]['coverage_changed'] if 7 in lk else float('nan'):.3f}")

print("\n".join(out))
