"""Summary of the table / network / hybrid comparison and of the networks with wider views.
Run from the repository root: python code/sum_rfx_hybrid.py
Reads results/hybrid_s*.jsonl, results/rfx_rf*_s*.jsonl and results/rollout_ft_s*.jsonl.
"""
import json, glob, statistics as S
D = ['1', '2', '3', '5', '10']
def load(p): return [json.loads(l) for f in sorted(glob.glob(p)) for l in open(f) if l.strip()]
def ev(x): return 0.0 if x is None else x
def sd(v): return S.stdev(v) if len(v) > 1 else 0.0
H = load('results/hybrid_s*.jsonl')
names = list(H[0]['pred']); seeds = sorted(set(r['seed'] for r in H)); G = sorted(set(r['game'] for r in H))
print('HYBRID records', len(H), 'seeds', seeds, 'games', len(G))
print('one-step: exact(mean+-sd over seeds) event-exact precision recall')
for n in names:
    ex, ee, pr, rc = [], [], [], []
    for s in seeds:
        R = [r['pred'][n]['one_step'] for r in H if r['seed'] == s]
        ex.append(S.mean(o['exact'] for o in R)); ee.append(S.mean(ev(o['event_exact']) for o in R))
        pr.append(S.mean(o['precision'] for o in R)); rc.append(S.mean(o['recall'] for o in R))
    print(f"{n:15s} {S.mean(ex):.3f}+-{sd(ex):.3f} ev {S.mean(ee):.3f} pr {S.mean(pr):.3f} rc {S.mean(rc):.3f}")
print('rollouts exact at d=1 2 3 5 10 (same windows)')
for n in names + ['static']:
    key, src = ('static', names[0]) if n == 'static' else ('exact', n)
    row = []
    for d in D:
        v = [S.mean(r['pred'][src]['rollout'][d][key] for r in H if r['seed'] == s and 'rollout' in r['pred'][src]) for s in seeds]
        row.append(S.mean(v))
    print(f"{n:15s} " + " ".join(f"{x:.3f}" for x in row))
def gm(n, g): return S.mean(r['pred'][n]['one_step']['exact'] for r in H if r['game'] == g)
for a, b in [('hybrid_k13', 'lookup_k13'), ('hybrid_k13', 'cnn'), ('hybrid_backoff', 'lookup_backoff'), ('lookup_backoff', 'lookup_k13'), ('hybrid_backoff', 'hybrid_k13')]:
    w = sum(gm(a, g) > gm(b, g) + 1e-9 for g in G); l = sum(gm(a, g) < gm(b, g) - 1e-9 for g in G)
    print(f"{a} vs {b}: win {w} lose {l} tie {len(G) - w - l}")
X = load('results/rfx_rf*_s*.jsonl') + load('results/rollout_ft_s*.jsonl')
print('RF sweep: one-step exact(mean+-sd) event | rollout d=1 2 3 5 10')
for rf in sorted(set(r['rf'] for r in X)):
    for arm in ('scratch', 'pretrain'):
        ex, ee, ro = [], [], {d: [] for d in D}
        for s in sorted(set(r['seed'] for r in X)):
            R = [r for r in X if r['rf'] == rf and r['arm'] == arm and r['seed'] == s]
            if not R: continue
            ex.append(S.mean(r['one_step']['exact'] for r in R)); ee.append(S.mean(r['one_step']['event_exact'] for r in R))
            for d in D: ro[d].append(S.mean(r['rollout'][d]['exact'] for r in R if r['rollout']))
        print(f"RF{rf:<2} {arm:8s} n={len(ex)} {S.mean(ex):.3f}+-{sd(ex):.3f} ev {S.mean(ee):.3f} | " + " ".join(f"{S.mean(ro[d]):.3f}" for d in D))
