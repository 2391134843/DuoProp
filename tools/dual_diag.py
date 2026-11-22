"""Atom-level diagnosis of Section 4 for any link predictor (generalises duoprop/figdata.py dual(), which is fixed to the
published SpherE checkpoint on FB15k-237).  Top-k unobserved head-side candidates of rows with a hidden answer and of random
rows; head side at T=1 (QTO) and at the selected T, tail side, Duo, Duo x matchability.  Output: runs/paper/dualdiag_<name>.json
python tools/dual_diag.py NAME DATA CKPT T ALPHA MATCH [RB]"""
import collections
import json
import random
import sys

import torch
from scipy.stats import rankdata

from duoprop.scorer import make_scorer, read_stats

name, DATA, CKPT, T, ALPHA, MATCH = sys.argv[1:7]
T, ALPHA = float(T), float(ALPHA)
RB = float(sys.argv[7]) if len(sys.argv) > 7 else 2.0
n, nrel = read_stats(DATA)


def triples(split):
    out = collections.defaultdict(set)
    for line in open(f'{DATA}/{split}.txt'):
        h, r, t = map(int, line.split())
        out[(h, r)].add(t)
    return out


def auc(pos, neg):
    ranks = rankdata(list(pos) + list(neg))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


sc = make_scorer(CKPT, 'cuda')
train = triples('train')
num = torch.ones(nrel, n, device='cuda')
for (h, r), ts in train.items():
    num[r, h] = len(ts)


def rows(r, hs, temps):
    """calibration of build.py: min(1 - 1e-4, n_h * softmax(s / T)), training edges = 1; one block per temperature"""
    hs = torch.as_tensor(hs, device='cuda')
    S, out = sc.scores(hs, r), []
    for temp in temps:
        P = torch.softmax(S / temp, dim=1) * num[r, hs].unsqueeze(1)
        P.clamp_(max=1 - 1e-4)
        for i, h in enumerate(hs.tolist()):
            ts = train.get((h, r))
            if ts:
                P[i, list(ts)] = 1.0
        out.append(P)
    return out


hidden = collections.defaultdict(set)
for split in ('valid', 'test'):
    for key, ts in triples(split).items():
        hidden[key] |= ts - train.get(key, set())
M = torch.load(MATCH, weights_only=False).float()
rng = random.Random(0)
rws = rng.sample(sorted(x for x, v in hidden.items() if v), 1500) + [(rng.randrange(n), rng.randrange(nrel)) for _ in range(1500)]
by_rel = collections.defaultdict(list)
for h, r in dict.fromkeys(rws):
    by_rel[r].append(h)
C = []
for r, hs in sorted(by_rel.items()):
    Pf1, Pf = rows(r, hs, (1.0, T))
    for i, h in enumerate(hs):
        if (h, r) in train:
            Pf1[i, list(train[(h, r)])] = -1.0
            Pf[i, list(train[(h, r)])] = -1.0
    val1, idx = Pf1.topk(20, dim=1)  # candidates: top-20 unobserved tails of QTO's head side
    val = Pf.gather(1, idx)
    need = collections.defaultdict(list)
    for i, h in enumerate(hs):
        for t in idx[i].tolist():
            need[t].append(h)
    U, pb, pb1 = sorted(need), {}, {}
    for s0 in range(0, len(U), 512):
        Pb1, Pb = rows(r ^ 1, U[s0:s0 + 512], (1.0, T))
        for j, t in enumerate(U[s0:s0 + 512]):
            hh = torch.tensor(need[t], device='cuda')
            pb.update(zip(((h, t) for h in need[t]), Pb[j, hh].tolist()))
            pb1.update(zip(((h, t) for h in need[t]), Pb1[j, hh].tolist()))
    for i, h in enumerate(hs):
        obs, m = (h, r) in train, M[r, h].item()
        for v, v1, t in zip(val[i].tolist(), val1[i].tolist(), idx[i].tolist()):
            d = v ** ALPHA * pb[(h, t)] ** (1 - ALPHA)
            C.append(dict(label=int(t in hidden.get((h, r), ())), obs=int(obs), self=int(t == h), head_T1=v1, tail_T1=pb1[(h, t)],
                          dual_T1=v1 ** ALPHA * pb1[(h, t)] ** (1 - ALPHA), head=v, tail=pb[(h, t)], dual=d,
                          final=d if obs else d * m ** RB))
S = dict(name=name, data=DATA, ckpt=CKPT, T=T, alpha=ALPHA, rb=RB, n=len(C), n_hidden=sum(c['label'] for c in C))
for sub_name, sub in [('all', C), ('rows_without_training_tail', [c for c in C if not c['obs']])]:
    p, q = [c for c in sub if c['label']], [c for c in sub if not c['label']]
    S[sub_name] = {f'AUC_{s}': round(auc([c[s] for c in p], [c[s] for c in q]), 4) for s in ('head_T1', 'dual_T1', 'head', 'dual', 'final')}
    S[sub_name].update(n_pos=len(p), n_neg=len(q))
conf = [c for c in C if c['head_T1'] >= 0.5]
cp, cn = [c for c in conf if c['label']], [c for c in conf if not c['label']]
S['head_T1>=0.5'] = {'n': len(conf), 'frac_hidden_fact': round(len(cp) / max(len(conf), 1), 4),
                      'nonfacts_tail_T1<0.01': round(sum(c['tail_T1'] < 0.01 for c in cn) / max(len(cn), 1), 4),
                      'facts_tail_T1<0.01': round(sum(c['tail_T1'] < 0.01 for c in cp) / max(len(cp), 1), 4),
                      'nonfacts_self_loops': round(sum(c['self'] for c in cn) / max(len(cn), 1), 4)}
json.dump(S, open(f'runs/paper/dualdiag_{name}.json', 'w'), indent=1)
print(json.dumps(S), flush=True)
print('PART DONE', flush=True)
