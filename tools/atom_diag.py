"""Head-side / tail-side readings of single atoms under a DuoProp setting (failure diagnosis for the case study).
python tools/atom_diag.py DATA CKPT T ALPHA EPS MATCH h:r:t [h:r:t ...]"""
import collections
import os
import sys

import torch

from duoprop.scorer import make_scorer, read_stats

data, ckpt, T, alpha, eps, match = sys.argv[1:7]
T, alpha, eps = float(T), float(alpha), float(eps)
n, nrel = read_stats(data)
sc = make_scorer(ckpt, 'cuda')
tails = collections.defaultdict(set)
for l in open(os.path.join(data, 'train.txt')):
    h, r, t = map(int, l.split())
    tails[(h, r)].add(t)
m = torch.load(match, weights_only=False).float()
refl = {int(l.split()[1]) for l in open(os.path.join(data, 'train.txt')) if l.split()[0] == l.split()[2]}


def P(h, r, t):
    S = sc.scores(torch.tensor([h], device='cuda'), r)[0]
    p = torch.softmax(S / T, 0) * max(1, len(tails[(h, r)]))
    p = p.clamp(max=1 - 1e-4)
    rank = int((p > p[t]).sum()) + 1
    return (1.0 if t in tails[(h, r)] else float(p[t])), rank, int(p.argmax())


for spec in sys.argv[7:]:
    h, r, t = map(int, spec.split(':'))
    ri = r ^ 1
    ph, rk_h, top_h = P(h, r, t)
    pt, rk_t, top_t = P(t, ri, h)
    a = ph ** alpha * pt ** (1 - alpha)
    obs = len(tails[(h, r)]) > 0
    print(dict(h=h, r=r, t=t, head=round(ph, 5), head_rank=rk_h, head_top=top_h, tail=round(pt, 6), tail_rank=rk_t, tail_top=top_t,
               mean=round(a, 6), below_eps=a < eps, row_observed=obs, m=round(float(m[r, h]), 4), irreflexive=r not in refl))
