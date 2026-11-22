"""Validation probe: how the existential variable is aggregated in a projection.
QTO / max-product keeps only the single best path to each answer:  y_a = max_v T_v A_va  (Viterbi / MAP).
Marginal-style alternatives count every supporting path:
  nor    noisy-or (probabilistic sum, the t-conorm the reasoner already uses for union):
         y_a = 1 - prod_v (1 - T_v A_va);   mixed with max: max + lam * (nor - max)
  pnorm  y_a = min(1, (sum_v (T_v A_va)^p)^(1/p)),  p -> inf recovers max
Only projections whose input is a variable set change (anchored projections have one source).
Usage: python -m duoprop.aggregation <data> <name> <cache> <match> <rb> [limit]"""
import json
import os
import sys

import torch

from duoprop.reason import Reasoner, evaluate, load_cache, load_split


class AggReasoner(Reasoner):
    def __init__(self, *a, mode='max', lam=1.0, p=2.0, **kw):
        super().__init__(*a, **kw)
        self.mode, self.lam, self.p = mode, lam, p

    def project(self, T, r, mats=None):
        mats = self.cur if mats is None else mats
        rows = torch.nonzero(T > 0, as_tuple=True)[0]
        out = torch.zeros(self.n, device=self.dev)
        if len(rows) == 0:
            return out
        pos, col, val = mats[r].gather(rows)
        x = T[rows][pos] * val
        out.scatter_reduce_(0, col, x, reduce='amax', include_self=True)
        if self.mode == 'max' or len(rows) == 1:
            return out
        if self.mode == 'nor':
            s = torch.zeros(self.n, device=self.dev).scatter_add_(0, col, torch.log1p(-x.clamp(max=1 - 1e-6)))
            return out + self.lam * ((1 - torch.exp(s)) - out)
        s = torch.zeros(self.n, device=self.dev).scatter_add_(0, col, x.pow(self.p))
        return s.pow(1 / self.p).clamp(max=1.0)


if __name__ == '__main__':
    data, name, cache, match, rb = sys.argv[1:6]
    limit = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
    n = int(open(os.path.join(data, 'stats.txt')).read().split('\n')[0].split(' ')[-1])
    adj = load_cache(cache, n, 'cuda', None, torch.load(match, weights_only=False), float(rb))
    q, e, h = load_split(data, 'valid')
    out = f'runs/results/{name}_agg_valid.jsonl'

    def run(tag, R):
        res = evaluate(R, q, e, h, limit=limit)
        with open(out, 'a') as f:
            f.write(json.dumps(dict(cand=tag, res=res)) + '\n')
        print(f"{tag:28s} AVG_p={res['AVG_p']:.2f} AVG_n={res['AVG_n']:.2f} | " +
              ' '.join(f"{t}:{res[t]['MRR']:.1f}" for t in res if t not in ('AVG_p', 'AVG_n')), flush=True)

    run('max (base)', AggReasoner(adj, n, 'cuda'))
    for lam in [0.25, 0.5, 1.0]:
        run(f'nor lam={lam}', AggReasoner(adj, n, 'cuda', mode='nor', lam=lam))
    for p in [2.0, 4.0]:
        run(f'pnorm p={p}', AggReasoner(adj, n, 'cuda', mode='pnorm', p=p))
