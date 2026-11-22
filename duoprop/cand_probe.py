"""Validation probes for Prop candidates on top of Duo (training-free, plug-in):
  P1  pseudo-relevance feedback: prototype of the query's top-k answers (embedding space) re-ranks candidates
  P2  structural smoothing (Correct&Smooth-style): one diffusion step of the answer vector over an entity
      kNN graph built from shared (relation, neighbour) profiles of the training graph
  P3  sharpening of intermediate (variable) fuzzy sets before the next projection
  P4  union by max (Goedel) instead of probabilistic sum
Usage: python -m duoprop.cand_probe <data> <name> <cache> <match> <rb> <ckpt> [limit]"""
import json
import os
import sys

import torch

from duoprop.reason import Reasoner, evaluate, load_cache, load_split
from duoprop.scorer import make_scorer


def profile_knn(data, n, k=50, block=2048):
    """Top-k structural neighbours: cosine similarity of IDF-weighted neighbour sets in the training graph
    (entities sharing specific neighbours, e.g. the same actor or genre, are similar; hubs are down-weighted)."""
    A = torch.zeros(n, n, device='cuda')
    for line in open(os.path.join(data, 'train.txt')):
        h, r, t = map(int, line.split())
        A[h, t] = 1.0
    A = ((A + A.t()) > 0).float()
    deg = A.sum(0)
    idf = torch.log((n + 1) / (deg + 1))
    Aw = A * idf
    Aw = (Aw / Aw.norm(dim=1, keepdim=True).clamp(min=1e-12)).half()
    nbr = torch.empty(n, k, dtype=torch.long, device='cuda')
    val = torch.empty(n, k, device='cuda')
    for s0 in range(0, n, block):
        G = (Aw[s0:s0 + block] @ Aw.t()).float()
        G[torch.arange(G.shape[0]), torch.arange(s0, s0 + G.shape[0])] = 0
        v, i = G.topk(k, dim=1)
        nbr[s0:s0 + len(v)], val[s0:s0 + len(v)] = i, v
    del A, Aw
    torch.cuda.empty_cache()
    return nbr, val


class ProbeReasoner(Reasoner):
    def __init__(self, *a, sharpen=1.0, union='prob', **kw):
        super().__init__(*a, **kw)
        self.sharpen, self.union = sharpen, union

    def path(self, T, rels):
        nz = (T > 0).sum()
        if self.sharpen != 1.0 and nz > 1:  # variable (non-anchor) input set
            T = T.clamp(min=0) ** self.sharpen
        return super().path(T, rels)

    def execute(self, q, s):
        if self.union == 'max' and s[-1] == ('u',):
            return torch.stack([self.branch(q[i], s[i], 'union') for i in range(len(s) - 1)]).max(0)[0]
        return super().execute(q, s)


def wrap(R, post):
    base = R.execute

    def ex(q, s, top=[True]):
        if not top[0]:
            return base(q, s)
        top[0] = False
        try:
            return post(base(q, s))
        finally:
            top[0] = True
    R.execute = ex
    return R


if __name__ == '__main__':
    data, name, cache, match, rb, ckpt = sys.argv[1:7]
    limit = int(sys.argv[7]) if len(sys.argv) > 7 else 1000
    n = int(open(os.path.join(data, 'stats.txt')).read().split('\n')[0].split(' ')[-1])
    adj = load_cache(cache, n, 'cuda', None, torch.load(match, weights_only=False), float(rb))
    q, e, h = load_split(data, 'valid')
    X = make_scorer(ckpt, 'cuda').features().float()
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = X / X.norm(dim=1, keepdim=True)
    nbr, nval = profile_knn(data, n)
    out = f'runs/results/{name}_cand_valid.jsonl'

    def run(tag, R):
        res = evaluate(R, q, e, h, limit=limit)
        with open(out, 'a') as f:
            f.write(json.dumps(dict(cand=tag, res=res)) + '\n')
        print(f"{tag:28s} AVG_p={res['AVG_p']:.2f} AVG_n={res['AVG_n']:.2f} | " +
              ' '.join(f"{t}:{res[t]['MRR']:.1f}" for t in res if t not in ('AVG_p', 'AVG_n')), flush=True)

    run('base', Reasoner(adj, n, 'cuda'))
    for k in [5, 20]:
        for beta in [0.01, 0.05, 0.2]:
            def prf(T, k=k, beta=beta):
                w, i = T.topk(k)
                c = (w.unsqueeze(1) * X[i]).sum(0)
                c = c / c.norm().clamp(min=1e-12)
                return T + beta * (X @ c).clamp(min=0)
            run(f'P1 prf k={k} beta={beta}', wrap(Reasoner(adj, n, 'cuda'), prf))
    for beta in [0.05, 0.2, 0.5]:
        def smooth(T, beta=beta):
            nb = (nval * T[nbr]).sum(1) / nval.sum(1).clamp(min=1e-12)
            return T + beta * nb
        run(f'P2 smooth beta={beta}', wrap(Reasoner(adj, n, 'cuda'), smooth))
    for g in [0.5, 2.0]:
        run(f'P3 sharpen gamma={g}', ProbeReasoner(adj, n, 'cuda', sharpen=g))
    run('P4 union=max', ProbeReasoner(adj, n, 'cuda', union='max'))
