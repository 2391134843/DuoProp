"""Stage-2 probe of neighbourhood propagation (Prop) on top of Duo.
Answer vector T is smoothed over a kNN graph of entities with similar *training-graph profiles*:
  ent   untyped neighbour set (IDF weighted)                              -- cand_probe.py's version
  rel   relation-typed neighbour set {(r, neighbour)}, hashed to 2^16 dims (IDF weighted)
  type  relation signature: which relations an entity has, in/out (IDF weighted counts)
where: 'final'  T + beta * mean_kNN(T) on the answer set only
       'every'  the same after every projection (intermediate variables included)
Usage: python -m duoprop.neighbours <data> <name> <cache> <match> <rb> <stage> [limit]"""
import json
import os
import sys

import torch

from duoprop.cand_probe import wrap
from duoprop.reason import Reasoner, evaluate, load_cache, load_split


def triples(data):
    for line in open(os.path.join(data, 'train.txt')):
        yield map(int, line.split())


def profiles(data, n, nrel, kind, dim=1 << 16):
    if kind == 'ent':
        P = torch.zeros(n, n, device='cuda')
        for h, r, t in triples(data):
            P[h, t] = 1.0
        P = ((P + P.t()) > 0).float()
    elif kind == 'rel':
        P = torch.zeros(n, dim, device='cuda', dtype=torch.half)
        hs, fs = [], []
        for h, r, t in triples(data):  # train.txt contains both directions
            hs.append(h)
            fs.append((r * 1000003 + t * 7919) % dim)
        P.index_put_((torch.tensor(hs, device='cuda'), torch.tensor(fs, device='cuda')),
                     torch.ones(len(hs), device='cuda', dtype=torch.half))
    else:
        P = torch.zeros(n, nrel, device='cuda')
        for h, r, t in triples(data):
            P[h, r] += 1.0
        P = torch.log1p(P)
    df = sum((P[i:i + 4096] > 0).float().sum(0) for i in range(0, n, 4096))  # chunked: a full bool sum is int64
    idf = torch.log((n + 1) / (df + 1)).to(P.dtype)
    P *= idf
    norm = torch.cat([P[i:i + 4096].float().norm(dim=1) for i in range(0, n, 4096)])
    P /= norm.clamp(min=1e-12).to(P.dtype).unsqueeze(1)
    return P.half()


def knn(P, k, block=2048):
    n = P.shape[0]
    nbr = torch.empty(n, k, dtype=torch.long, device='cuda')
    val = torch.empty(n, k, device='cuda')
    for s0 in range(0, n, block):
        G = (P[s0:s0 + block] @ P.t()).float()
        G[torch.arange(G.shape[0]), torch.arange(s0, s0 + G.shape[0])] = 0
        v, i = G.topk(k, dim=1)
        nbr[s0:s0 + len(v)], val[s0:s0 + len(v)] = i, v.clamp(min=0)
    return nbr, val


class SmoothReasoner(Reasoner):
    def __init__(self, *a, nbr=None, val=None, beta=0.0, **kw):
        super().__init__(*a, **kw)
        self.nbr, self.val, self.beta = nbr, val, beta

    def smooth(self, T):
        return T + self.beta * (self.val * T[self.nbr]).sum(1) / self.val.sum(1).clamp(min=1e-12)

    def project(self, T, r, mats=None):
        return self.smooth(super().project(T, r, mats))


if __name__ == '__main__':
    data, name, cache, match, rb, stage = sys.argv[1:7]
    limit = int(sys.argv[7]) if len(sys.argv) > 7 else 1000
    st = open(os.path.join(data, 'stats.txt')).read().split('\n')
    n, nrel = int(st[0].split(' ')[-1]), int(st[1].split(' ')[-1])
    adj = load_cache(cache, n, 'cuda', None, torch.load(match, weights_only=False), float(rb))
    q, e, h = load_split(data, 'valid')
    out = f'runs/results/{name}_p2_valid.jsonl'

    def run(tag, R):
        res = evaluate(R, q, e, h, limit=limit)
        with open(out, 'a') as f:
            f.write(json.dumps(dict(cand=tag, res=res)) + '\n')
        print(f"{tag:34s} AVG_p={res['AVG_p']:.2f} AVG_n={res['AVG_n']:.2f} | " +
              ' '.join(f"{t}:{res[t]['MRR']:.1f}" for t in res if t not in ('AVG_p', 'AVG_n')), flush=True)

    def final(nbr, val, beta):
        return wrap(Reasoner(adj, n, 'cuda'),
                    lambda T: T + beta * (val * T[nbr]).sum(1) / val.sum(1).clamp(min=1e-12))

    if stage == '1':
        run('base', Reasoner(adj, n, 'cuda'))
        for kind in ['ent', 'rel', 'type']:
            P = profiles(data, n, nrel, kind)
            nbr, val = knn(P, 100)
            del P
            torch.cuda.empty_cache()
            for k in [20, 50, 100]:
                run(f'{kind} k={k} beta=0.05 final', final(nbr[:, :k], val[:, :k], 0.05))
            torch.save((nbr.cpu(), val.cpu()), f'cache/art/knn_{name}_{kind}.pt')
    else:
        kind, k = stage.split(':')
        k = int(k)
        nbr, val = (x[:, :k].cuda() for x in torch.load(f'cache/art/knn_{name}_{kind}.pt'))
        for beta in [0.02, 0.1, 0.2]:
            run(f'{kind} k={k} beta={beta} final', final(nbr, val, beta))
        for beta in [0.02, 0.05, 0.1]:
            run(f'{kind} k={k} beta={beta} every', SmoothReasoner(adj, n, 'cuda', nbr=nbr, val=val, beta=beta))
