"""Efficiency of DuoProp vs QTO at inference (TPAMI asks for it): wall time per query type and peak GPU memory on the
test split, same GPU, same evaluator; run with the GPU to itself (queue weight = capacity).
Usage: python analysis/bench.py <data> <out.json> <label>=<cache>:<rb>:<irr>:<beta>:<agg>:<neg>:<knn tag>:<match> [...]"""
import json
import sys
import time

import torch

from duoprop.collective_eval import CollReasoner, get_knn, irreflexive
from duoprop.reason import STRUCT, evaluate, load_cache, load_split

if __name__ == '__main__':
    data, out = sys.argv[1], sys.argv[2]
    n = int(open(data + '/stats.txt').read().split('\n')[0].split(' ')[-1])
    q, e, h = load_split(data, 'test')
    res = {}
    for spec in sys.argv[3:]:
        label, rest = spec.split('=', 1)
        cache, rb, irr, beta, agg, neg, knn_tag, match = rest.split(':')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        m = torch.load(match, weights_only=False) if float(rb) > 0 else None
        adj = load_cache(cache, n, 'cuda', None, m, float(rb))
        if irr == '1':
            irreflexive(adj, data)
        nbr, val = get_knn(data, 'rel+emb', 20, knn_tag)
        torch.cuda.synchronize()
        load_s = time.time() - t0
        kw = dict(mode='max') if agg == 'max' else dict(mode='pnorm', p=float(agg))
        R = CollReasoner(adj, n, 'cuda', neg_scale=float(neg), nbr=nbr, val=val, beta=float(beta), **kw)
        per_type = {}
        for s, name in STRUCT.items():
            if s not in q:
                continue
            torch.cuda.synchronize()
            t = time.time()
            r = evaluate(R, q, e, h, types=[name])
            torch.cuda.synchronize()
            per_type[name] = dict(seconds=time.time() - t, n=r[name]['n'], MRR=r[name]['MRR'],
                                  ms_per_query=1000 * (time.time() - t) / r[name]['n'])
        res[label] = dict(cache=cache, beta=float(beta), agg=agg, load_seconds=load_s,
                          total_seconds=sum(v['seconds'] for v in per_type.values()),
                          peak_gpu_GB=torch.cuda.max_memory_allocated() / 1e9,
                          nnz=sum(int(a.val.numel()) for a in adj), per_type=per_type)
        print(label, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res[label].items() if k != 'per_type'},
              flush=True)
        del adj, R
    json.dump(res, open(out, 'w'), indent=1)
    print('PART DONE', flush=True)
