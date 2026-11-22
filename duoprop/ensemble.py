"""Atom-matrix ensembles: weighted mean of the Duo atom matrices of several evaluated runs, each at its own
validation-selected configuration (positives-only DuoProp pick of MS_<run>_finalP: cache, matchability scaling,
irreflexive prior), saved as a cache (computed on the CPU, relation by relation) so DuoProp can reason over it.
Runs can be seeds of one backbone (seed ensemble) or different backbones (cross-backbone fusion).
Usage: python -m duoprop.ensemble <data> <out.pt> <run>[@weight] [<run>[@weight] ...]
       python -m duoprop.ensemble <data> <out.pt> --caches <cache>:<match>:<rb> [...]  (same configuration, equal weights)
       python -m duoprop.ensemble <data> <family> <seeds e.g. 0,1,2> <out.pt>      (seed ensemble, equal weights)"""
import json
import sys

import torch

from duoprop.collective_eval import irreflexive
from duoprop.reason import load_cache

if __name__ == '__main__':
    data = sys.argv[1]
    specs = None  # explicit caches: <cache>:<match>:<rb> (equal weights, irreflexive prior applied)
    if len(sys.argv) > 3 and sys.argv[3] == '--caches':
        out = sys.argv[2]
        specs = [s.split(':') for s in sys.argv[4:]]
        runs = [(c, 1.0) for c, _, _ in specs]
    elif len(sys.argv) == 5 and sys.argv[4].endswith('.pt') and sys.argv[3].replace(',', '').isdigit():  # old form
        fam, out = sys.argv[2], sys.argv[4]
        runs = [(f'{fam}_s{k}', 1.0) for k in sys.argv[3].split(',')]
    else:
        out = sys.argv[2]
        runs = [(s.split('@')[0], float(s.split('@')[1]) if '@' in s else 1.0) for s in sys.argv[3:]]
    total = sum(w for _, w in runs)
    n = int(open(data + '/stats.txt').read().split('\n')[0].split(' ')[-1])
    keys, vals = None, None
    for i, (run, w) in enumerate(runs):
        if specs:
            rec = dict(cache=specs[i][0], eps='none', row_beta=float(specs[i][2]), irreflexive=True)
            match = specs[i][1]
        else:
            rec = [json.loads(l) for l in open(f'runs/results/MS_{run}_finalP_coll_test.jsonl')][-1]
            match = f'cache/art/match_{run}.pt'
        m = torch.load(match, weights_only=False) if rec['row_beta'] > 0 else None
        adj = load_cache(rec['cache'], n, 'cpu', None if rec['eps'] == 'none' else float(rec['eps']), m, rec['row_beta'])
        if rec.get('irreflexive'):
            irreflexive(adj, data)
        if keys is None:
            keys, vals = [[] for _ in adj], [[] for _ in adj]
        for r, M in enumerate(adj):
            rows = torch.repeat_interleave(torch.arange(n), M.crow[1:] - M.crow[:-1])
            keys[r].append(rows * n + M.col.long())
            vals[r].append(M.val.float() * (w / total))
        print(f'{run} (weight {w / total:.2f}): {rec["cache"]} rb={rec["row_beta"]} irr={rec.get("irreflexive")}', flush=True)
        del adj
    crow, col, val = [], [], []
    for r in range(len(keys)):
        kk, vv = torch.cat(keys[r]), torch.cat(vals[r])
        uk, inv = torch.unique(kk, return_inverse=True)
        s = torch.zeros(len(uk)).index_add_(0, inv, vv)
        c = torch.zeros(n + 1, dtype=torch.long)
        c[1:] = torch.bincount(uk // n, minlength=n).cumsum(0)
        crow.append(c), col.append((uk % n).to(torch.int32)), val.append(s)
        keys[r] = vals[r] = None
    torch.save({'crow': crow, 'col': col, 'val': val}, out)
    print('saved', out, 'nnz =', sum(v.numel() for v in val), flush=True)
