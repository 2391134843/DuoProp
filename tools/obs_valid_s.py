"""Validation AVG_n of the selected DuoProp setting for negation scalings s in {1, 3, 6} (validation subsets of 1,000
queries per type, as in the selection).  On validation queries all known edges are training edges and already have truth 1,
so validation matches the observed-graph test protocol and can select s for it.  Output: runs/paper/obsval_<name>.json
python tools/obs_valid_s.py NAME DATA SEL_NAME KNN_TAG"""
import json
import sys

import torch

from duoprop.collective_eval import CollReasoner, get_knn, irreflexive
from duoprop.reason import NEG, evaluate, load_cache, load_split
from duoprop.scorer import read_stats

name, data, sel, ktag = sys.argv[1:5]
n, _ = read_stats(data)
q, easy, hard = load_split(data, 'valid')
rec = [json.loads(l) for l in open(f'runs/results/{sel}_coll_test.jsonl')][-1]
m = torch.load(f'cache/art/match_{ktag}.pt', weights_only=False) if rec['row_beta'] > 0 else None
adj = load_cache(rec['cache'], n, 'cuda', None if rec['eps'] == 'none' else float(rec['eps']), m, rec['row_beta'])
if rec.get('irreflexive'):
    irreflexive(adj, data)
nbr, val = get_knn(data, rec['kind'], rec['k'], ktag)
kw = dict(mode='max') if rec['agg'] == 'max' else dict(mode='pnorm', p=float(rec['agg']))
out = dict(name=name, sel=sel)
for s in (1.0, 3.0, 6.0):
    r = evaluate(CollReasoner(adj, n, 'cuda', neg_scale=s, nbr=nbr, val=val, beta=rec['beta'], update=rec.get('update', 'add'),
                              steps=rec.get('steps', 1), **kw), q, easy, hard, types=NEG, limit=1000)
    out[f'valid_AVG_n_s{s:g}'] = r['AVG_n']
    print(name, s, r['AVG_n'], flush=True)
json.dump(out, open(f'runs/paper/obsval_{name}.json', 'w'), indent=1)
print('PART DONE', flush=True)
