"""Test a fixed DuoProp setting (no selection) under the standard and the consistent protocol, e.g. the setting selected
for one training seed applied to other seeds.  Output: runs/paper/fixed_<name>.json
python tools/fixed_eval.py NAME DATA CACHE MATCH RB NEG BETA AGG IRR KNN_TAG"""
import json
import sys

import torch

from duoprop.collective_eval import CollReasoner, get_knn, irreflexive, observe
from duoprop.reason import NEG, ORDER, evaluate, load_cache, load_split
from duoprop.scorer import read_stats

name, data, cache, match, rb, neg, beta, agg, irr, ktag = sys.argv[1:11]
rb, neg, beta = float(rb), float(neg), float(beta)
n, _ = read_stats(data)
q, e, h = load_split(data, 'test')
m = torch.load(match, weights_only=False) if rb > 0 else None
adj = load_cache(cache, n, 'cuda', None, m, rb)
if irr == '1':
    irreflexive(adj, data)
nbr, val = get_knn(data, 'rel+emb', 20, ktag)
kw = dict(mode='max') if agg == 'max' else dict(mode='pnorm', p=float(agg))
R = lambda s: CollReasoner(adj, n, 'cuda', neg_scale=s, nbr=nbr, val=val, beta=beta, **kw)
std = evaluate(R(neg), q, e, h)
out = dict(name=name, cache=cache, rb=rb, neg=neg, beta=beta, agg=agg, irr=irr, standard=std)
print('standard', round(std['AVG_p'], 2), round(std['AVG_n'], 2), flush=True)
observe(adj, data, 'valid')
pos = evaluate(R(1.0), q, e, h, types=ORDER[:9])
out['consistent'] = dict(AVG_p=pos['AVG_p'])
for s in (1.0, 3.0, 6.0):
    out['consistent'][f'AVG_n_s{s:g}'] = evaluate(R(s), q, e, h, types=NEG)['AVG_n']
print('consistent', json.dumps(out['consistent']), flush=True)
json.dump(out, open(f'runs/paper/fixed_{name}.json', 'w'), indent=1)
print('PART DONE', flush=True)
