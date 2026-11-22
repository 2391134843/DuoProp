"""QTO (forward matrix, no Duo/Prop) on one link predictor under the standard and the consistent protocol, the QTO
counterpart of fixed_eval.py (e.g. for extra training seeds).  Output: runs/paper/qtofixed_<name>.json
python tools/qto_fixed.py NAME DATA FORWARD_CACHE NEG"""
import json
import sys

from duoprop.collective_eval import observe
from duoprop.reason import NEG, ORDER, Reasoner, evaluate, load_cache, load_split
from duoprop.scorer import read_stats

name, data, cache, neg = sys.argv[1:5]
neg = float(neg)
n, _ = read_stats(data)
q, e, h = load_split(data, 'test')
adj = load_cache(cache, n, 'cuda')
R = lambda s: Reasoner(adj, n, 'cuda', s)
std = evaluate(R(neg), q, e, h)
out = dict(name=name, cache=cache, neg=neg, standard=std)
print('standard', round(std['AVG_p'], 2), round(std['AVG_n'], 2), flush=True)
observe(adj, data, 'valid')
pos = evaluate(R(1.0), q, e, h, types=ORDER[:9])
out['consistent'] = dict(AVG_p=pos['AVG_p'])
for s in (1.0, 3.0, 6.0):
    out['consistent'][f'AVG_n_s{s:g}'] = evaluate(R(s), q, e, h, types=NEG)['AVG_n']
print('consistent', json.dumps(out['consistent']), flush=True)
json.dump(out, open(f'runs/paper/qtofixed_{name}.json', 'w'), indent=1)
print('PART DONE', flush=True)
