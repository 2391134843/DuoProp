"""Test evaluation under the observed-graph protocol (validation edges known at test time and set to truth 1, as in CKGC
and in the consistent protocol of the paper's negation section).  No selection: DuoProp is the validation-selected setting
of runs/results/<sel>_coll_test.jsonl (last record); optionally QTO's forward matrix of the same link predictor.
Positive types once, negation types for the selected s and for s in {1, 3, 6}.  Output: runs/paper/obs_<name>.json
python tools/obs_eval.py NAME DATA SEL_NAME KNN_TAG [QTO_CACHE QTO_EPS]"""
import json
import sys
import time

import torch

from duoprop.collective_eval import CollReasoner, get_knn, irreflexive, observe
from duoprop.reason import NEG, ORDER, Reasoner, evaluate, load_cache, load_split
from duoprop.scorer import read_stats

name, data, sel, ktag = sys.argv[1:5]
qcache, qeps = (sys.argv[5], sys.argv[6]) if len(sys.argv) > 6 else (None, None)
n, _ = read_stats(data)
dev = 'cuda'
q, easy, hard = load_split(data, 'test')
rec = [json.loads(l) for l in open(f'runs/results/{sel}_coll_test.jsonl')][-1]
POS = [t for t in ORDER if t not in NEG]
out = dict(name=name, sel=sel, setting={k: rec[k] for k in ('cache', 'eps', 'row_beta', 'neg_scale', 'kind', 'k', 'beta', 'agg',
                                                          'irreflexive')})


def run(make, tag):
    t0 = time.time()
    res = evaluate(make(1.0), q, easy, hard, types=POS)
    r = dict(pos={t: res[t]['MRR'] for t in POS}, AVG_p=res['AVG_p'])
    for s in sorted({1.0, 3.0, 6.0, float(rec['neg_scale'])}):
        rn = evaluate(make(s), q, easy, hard, types=NEG)
        r[f'neg_s{s:g}'] = {t: rn[t]['MRR'] for t in NEG}
        r[f'AVG_n_s{s:g}'] = rn['AVG_n']
    print(tag, json.dumps(r), f'{time.time() - t0:.0f}s', flush=True)
    return r


m = torch.load(f'cache/art/match_{ktag}.pt', weights_only=False) if rec['row_beta'] > 0 else None
adj = load_cache(rec['cache'], n, dev, None if rec['eps'] == 'none' else float(rec['eps']), m, rec['row_beta'])
if rec.get('irreflexive'):
    irreflexive(adj, data)
observe(adj, data, 'valid')
nbr, val = get_knn(data, rec['kind'], rec['k'], ktag)
kw = dict(mode='max') if rec['agg'] == 'max' else dict(mode='pnorm', p=float(rec['agg']))
out['duoprop'] = run(lambda s: CollReasoner(adj, n, dev, neg_scale=s, nbr=nbr, val=val, beta=rec['beta'],
                                            update=rec.get('update', 'add'), steps=rec.get('steps', 1), **kw), 'DuoProp')
del adj
torch.cuda.empty_cache()
if qcache:
    adq = load_cache(qcache, n, dev, None if qeps == 'none' else float(qeps))
    observe(adq, data, 'valid')
    out['qto'] = run(lambda s: Reasoner(adq, n, dev, s), 'QTO')
json.dump(out, open(f'runs/paper/obs_{name}.json', 'w'), indent=1)
print('PART DONE', flush=True)
