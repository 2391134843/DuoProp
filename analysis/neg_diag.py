"""Negation diagnostic: Duo-only atoms vs QTO's own atoms under negation scales s and two test protocols
(standard: only training edges are known; observed: the validation graph is known at test time, as in BetaE's
definition of easy test answers).  Validation is always evaluated on the standard protocol (its own observed graph).
Diagnostic only: 1000 validation / 3000 test queries per type (fixed subsets); s only affects negation types.
Usage: python analysis/neg_diag.py <data> <name> <match> <spec> [<spec> ...]   spec = label=cache:rb:irr (irr 0/1)"""
import json
import sys

import torch

from duoprop.collective_eval import irreflexive, observe
from duoprop.reason import NEG, ORDER, Reasoner, evaluate, load_cache, load_split

if __name__ == '__main__':
    data, name, match = sys.argv[1:4]
    specs = [s.split('=') for s in sys.argv[4:]]
    n = int(open(data + '/stats.txt').read().split('\n')[0].split(' ')[-1])
    m = torch.load(match, weights_only=False)
    splits = {s: load_split(data, s) for s in ('valid', 'test')}
    out = open(f'runs/results/{name}.jsonl', 'a')
    for label, spec in specs:
        cache, rb, irr = spec.split(':')
        for protocol in ('standard', 'observed'):
            adj = load_cache(cache, n, 'cuda', None, m if float(rb) > 0 else None, float(rb))
            if irr == '1':
                irreflexive(adj, data)
            if protocol == 'observed':
                observe(adj, data, 'valid')
            avg_p = {}  # positive queries do not depend on s: evaluated once per split, negation types for every s
            for s in (1.0, 3.0, 6.0):
                for split in (('valid', 'test') if protocol == 'standard' else ('test',)):
                    q, e, h = splits[split]
                    res = evaluate(Reasoner(adj, n, 'cuda', neg_scale=s), q, e, h, types=ORDER if s == 1.0 else NEG,
                                   limit=1000 if split == 'valid' else 3000)
                    avg_p.setdefault(split, res.get('AVG_p'))
                    rec = dict(label=label, cache=cache, rb=float(rb), irr=irr == '1', protocol=protocol, neg_scale=s,
                               split=split, AVG_p=avg_p[split], AVG_n=res['AVG_n'],
                               neg={t: res[t]['MRR'] for t in ('2in', '3in', 'inp', 'pin', 'pni')})
                    out.write(json.dumps(rec) + '\n')
                    out.flush()
                    print(f"{label:12s} {protocol:8s} s={s:.0f} {split:5s} AVG_p={avg_p[split]:.2f} AVG_n={res['AVG_n']:.2f} "
                          + ' '.join(f'{t}:{v:.1f}' for t, v in rec['neg'].items()), flush=True)
            del adj
            torch.cuda.empty_cache()
    print('PART DONE', flush=True)
