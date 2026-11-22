"""Summarise a validation sweep (<tag>_valid.jsonl): top configurations and per-factor bests."""
import collections
import json
import os
import re
import sys

ORDER = ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up', '2in', '3in', 'inp', 'pin', 'pni']

if __name__ == '__main__':
    recs = [json.loads(l) for l in open(sys.argv[1])]
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    for r in recs:
        r['name'] = os.path.basename(r['cache']).split('_', 2)[-1][:-3]
    recs.sort(key=lambda r: -r['score'])
    print(f'{len(recs)} records')
    print(f'{"config":28s} {"rb":>3s} {"eps":>5s} {"s":>3s} {"AVG_p":>6s} {"AVG_n":>6s} ' + ' '.join(f'{q:>5s}' for q in ORDER))
    for r in recs[:topk]:
        x = r['res']
        print(f'{r["name"]:28s} {r.get("row_beta", 0):3.0f} {r["eps"]:>5s} {r["neg_scale"]:3.0f} {x["AVG_p"]:6.2f} {x["AVG_n"]:6.2f} '
              + ' '.join(f'{x[q]["MRR"]:5.1f}' for q in ORDER))
    print('-- best AVG_p / best AVG_n per config family')
    fam = collections.defaultdict(lambda: [0, 0, ''])
    for r in recs:
        m = re.match(r'(dual|forward)_T([\d.]+)(?:_a([\d.]+)_b([\d.]+))?', r['name'])
        key = r['name']
        f = fam[key]
        f[0] = max(f[0], r['res']['AVG_p'])
        f[1] = max(f[1], r['res']['AVG_n'])
    for k in sorted(fam, key=lambda k: -(fam[k][0] + fam[k][1]))[:topk]:
        print(f'{k:28s} best AVG_p={fam[k][0]:.2f} best AVG_n={fam[k][1]:.2f}')
