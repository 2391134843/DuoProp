"""Per-query test evaluation of two reasoners on the same link predictor (A = QTO, B = DuoProp with its validation-selected
setting), for significance tests, fine-grained analysis and case studies.  No selection happens here: B is read from the
finished selection record runs/results/<sel>_coll_test.jsonl (last record), A is QTO's forward matrix at its threshold.
For every test query one JSON line: type, query, hard answers, filtered ranks of the hard answers under A and B, the 20
best non-easy candidates under A and B, and the cosine between the truth-weighted centroid of the top-100 candidates and
the centroid of the hard answers in the link predictor's standardised feature space (a measure of where the fuzzy answer
set puts its mass).
python tools/perquery.py NAME DATA CKPT QTO_CACHE QTO_EPS QTO_S SEL_NAME KNN_TAG [BUILD_EPS]"""
import gzip
import json
import os
import subprocess
import sys
import time

import torch

from duoprop.collective_eval import CollReasoner, get_knn, irreflexive
from duoprop.reason import ORDER, STRUCT, Reasoner, filtered_ranks, load_cache, load_split
from duoprop.scorer import make_scorer, read_stats

name, data, ckpt, qcache, qeps, qs, sel, ktag = sys.argv[1:9]
build_eps = sys.argv[9] if len(sys.argv) > 9 else None
n, _ = read_stats(data)
dev = 'cuda'
built = False
if not os.path.exists(qcache):  # rebuild QTO's forward matrix (T = 1) once, delete it at the end
    prefix = qcache[:-len('_forward_T1.0.pt')]
    PY = [sys.executable, '-W', 'ignore', '-m']
    subprocess.run(PY + ['duoprop.build', '--data', data, '--ckpt', ckpt, '--out', prefix, '--mode', 'forward', '--temps', '1.0',
                         '--eps', build_eps, '--shard', '0', '--nshards', '1'], check=True)
    subprocess.run(PY + ['duoprop.build', '--data', data, '--out', prefix, '--mode', 'forward', '--merge', '--nshards', '1',
                         '--temps', '1.0'], check=True)
    built = True
F = make_scorer(ckpt, dev).features().float()
F = (F - F.mean(0)) / (F.std(0) + 1e-6)
F = F / F.norm(dim=1, keepdim=True)
q, easy, hard = load_split(data, 'test')
rec = [json.loads(l) for l in open(f'runs/results/{sel}_coll_test.jsonl')][-1]


def run(reasoner, tag):
    out = {}
    t0 = time.time()
    for s, typ in STRUCT.items():
        if s not in q:
            continue
        for qq in sorted(q[s]):
            score = reasoner.execute(qq, s)
            r = filtered_ranks(score, easy[qq], hard[qq], dev)
            sc = score.clone()
            if easy[qq]:
                sc[torch.tensor(list(easy[qq]), device=dev)] = -1.0
            v, i = sc.topk(100)
            w = v.clamp(min=0)
            c = (w.unsqueeze(1) * F[i]).sum(0) / w.sum().clamp(min=1e-12)
            a = F[torch.tensor(list(hard[qq]), device=dev)].mean(0)
            cos = float(torch.nn.functional.cosine_similarity(c, a, dim=0))
            out[(typ, qq)] = (r.tolist(), [[int(x), round(float(y), 5)] for x, y in zip(i[:20].tolist(), v[:20].tolist())], round(cos, 4))
    print(f'{tag}: {len(out)} queries in {time.time() - t0:.0f}s', flush=True)
    return out


adjA = load_cache(qcache, n, dev, None if qeps == 'none' else float(qeps))
A = run(Reasoner(adjA, n, dev, float(qs)), 'QTO')
del adjA
torch.cuda.empty_cache()
m = torch.load(f'cache/art/match_{ktag}.pt', weights_only=False) if rec['row_beta'] > 0 else None
adjB = load_cache(rec['cache'], n, dev, None if rec['eps'] == 'none' else float(rec['eps']), m, rec['row_beta'])
if rec.get('irreflexive'):
    irreflexive(adjB, data)
nbr, val = get_knn(data, rec['kind'], rec['k'], ktag)
kw = dict(mode='max') if rec['agg'] == 'max' else dict(mode='pnorm', p=float(rec['agg']))
B = run(CollReasoner(adjB, n, dev, neg_scale=rec['neg_scale'], nbr=nbr, val=val, beta=rec['beta'],
                     update=rec.get('update', 'add'), steps=rec.get('steps', 1), **kw), 'DuoProp')
os.makedirs('runs/paper', exist_ok=True)
with gzip.open(f'runs/paper/perquery_{name}.jsonl.gz', 'wt') as f:
    f.write(json.dumps(dict(meta=dict(name=name, data=data, ckpt=ckpt, qto_cache=qcache, qto_eps=qeps, qto_s=float(qs), sel=sel,
                                      duoprop=dict((k, rec[k]) for k in ('cache', 'row_beta', 'eps', 'neg_scale', 'kind', 'k',
                                                                          'beta', 'agg', 'irreflexive'))))) + '\n')
    for key in A:
        typ, qq = key
        f.write(json.dumps(dict(t=typ, q=qq, h=sorted(hard[qq]), rA=A[key][0], rB=B[key][0], topA=A[key][1], topB=B[key][1],
                                cA=A[key][2], cB=B[key][2])) + '\n')
if built:
    os.remove(qcache)
print('PART DONE', flush=True)
