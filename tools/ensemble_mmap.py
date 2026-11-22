"""Low-memory `python -m duoprop.ensemble <data> <out.pt> --caches ...`: the same weighted mean of Duo atom matrices
(matchability scaling and irreflexive prior applied to every member as in load_cache / irreflexive), but the member
caches are memory-mapped and fused relation by relation, so host memory holds little more than the output.
Needed for FB15k members of 50+ GB each (duoprop.ensemble keeps every member's entries in memory at once).
Usage: python tools/ensemble_mmap.py <data> <out.pt> <cache>:<match>:<rb>[:<weight>] [...]"""
import os
import sys

import torch

data, out = sys.argv[1], sys.argv[2]
specs = [s.split(':') for s in sys.argv[3:]]
w = [float(s[3]) if len(s) > 3 else 1.0 for s in specs]
total = sum(w)
n = int(open(data + '/stats.txt').read().split('\n')[0].split(' ')[-1])
reflexive = set()
for line in open(os.path.join(data, 'train.txt')):
    h, r, t = line.split()
    if h == t:
        reflexive.add(int(r))
mats = [torch.load(s[0], map_location='cpu', weights_only=False, mmap=True) for s in specs]
match = [torch.load(s[1], weights_only=False) if float(s[2]) > 0 else None for s in specs]
for s, wi in zip(specs, w):
    print(f'{s[0]} (weight {wi / total:.2f}) rb={s[2]} irr=True', flush=True)
crow, col, val = [], [], []
for r in range(len(mats[0]['crow'])):
    kk, vv = [], []
    for i, s in enumerate(specs):
        cr, co, va = mats[i]['crow'][r], mats[i]['col'][r], mats[i]['val'][r]
        rows = torch.repeat_interleave(torch.arange(n), cr[1:] - cr[:-1])
        rb = float(s[2])
        if match[i] is not None and rb > 0:
            observed = torch.zeros(n, dtype=torch.bool)
            observed[rows[va >= 1.0]] = True
            scale = torch.where(observed, torch.ones(n), match[i][r].float().pow(rb))
            va = torch.where(va >= 1.0, va, va * scale[rows])
        va = va.float()
        if r not in reflexive:
            va = torch.where(co.long() == rows, torch.zeros_like(va), va)
        kk.append(rows * n + co.long())
        vv.append(va * (w[i] / total))
    uk, inv = torch.unique(torch.cat(kk), return_inverse=True)
    sm = torch.zeros(len(uk)).index_add_(0, inv, torch.cat(vv))
    c = torch.zeros(n + 1, dtype=torch.long)
    c[1:] = torch.bincount(uk // n, minlength=n).cumsum(0)
    crow.append(c), col.append((uk % n).to(torch.int32)), val.append(sm)
    del kk, vv, uk, inv
torch.save({'crow': crow, 'col': col, 'val': val}, out)
print('saved', out, 'nnz =', sum(v.numel() for v in val), flush=True)
