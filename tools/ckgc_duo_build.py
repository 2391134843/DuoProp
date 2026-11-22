"""Duo on top of CKGC (Xiao & Cao, 2024): truth matrices from a trained CKGC model, read from both sides.
CKGC's calibration of (h, r, t) is min(1, min(1, N_hr softmax_t f(h, r, t)) * max(1e-5, 1 + W[h, r])) with N_hr the number of
training tails of (h, r), floored at alpha_floor (0.1 in CKGC); W is learned on complex training queries.  Duo combines the
head-side value of (h, r, t) with the tail-side value of (t, r^-1, h): A = P_r^a * (P_r^-1)^T^(1-a); a = 1 is CKGC itself.
Training edges get truth 1 (validation edges are added at test time by the evaluation, as in CKGC).  Blocked over heads, so
NELL995 (63k entities) fits.  Also writes the embedding kNN graph of CKGC's link predictor for Prop.
Output: <OUT>_dual_T1.0_a<a>_b0.0.pt for every a, cache/art/knn_<ds>_emb_<KNN_TAG>.pt
python tools/ckgc_duo_build.py DATA CKGC_CKPT OUT ALPHAS EPS KNN_TAG [ALPHA_FLOOR]"""
import collections
import os
import sys

import torch

from duoprop.collective_eval import DS
from duoprop.scorer import read_stats

data, ckpt, out, alphas, eps, ktag = sys.argv[1:7]
alphas, eps = [float(a) for a in alphas.split(',')], float(eps)
floor = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
n, nrel = read_stats(data)
dev = 'cuda'
st = torch.load(ckpt, map_location='cpu', weights_only=False)['model']
E = st['kgc_model.entity.weight'].to(dev)
R = st['kgc_model.relation.weight'].to(dev)
W = st['weight'].to(dev)  # (n, nrel)
assert E.shape[0] == n and R.shape[0] == nrel, (E.shape, R.shape, n, nrel)
tails = collections.defaultdict(set)
for line in open(f'{data}/train.txt'):
    h, r, t = map(int, line.split())
    tails[(r, h)].add(t)
deg = torch.zeros(nrel, n, device=dev)
for (r, h), ts in tails.items():
    deg[r, h] = len(ts)
N = deg.clamp(min=floor)
E1, E2 = E.chunk(2, dim=-1)
B = 2048


def query(hs, r):
    """ComplEx query vectors of CKGC (real and imaginary parts concatenated) for heads hs and relation r"""
    h1, h2 = E1[hs], E2[hs]
    r1, r2 = R[r].chunk(2, dim=-1)
    return torch.cat([h1 * r1 - h2 * r2, h2 * r1 + h1 * r2], dim=-1)


def lse(r):
    """log-sum-exp over tails of every row of relation r"""
    v = torch.empty(n, device=dev)
    for s in range(0, n, B):
        hs = torch.arange(s, min(n, s + B), device=dev)
        v[s:s + len(hs)] = torch.logsumexp(query(hs, r) @ E.t(), dim=1)
    return v


def calib(S, rows, r, lz):
    """CKGC calibration of a score block S (rows = head ids of the block, softmax over all tails)"""
    P = torch.exp(S - lz[rows].unsqueeze(1)) * N[r, rows].unsqueeze(1)
    P = P.clamp(max=1.0) * (1 + W[rows, r]).clamp(min=1e-5).unsqueeze(1)
    return P.clamp(max=1.0)


res = {a: {'crow': [None] * nrel, 'col': [None] * nrel, 'val': [None] * nrel} for a in alphas}
for k in range(0, nrel, 2):
    for x, y in ((k, k + 1), (k + 1, k)):
        lx, ly = lse(x), lse(y)
        cr = {a: [torch.zeros(1, dtype=torch.long)] for a in alphas}
        cc, cv = {a: [] for a in alphas}, {a: [] for a in alphas}
        off = {a: 0 for a in alphas}
        all_t = torch.arange(n, device=dev)
        for s in range(0, n, B):
            hs = torch.arange(s, min(n, s + B), device=dev)
            Pf = calib(query(hs, x) @ E.t(), hs, x, lx)  # head side of (h, x, t): rows hs, all t
            Qb = torch.cat([calib(query(all_t[u:u + B], y) @ E[hs].t(), all_t[u:u + B], y, ly) for u in range(0, n, B)])  # (n, |hs|): tail side
            for a in alphas:
                A = Pf.clone() if a == 1.0 else Pf.pow(a) * Qb.t().pow(1 - a)
                for i, h in enumerate(hs.tolist()):
                    ts = tails.get((x, h))
                    if ts:
                        A[i, list(ts)] = 1.0
                rows, cols = torch.nonzero(A >= eps, as_tuple=True)
                cnt = torch.bincount(rows, minlength=len(hs)).cumsum(0) + off[a]
                cr[a].append(cnt.cpu())
                off[a] = int(cnt[-1]) if len(cnt) else off[a]
                cc[a].append(cols.to(torch.int32).cpu())
                cv[a].append(A[rows, cols].float().cpu())
                del A
            del Pf, Qb
        for a in alphas:
            res[a]['crow'][x] = torch.cat(cr[a])
            res[a]['col'][x] = torch.cat(cc[a])
            res[a]['val'][x] = torch.cat(cv[a])
    print('pair', k, 'done', flush=True)
for a in alphas:
    torch.save(res[a], f'{out}_dual_T1.0_a{a}_b0.0.pt')
    print('saved', a, sum(v.numel() for v in res[a]['val']), flush=True)
X = E.float()
X = (X - X.mean(0)) / (X.std(0) + 1e-6)
X = X / X.norm(dim=1, keepdim=True)
nbr, val = torch.empty(n, 100, dtype=torch.long, device=dev), torch.empty(n, 100, device=dev)
for s0 in range(0, n, 4096):
    G = X[s0:s0 + 4096] @ X.t()
    G[torch.arange(G.shape[0]), torch.arange(s0, s0 + G.shape[0])] = -1
    v, i = G.topk(100, dim=1)
    nbr[s0:s0 + len(v)], val[s0:s0 + len(v)] = i, v.clamp(min=0)
torch.save((nbr.cpu(), val.cpu()), f'cache/art/knn_{DS[os.path.basename(data.rstrip("/"))]}_emb_{ktag}.pt')
print('PART DONE', flush=True)
