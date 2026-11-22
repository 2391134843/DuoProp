"""Fast fuzzy-logic CLQA evaluator (QTO semantics) over per-relation CSR truth matrices.

Reproduces model.py/main.py of the original code exactly when neg='atom' (verified: identical
per-type MRR on FB15k-237), but with negation applied after the projection it scopes over
(neg='after', the BetaE definition of pni; identical to 'atom' for all other query types).
"""
import argparse
import json
import os
import pickle
import random
import time

import torch

STRUCT = {
    ('e', ('r',)): '1p', ('e', ('r', 'r')): '2p', ('e', ('r', 'r', 'r')): '3p',
    (('e', ('r',)), ('e', ('r',))): '2i', (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip', (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    (('e', ('r',)), ('e', ('r', 'n'))): '2in', (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp', (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u', ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up',
}
ORDER = ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up', '2in', '3in', 'inp', 'pin', 'pni']
POS, NEG = ORDER[:9], ORDER[9:]


class CSR:
    def __init__(self, crow, col, val, n):
        self.crow, self.col, self.val, self.n = crow, col, val, n

    def gather(self, rows):
        start = self.crow[rows]
        lens = self.crow[rows + 1] - start
        total = int(lens.sum())
        if total == 0:
            e = torch.empty(0, dtype=torch.long, device=rows.device)
            return e, e, torch.empty(0, device=rows.device)
        pos = torch.repeat_interleave(torch.arange(len(rows), device=rows.device), lens)
        base = torch.repeat_interleave(start - torch.cumsum(lens, 0) + lens, lens)
        idx = base + torch.arange(total, device=rows.device)
        return pos, self.col[idx].long(), self.val[idx]

    def dense_rows(self, rows):
        pos, col, val = self.gather(rows)
        out = torch.zeros(len(rows), self.n, device=rows.device)
        out[pos, col] = val
        return out


def lift_evidence(crow, col, val, n, r, keys, lam):
    """Open-world lift: atoms supported by a mined multi-hop rule get truth >= lam (lam > 0), or, for
    lam < 0, existing atoms with rule support are boosted multiplicatively by (1 + |lam|)."""
    lo = torch.searchsorted(keys, torch.tensor(r * n * n))
    hi = torch.searchsorted(keys, torch.tensor((r + 1) * n * n))
    if hi <= lo:
        return crow, col, val
    ek = keys[lo:hi] - r * n * n
    rows = torch.repeat_interleave(torch.arange(n), crow[1:] - crow[:-1])
    if lam < 0:
        k = rows * n + col.long()
        idx = torch.searchsorted(ek, k).clamp(max=len(ek) - 1)
        hit = (ek[idx] == k) & (val < 1.0)
        val = torch.where(hit, (val * (1 - lam)).clamp(max=1 - 1e-4), val)
        return crow, col, val
    k = torch.cat([rows * n + col.long(), ek])
    v = torch.cat([val, torch.full((len(ek),), lam)])
    uk, inv = torch.unique(k, return_inverse=True)
    uv = torch.zeros(len(uk)).scatter_reduce_(0, inv, v, reduce='amax', include_self=False)
    rows = uk // n
    crow = torch.zeros(n + 1, dtype=torch.long)
    crow[1:] = torch.bincount(rows, minlength=n).cumsum(0)
    return crow, (uk % n).to(torch.int32), uv


def load_cache(path, n, device, eps=None, row_match=None, row_beta=0.0, evid=None, evid_lambda=0.0, obs_scale=1.0):
    """row_match: (nrel, n) matchability m(h, r); if given, every non-observed entry of row h of
    relation r is scaled by m(h, r)^row_beta (query-side matchability)."""
    d = torch.load(path, map_location='cpu', weights_only=False)
    mats = []
    for r, (crow, col, val) in enumerate(zip(d['crow'], d['col'], d['val'])):
        if eps is not None:
            keep = val >= eps
            rows = torch.repeat_interleave(torch.arange(n), crow[1:] - crow[:-1])[keep]
            col, val = col[keep], val[keep]
            crow = torch.zeros(n + 1, dtype=torch.long)
            crow[1:] = torch.bincount(rows, minlength=n).cumsum(0)
        if row_match is not None and row_beta > 0:
            # a query with an observed answer has one for sure (m = 1); otherwise use the learned
            # probability that the query has any answer.  Applied after thresholding, so the sparsity
            # pattern (and the ranking inside every row) is unchanged.
            rows = torch.repeat_interleave(torch.arange(n), crow[1:] - crow[:-1])
            observed = torch.zeros(n, dtype=torch.bool)
            observed[rows[val >= 1.0]] = True
            scale = torch.where(observed, torch.ones(n), row_match[r].float().pow(row_beta))
            val = torch.where(val >= 1.0, val, val * scale[rows])
        if obs_scale != 1.0:
            # rows with observed answers still miss answers (incomplete KG): lift their unobserved entries
            rows = torch.repeat_interleave(torch.arange(n), crow[1:] - crow[:-1])
            observed = torch.zeros(n, dtype=torch.bool)
            observed[rows[val >= 1.0]] = True
            val = torch.where((val < 1.0) & observed[rows], (val * obs_scale).clamp(max=1 - 1e-4), val)
        if evid is not None and evid_lambda != 0:
            crow, col, val = lift_evidence(crow, col, val, n, r, evid, evid_lambda)
        mats.append(CSR(crow.to(device), col.to(device), val.float().to(device), n))
    return mats


class Reasoner:
    """roles: optional {'neg' | 'union' | 'and': adjacency} - role-aware calibration, i.e. atoms used under
    negation, inside union branches or inside intersection branches read their own calibrated matrices;
    every other projection reads `adj`."""

    def __init__(self, adj, n, device, neg_scale=1.0, neg='after', roles=None, neg_set=None):
        self.adj, self.n, self.dev, self.neg_scale, self.neg = adj, n, device, neg_scale, neg
        self.roles = roles or {}
        self.cur = adj
        self.neg_set = neg_set  # rho: negate a cardinality-adaptive answer set (cumulative-mass prediction set)

    def negate(self, x):
        out = torch.clamp(self.neg_scale * x, max=1.0)
        if self.neg_set:
            v, i = torch.sort(x, descending=True)
            mass = torch.cumsum(v, 0)
            inside = (mass - v) < self.neg_set * mass[-1].clamp(min=1e-12)  # smallest prefix reaching rho of the mass
            inside &= v > 0
            member = torch.zeros_like(x)
            member[i[inside]] = 1.0
            out = torch.maximum(out, member)
        return 1.0 - out

    def project(self, T, r, mats=None):
        mats = self.cur if mats is None else mats
        rows = torch.nonzero(T > 0, as_tuple=True)[0]
        out = torch.zeros(self.n, device=self.dev)
        if len(rows):
            pos, col, val = mats[r].gather(rows)
            out.scatter_reduce_(0, col, T[rows][pos] * val, reduce='amax', include_self=True)
        return out

    def project_neg_atom(self, T, r, chunk=512):
        rows = torch.nonzero(T > 0, as_tuple=True)[0]
        out = torch.zeros(self.n, device=self.dev)
        for i in range(0, len(rows), chunk):
            rr = rows[i:i + chunk]
            d = 1.0 - torch.clamp(self.neg_scale * self.adj[r].dense_rows(rr), max=1.0)
            out = torch.maximum(out, (T[rr].unsqueeze(1) * d).max(0)[0])
        return out

    def path(self, T, rels):
        i = 0
        while i < len(rels):
            if i + 1 < len(rels) and rels[i + 1] == -2:
                if self.neg == 'atom':
                    T = self.project_neg_atom(T, rels[i])
                else:
                    T = self.negate(self.project(T, rels[i], self.roles.get('neg')))
                i += 2
            else:
                T = self.project(T, rels[i])
                i += 1
        return T

    def execute(self, q, s):
        if all(x in ('r', 'n') for x in s[-1]):
            if s[0] == 'e':
                T = torch.zeros(self.n, device=self.dev)
                T[q[0]] = 1.0
            else:
                T = self.execute(q[0], s[0])
            return self.path(T, q[-1])
        if s[-1] == ('u',):
            Ts = torch.stack([self.branch(q[i], s[i], 'union') for i in range(len(s) - 1)])
            return 1.0 - (1.0 - Ts).prod(0)
        return torch.stack([self.branch(q[i], s[i], 'and') for i in range(len(s))]).prod(0)

    def branch(self, q, s, role):
        prev = self.cur
        self.cur = self.roles.get(role, prev)
        try:
            return self.execute(q, s)
        finally:
            self.cur = prev


def filtered_ranks(score, easy, hard, dev):
    ranking = torch.argsort(torch.argsort(score, descending=True))
    ans = list(easy) + list(hard)
    cur, ind = torch.sort(ranking[torch.tensor(ans, device=dev)])
    cur = cur - torch.arange(len(ans), device=dev) + 1
    return cur[ind >= len(easy)].float()


def evaluate(reasoner, queries, easy, hard, types=ORDER, limit=None, seed=0):
    res = {}
    for s, name in STRUCT.items():
        if name not in types or s not in queries:
            continue
        qs = sorted(queries[s])
        if limit and len(qs) > limit:
            qs = random.Random(seed).sample(qs, limit)
        m = {'MRR': 0., 'H1': 0., 'H3': 0., 'H10': 0.}
        for q in qs:
            r = filtered_ranks(reasoner.execute(q, s), easy[q], hard[q], reasoner.dev)
            m['MRR'] += (1.0 / r).mean().item()
            m['H1'] += (r <= 1).float().mean().item()
            m['H3'] += (r <= 3).float().mean().item()
            m['H10'] += (r <= 10).float().mean().item()
        res[name] = {k: 100 * v / len(qs) for k, v in m.items()}
        res[name]['n'] = len(qs)
    for k, group in [('AVG_p', POS), ('AVG_n', NEG)]:
        vals = [res[t]['MRR'] for t in group if t in res]
        if vals:
            res[k] = sum(vals) / len(vals)
    return res


def fmt(res):
    keys = [k for k in ['AVG_p', 'AVG_n'] + ORDER if k in res]
    vals = [res[k]['MRR'] if isinstance(res[k], dict) else res[k] for k in keys]
    return ' '.join(f'{k:>6}' for k in keys) + '\n' + ' '.join(f'{v:6.2f}' for v in vals)


def load_split(data, split):
    ld = lambda f: pickle.load(open(os.path.join(data, f'{split}-{f}.pkl'), 'rb'))
    return ld('queries'), ld('easy-answers'), ld('hard-answers')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--caches', required=True, help='comma-separated cache files')
    ap.add_argument('--split', default='valid')
    ap.add_argument('--eps', default='none', help='comma-separated eval thresholds (none = as stored)')
    ap.add_argument('--neg_scales', default='1')
    ap.add_argument('--neg', default='after', choices=['after', 'atom'])
    ap.add_argument('--types', default=','.join(ORDER))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', required=True, help='jsonl file to append results to')
    args = ap.parse_args()
    from duoprop.scorer import read_stats
    n, _ = read_stats(args.data)
    q, e, h = load_split(args.data, args.split)
    for cache in args.caches.split(','):
        for eps in args.eps.split(','):
            adj = load_cache(cache, n, 'cuda', None if eps == 'none' else float(eps))
            for s in args.neg_scales.split(','):
                t0 = time.time()
                res = evaluate(Reasoner(adj, n, 'cuda', float(s), args.neg), q, e, h,
                               args.types.split(','), args.limit)
                rec = dict(cache=os.path.basename(cache), eps=eps, neg_scale=float(s), neg=args.neg,
                           split=args.split, limit=args.limit, res=res)
                with open(args.out, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                print(f'== {os.path.basename(cache)} eps={eps} s={s} ({time.time() - t0:.0f}s)\n{fmt(res)}', flush=True)
            del adj
            torch.cuda.empty_cache()
