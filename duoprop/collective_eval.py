"""Prop plug-in evaluation: neighbourhood propagation after every projection (+ optional p-norm
existential aggregation) on top of any truth-matrix cache.  Protocol: grid on validation (fixed 1000
queries per type), select by validation AVG_p (--criterion pos, the main rule of the paper) or AVG_p + AVG_n
(--criterion sum), then evaluate base and the selected setting once on test.
  --tag T        take cache / eps / row_beta / neg_scale from FINAL_SELECTION.json (best_valid of tag T),
                 matchability from cache/art/match_T.pt
  --cache ...    or give the cache explicitly (e.g. forward T=1 atoms without Duo); --tag then only
                 names the backbone whose embedding builds the 'emb' kNN graph"""
import argparse
import glob
import json
import os
import re
import subprocess

import torch

from duoprop.reason import Reasoner, evaluate, load_cache, load_split

DS = {'FB15k-237-betae': 'fb237', 'FB15k-betae': 'fb15k', 'NELL-betae': 'nell'}


class AggReasoner(Reasoner):
    def __init__(self, *a, mode='max', lam=1.0, p=2.0, **kw):
        super().__init__(*a, **kw)
        self.mode, self.lam, self.p = mode, lam, p

    def project(self, T, r, mats=None):
        mats = self.cur if mats is None else mats
        rows = torch.nonzero(T > 0, as_tuple=True)[0]
        out = torch.zeros(self.n, device=self.dev)
        if len(rows) == 0:
            return out
        pos, col, val = mats[r].gather(rows)
        x = T[rows][pos] * val
        out.scatter_reduce_(0, col, x, reduce='amax', include_self=True)
        if self.mode == 'max' or len(rows) == 1:
            return out
        if self.mode == 'nor':
            s = torch.zeros(self.n, device=self.dev).scatter_add_(0, col, torch.log1p(-x.clamp(max=1 - 1e-6)))
            return out + self.lam * ((1 - torch.exp(s)) - out)
        s = torch.zeros(self.n, device=self.dev).scatter_add_(0, col, x.pow(self.p))
        return s.pow(1 / self.p).clamp(max=1.0)


def triples(data):
    for line in open(os.path.join(data, 'train.txt')):
        yield map(int, line.split())


def profiles(data, n, nrel, kind, dim=1 << 16):
    if kind == 'ent':
        P = torch.zeros(n, n, device='cuda')
        for h, r, t in triples(data):
            P[h, t] = 1.0
        P = ((P + P.t()) > 0).float()
    elif kind == 'rel':
        P = torch.zeros(n, dim, device='cuda', dtype=torch.half)
        hs, fs = [], []
        for h, r, t in triples(data):  # train.txt contains both directions
            hs.append(h)
            fs.append((r * 1000003 + t * 7919) % dim)
        P.index_put_((torch.tensor(hs, device='cuda'), torch.tensor(fs, device='cuda')),
                     torch.ones(len(hs), device='cuda', dtype=torch.half))
    else:
        P = torch.zeros(n, nrel, device='cuda')
        for h, r, t in triples(data):
            P[h, r] += 1.0
        P = torch.log1p(P)
    df = sum((P[i:i + 4096] > 0).float().sum(0) for i in range(0, n, 4096))  # chunked: a full bool sum is int64
    idf = torch.log((n + 1) / (df + 1)).to(P.dtype)
    P *= idf
    norm = torch.cat([P[i:i + 4096].float().norm(dim=1) for i in range(0, n, 4096)])
    P /= norm.clamp(min=1e-12).to(P.dtype).unsqueeze(1)
    return P.half()


def knn(P, k, block=2048):
    n = P.shape[0]
    nbr = torch.empty(n, k, dtype=torch.long, device='cuda')
    val = torch.empty(n, k, device='cuda')
    for s0 in range(0, n, block):
        G = (P[s0:s0 + block] @ P.t()).float()
        G[torch.arange(G.shape[0]), torch.arange(s0, s0 + G.shape[0])] = 0
        v, i = G.topk(k, dim=1)
        nbr[s0:s0 + len(v)], val[s0:s0 + len(v)] = i, v.clamp(min=0)
    return nbr, val


class CollReasoner(AggReasoner):
    """update='add': x + beta*S(x);  update='or': probabilistic sum x + beta*S(x) - x*beta*S(x), i.e. the fuzzy OR of
    the atom's own truth and its neighbours' evidence (entries near 1 stay put).  steps: propagation steps per hop."""

    def __init__(self, *a, nbr=None, val=None, beta=0.0, update='add', steps=1, **kw):
        super().__init__(*a, **kw)
        self.nbr, self.val, self.beta, self.update, self.steps = nbr, val, beta, update, steps

    def project(self, T, r, mats=None):
        out = super().project(T, r, mats)
        if self.beta > 0:
            for _ in range(self.steps):
                S = (self.val * out[self.nbr]).sum(1) / self.val.sum(1).clamp(min=1e-12)
                out = out + self.beta * (S if self.update == 'add' else (1 - out).clamp(min=0) * S)
        return out


def irreflexive(adj, data):
    """Irreflexive prior: zero A_r[h,h] for every relation that never links an entity to itself in the training
    graph (most relations; self-loops are a frequent kind of confident pseudo atom that both views agree on)."""
    reflexive = set()
    for line in open(os.path.join(data, 'train.txt')):
        h, r, t = line.split()
        if h == t:
            reflexive.add(int(r))
    for r, M in enumerate(adj):
        if r not in reflexive:
            rows = torch.repeat_interleave(torch.arange(M.n, device=M.col.device), M.crow[1:] - M.crow[:-1])
            M.val = torch.where(M.col.long() == rows, torch.zeros_like(M.val), M.val)
    return adj


def observe(adj, data, split='valid'):
    """Observed-graph protocol for test queries: BetaE defines the easy answers of test queries on the validation
    graph (train + valid edges), i.e. those edges are known at test time, so they get truth 1 like training edges.
    The original code (and QTO) only set training edges to 1; a large negation scaling partly compensates for it."""
    edges = {}
    for line in open(os.path.join(data, f'{split}.txt')):
        h, r, t = map(int, line.split())
        edges.setdefault(r, []).append((h, t))
    for r, M in enumerate(adj):
        if r not in edges:
            continue
        n, dev = M.n, M.col.device
        rows = torch.repeat_interleave(torch.arange(n, device=dev), M.crow[1:] - M.crow[:-1])
        e = torch.tensor(edges[r], device=dev)
        keys = torch.cat([rows * n + M.col.long(), e[:, 0] * n + e[:, 1]])
        vals = torch.cat([M.val, torch.ones(len(e), device=dev)])
        uk, inv = torch.unique(keys, return_inverse=True)
        uv = torch.zeros(len(uk), device=dev).scatter_reduce_(0, inv, vals, reduce='amax', include_self=False)
        crow = torch.zeros(n + 1, dtype=torch.long, device=dev)
        crow[1:] = torch.bincount(uk // n, minlength=n).cumsum(0)
        M.crow, M.col, M.val = crow, (uk % n).to(M.col.dtype), uv
    return adj


def tag_ckpt(tag):
    """checkpoint of a pipeline tag, as given to run_ckpt in runs/jobs.txt (shell expressions expanded)"""
    if not os.path.exists('runs/jobs.txt'):
        raise FileNotFoundError(f"pass --ckpt to build the embedding neighbour graph for tag {tag}")
    for line in open('runs/jobs.txt'):
        if 'run_ckpt' in line and f'--tag {tag} ' in line:
            expr = re.search(r'--ckpt (\$\(.*?\)|\S+)', line).group(1)
            path = subprocess.check_output(['bash', '-c', f'echo {expr}']).decode().strip()
            return path if path.endswith('checkpoint') else os.path.join(path, 'checkpoint')
    raise KeyError(tag)


def emb_knn(data, tag, k=100, ckpt=None):
    """kNN graph of the backbone's own entity representation (cosine of standardised features)"""
    from duoprop.scorer import make_scorer
    X = make_scorer(ckpt or tag_ckpt(tag), 'cuda').features().float()
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = X / X.norm(dim=1, keepdim=True)
    n = X.shape[0]
    nbr, val = torch.empty(n, k, dtype=torch.long, device='cuda'), torch.empty(n, k, device='cuda')
    for s0 in range(0, n, 4096):
        G = X[s0:s0 + 4096] @ X.t()
        G[torch.arange(G.shape[0]), torch.arange(s0, s0 + G.shape[0])] = -1
        v, i = G.topk(k, dim=1)
        nbr[s0:s0 + len(v)], val[s0:s0 + len(v)] = i, v.clamp(min=0)
    return nbr, val


def get_knn(data, kind, k, tag=None, ckpt=None):
    """kind: rel | ent | type | emb, or several joined by '+' (neighbour lists are concatenated)"""
    if '+' in kind:
        parts = [tuple(get_knn(data, kd, k, tag, ckpt)) for kd in kind.split('+')]
        return torch.cat([p[0] for p in parts], 1), torch.cat([p[1] for p in parts], 1)
    ds = DS[os.path.basename(data.rstrip("/"))]
    path = f'cache/art/knn_{ds}_{kind}.pt' if kind != 'emb' else f'cache/art/knn_{ds}_emb_{tag}.pt'
    if kind == 'emb' and not os.path.exists(path):
        nbr, val = emb_knn(data, tag, ckpt=ckpt)
        torch.save((nbr.cpu(), val.cpu()), path)
    if not os.path.exists(path):
        st = open(os.path.join(data, 'stats.txt')).read().split('\n')
        n, nrel = int(st[0].split(' ')[-1]), int(st[1].split(' ')[-1])
        P = profiles(data, n, nrel, kind)
        nbr, val = knn(P, 100)
        del P
        torch.cuda.empty_cache()
        torch.save((nbr.cpu(), val.cpu()), path)
    return (x[:, :k].cuda() for x in torch.load(path))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--tag')
    ap.add_argument('--ckpt', help='frozen checkpoint used to build the embedding neighbour graph for this tag')
    ap.add_argument('--cache')
    ap.add_argument('--match')
    ap.add_argument('--rb', type=float, default=0.0)
    ap.add_argument('--eps', default='none')
    ap.add_argument('--neg_scale', type=float, default=1.0)
    ap.add_argument('--kind', default='rel')
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--betas', default='0,0.05,0.1,0.2')
    ap.add_argument('--aggs', default='max,2,4')
    ap.add_argument('--extra', default='0.3:max', help='additional beta:agg pairs')
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--no_test', action='store_true')
    ap.add_argument('--select_from', help='prefix X: pick cache / beta / aggregation by the validation score of --criterion over '
                                          'runs/results/X*_coll_valid.jsonl (joint search over Duo and Prop), '
                                          'then evaluate it and the best beta=0 setting of the same pool once on test')
    ap.add_argument('--refine_top', type=int, default=0,
                    help='with --select_from: re-evaluate the top N candidates (and the top N Duo-only ones) on the full '
                         'validation set and select on that')
    ap.add_argument('--criterion', default='pos', choices=['sum', 'pos'],
                    help="selection score: AVG_p + AVG_n ('sum'), or AVG_p only ('pos'): validation negation does not "
                         'transfer to test (validation edges are observed at test time but not set to 1), see the evaluation protocol in the paper')
    ap.add_argument('--irreflexive', action='store_true', help='zero self-loops of relations without training self-loops')
    ap.add_argument('--sel_tag', help='take eps / row_beta / neg_scale from the Duo selection of a pipeline tag '
                                      '(runs/results/<tag>_test.jsonl), read when the job starts')
    ap.add_argument('--update', default='add', choices=['add', 'or'], help='propagation update (see CollReasoner)')
    ap.add_argument('--steps', type=int, default=1, help='propagation steps per projection')
    ap.add_argument('--view_weights', help="weight of each kNN view, in the order of --kind (e.g. '1,2' for rel+emb)")
    ap.add_argument('--sim_power', type=float, default=1.0, help='neighbour weights w -> w^power')
    ap.add_argument('--neg_fixed', type=float, help="negation scaling fixed to the base reasoner's published value "
                                                    '(overrides every other source; validation favours 1 for protocol '
                                                    'reasons, see the paper)')
    a = ap.parse_args()
    if a.sel_tag:
        s = [json.loads(l) for l in open(f'runs/results/{a.sel_tag}_test.jsonl')][-1]
        a.eps, a.rb, a.neg_scale = str(s['eps']), float(s['row_beta']), float(s['neg_scale'])
    picks = None
    if a.select_from:
        files = sorted({f for p in a.select_from.split(',') for f in glob.glob(f'runs/results/{p}*_coll_valid.jsonl')})
        recs = [json.loads(l) for f in files for l in open(f)]
        score = (lambda r: r['res']['AVG_p']) if a.criterion == 'pos' else (lambda r: r['res']['AVG_p'] + r['res']['AVG_n'])
        best = max(recs, key=score)
        base = max((r for r in recs if r['beta'] == 0 and r['agg'] == 'max'), key=score)
        print(f"{len(recs)} validation records; Duo only: {base['cache']} ({score(base):.2f}); "
              f"selected: {best['cache']} beta={best['beta']} agg={best['agg']} ({score(best):.2f})", flush=True)
        picks = [base, best]
        a.cache, a.eps, a.rb, a.neg_scale, a.kind, a.k = (best[x] for x in ('cache', 'eps', 'row_beta', 'neg_scale', 'kind', 'k'))
        a.irreflexive = best.get('irreflexive', False)
        a.update, a.steps = best.get('update', 'add'), best.get('steps', 1)
        a.view_weights, a.sim_power = best.get('view_weights'), best.get('sim_power', 1.0)
    if a.tag and not a.cache:
        sel = json.load(open('runs/results/FINAL_SELECTION.json'))
        b = next(v['best_valid'] for v in sel.values() if v and v['best_valid'].get('tag') == a.tag)
        a.cache, a.eps, a.rb, a.neg_scale = b['cache'], b['eps'], b.get('row_beta', 0.0), b['neg_scale']
        a.match = a.match or f'cache/art/match_{a.tag}.pt'
    if a.neg_fixed is not None and not picks:
        a.neg_scale = a.neg_fixed
    n = int(open(os.path.join(a.data, 'stats.txt')).read().split('\n')[0].split(' ')[-1])
    rm = torch.load(a.match, weights_only=False) if a.rb > 0 else None
    adj = load_cache(a.cache, n, 'cuda', None if a.eps == 'none' else float(a.eps), rm, a.rb)
    if a.irreflexive:
        irreflexive(adj, a.data)

    def load_knn():
        nb, vl = get_knn(a.data, a.kind, a.k, a.tag, a.ckpt)
        if a.sim_power != 1.0:
            vl = vl.pow(a.sim_power)
        if a.view_weights:
            vl = vl * torch.tensor([float(x) for x in a.view_weights.split(',')], device=vl.device).repeat_interleave(a.k)
        return nb, vl

    nbr, val = load_knn()
    grid =[(float(bt), ag) for bt in a.betas.split(',') for ag in a.aggs.split(',')]
    grid += [(float(x.split(':')[0]), x.split(':')[1]) for x in a.extra.split(',') if x]

    def reasoner(beta, agg):
        kw = dict(mode='max') if agg == 'max' else dict(mode='pnorm', p=float(agg))
        return CollReasoner(adj, n, 'cuda', neg_scale=a.neg_scale, nbr=nbr, val=val, beta=beta, update=a.update,
                            steps=a.steps, **kw)

    def run(split, beta, agg, q, e, h, limit):
        res = evaluate(reasoner(beta, agg), q, e, h, limit=limit)
        rec = dict(name=a.name, cache=a.cache, eps=a.eps, row_beta=a.rb, neg_scale=a.neg_scale, kind=a.kind,
                   k=a.k, beta=beta, agg=agg, split=split, limit=limit, res=res, irreflexive=a.irreflexive,
                   update=a.update, steps=a.steps, view_weights=a.view_weights, sim_power=a.sim_power)
        with open(f'runs/results/{a.name}_coll_{split}.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f"{split} beta={beta} agg={agg:4s} AVG_p={res['AVG_p']:.2f} AVG_n={res['AVG_n']:.2f} | " +
              ' '.join(f"{t}:{res[t]['MRR']:.1f}" for t in res if t not in ('AVG_p', 'AVG_n')), flush=True)
        return res

    def switch(r):
        """make record r's atoms and calibration current (reload only when they differ)"""
        global adj, nbr, val
        irr = r.get('irreflexive', False)
        knn = (r['kind'], r['k'], r.get('view_weights'), r.get('sim_power', 1.0))
        if knn != (a.kind, a.k, a.view_weights, a.sim_power):
            a.kind, a.k, a.view_weights, a.sim_power = knn
            nbr, val = load_knn()
        if r['cache'] != a.cache or r['row_beta'] != a.rb or r['eps'] != a.eps or irr != a.irreflexive:
            del adj
            torch.cuda.empty_cache()
            rm = torch.load(a.match, weights_only=False) if r['row_beta'] > 0 else None
            adj = load_cache(r['cache'], n, 'cuda', None if r['eps'] == 'none' else float(r['eps']), rm, r['row_beta'])
            if irr:
                irreflexive(adj, a.data)
        a.cache, a.eps, a.rb, a.neg_scale, a.irreflexive = r['cache'], r['eps'], r['row_beta'], r['neg_scale'], irr
        a.update, a.steps = r.get('update', 'add'), r.get('steps', 1)

    if picks:
        if a.refine_top:
            # second stage: the top candidates of the subset search (overall and Duo-only) on the full validation set
            ident = lambda r: (r['cache'], r['eps'], r['row_beta'], r['neg_scale'], r.get('irreflexive', False),
                               r['beta'], r['agg'], r.get('update', 'add'), r.get('steps', 1), r['kind'], r['k'],
                               r.get('view_weights'), r.get('sim_power', 1.0))
            top = sorted(recs, key=score, reverse=True)[:a.refine_top]
            topb = sorted((r for r in recs if r['beta'] == 0 and r['agg'] == 'max'), key=score, reverse=True)[:a.refine_top]
            cands = sorted({ident(r): r for r in top + topb}.values(),
                           key=lambda r: (r['cache'], r.get('irreflexive', False), r['row_beta']))
            q, e, h = load_split(a.data, 'valid')
            full = []
            for r in cands:
                switch(r)
                res = run('valid', r['beta'], r['agg'], q, e, h, None)
                full.append((res['AVG_p'] + (0 if a.criterion == 'pos' else res['AVG_n']), r))
            best = max(full, key=lambda x: x[0])
            base = max((x for x in full if x[1]['beta'] == 0 and x[1]['agg'] == 'max'), key=lambda x: x[0])
            print(f'full-validation selection over {len(cands)} candidates: Duo only {base[1]["cache"]} '
                  f'({base[0]:.2f}); selected {best[1]["cache"]} beta={best[1]["beta"]} agg={best[1]["agg"]} '
                  f'({best[0]:.2f})', flush=True)
            picks = [base[1], best[1]]
        q, e, h = load_split(a.data, 'test')
        for r in picks:
            switch(r)
            run('test', r['beta'], r['agg'], q, e, h, None)
        print('PART DONE', flush=True)
        raise SystemExit

    q, e, h = load_split(a.data, 'valid')
    scores = {}
    for beta, agg in grid:
        res = run('valid', beta, agg, q, e, h, a.limit)
        scores[(beta, agg)] = res['AVG_p'] + res['AVG_n']
    best = max(scores, key=scores.get)
    print('selected on valid:', best, f'{scores[best] - scores.get((0.0, "max"), float("nan")):+.2f}', flush=True)
    if not a.no_test:
        q, e, h = load_split(a.data, 'test')
        run('test', 0.0, 'max', q, e, h, None)
        if best != (0.0, 'max'):
            run('test', best[0], best[1], q, e, h, None)
    print('PART DONE', flush=True)
