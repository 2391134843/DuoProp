"""Build atomic truth matrices (one CSR per relation) from a link-predictor checkpoint (see scorer.py).

mode=forward : original QTO calibration, A_r[h,t] = min(1 - 1e-4, n_h * softmax_t(s(h,r,t)/T)), training edges = 1.
mode=dual    : bidirectional-consistent atoms (our method),
               A_r[h,t] = P_r[h,t]^alpha * P_r'[t,h]^(1-alpha) * (m_r[h] * m_r'[t])^beta,
               where r' is the inverse relation, P is the forward calibration above and m is the
               matchability (probability that a query has any answer, see match.py).  The paper uses beta = 0 and
               scales the rows without an observed answer by m^rho when the cache is loaded (reason.load_cache).
Several (T, alpha, beta) configurations are produced from a single scoring pass.

Usage: run one process per GPU with --shard k --nshards K, then once with --merge.
"""
import argparse
import itertools
import os

import torch

from duoprop.scorer import make_scorer, read_stats, read_train


def all_scores(sc, r, bs=512):
    n = sc.n
    S = torch.empty(n, n, device=sc.dev)
    for s in range(0, n, bs):
        hs = torch.arange(s, min(n, s + bs), device=sc.dev)
        S[s:s + len(hs)] = sc.scores(hs, r)
    return S


def calibrated(S, r, num, tails, T):
    P = torch.softmax(S / T, dim=1)
    P.mul_(num[r].unsqueeze(1)).clamp_(max=1 - 1e-4)
    hh, tt = tails[r]
    P[hh, tt] = 1.0
    return P


def to_csr(A, eps):
    n = A.shape[0]
    rows, cols = torch.nonzero(A >= eps, as_tuple=True)
    crow = torch.zeros(n + 1, dtype=torch.long, device=A.device)
    crow[1:] = torch.bincount(rows, minlength=n).cumsum(0)
    return crow.cpu(), cols.to(torch.int32).cpu(), A[rows, cols].float().cpu()


def tag(mode, T, a, b):
    return f'forward_T{T}' if mode == 'forward' else f'dual_T{T}_a{a}_b{b}'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt')
    ap.add_argument('--out', required=True, help='output prefix')
    ap.add_argument('--mode', default='dual', choices=['dual', 'forward'])
    ap.add_argument('--temps', default='1.0')
    ap.add_argument('--alphas', default='0.5')
    ap.add_argument('--betas', default='0')
    ap.add_argument('--match', default=None, help='matchability tensor from match.py')
    ap.add_argument('--eps', type=float, default=1e-4, help='storage threshold (eval may raise it)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    temps = [float(x) for x in args.temps.split(',')]
    alphas = [float(x) for x in args.alphas.split(',')] if args.mode == 'dual' else [1.0]
    betas = [float(x) for x in args.betas.split(',')] if args.mode == 'dual' else [0.0]
    configs = [tag(args.mode, *c) for c in itertools.product(temps, alphas, betas)]
    configs = list(dict.fromkeys(configs))

    if args.merge:
        for cfg in configs:
            parts = {}
            for k in range(args.nshards):
                parts.update(torch.load(f'{args.out}_{cfg}.shard{k}', weights_only=False))
            R = sorted(parts)
            assert R == list(range(len(R))), 'missing relations'
            torch.save({'crow': [parts[r][0] for r in R], 'col': [parts[r][1] for r in R],
                        'val': [parts[r][2] for r in R]}, f'{args.out}_{cfg}.pt')
            for k in range(args.nshards):
                os.remove(f'{args.out}_{cfg}.shard{k}')
            print('merged', cfg, 'nnz =', sum(parts[r][2].numel() for r in R), flush=True)
        raise SystemExit

    dev = 'cuda'
    n, nrel = read_stats(args.data)
    sc = make_scorer(args.ckpt, dev)
    assert sc.n == n
    num = torch.ones(nrel, n, device=dev)
    raw = read_train(args.data)
    tails = {r: ([], []) for r in range(nrel)}
    for (h, r), ts in raw.items():
        num[r, h] = len(ts)
        tails[r][0].extend([h] * len(ts))
        tails[r][1].extend(ts)
    tails = {r: (torch.tensor(a, dtype=torch.long, device=dev), torch.tensor(b, dtype=torch.long, device=dev))
             for r, (a, b) in tails.items()}
    match = None
    if args.match:
        match = torch.load(args.match, weights_only=False).float().to(dev)  # (nrel, n)
    out = {cfg: {} for cfg in configs}
    pairs = list(range(0, nrel, 2))
    for k in pairs[args.shard::args.nshards]:
        r, ri = k, k + 1
        Sf, Sb = all_scores(sc, r), all_scores(sc, ri)
        for T in temps:
            Pf = calibrated(Sf, r, num, tails, T)
            Pb = calibrated(Sb, ri, num, tails, T)
            if args.mode == 'forward':
                out[tag('forward', T, 0, 0)][r] = to_csr(Pf, args.eps)
                out[tag('forward', T, 0, 0)][ri] = to_csr(Pb, args.eps)
                continue
            for a, b in itertools.product(alphas, betas):
                for (x, y, P, Q) in [(r, ri, Pf, Pb), (ri, r, Pb, Pf)]:
                    A = P.pow(a) * Q.t().pow(1 - a)
                    if b > 0 and match is not None:
                        A *= (match[x].unsqueeze(1) * match[y].unsqueeze(0)).pow(b)
                    hh, tt = tails[x]
                    A[hh, tt] = 1.0
                    out[tag('dual', T, a, b)][x] = to_csr(A, args.eps)
                    del A
            del Pf, Pb
        del Sf, Sb
        print('pair', k, 'done', flush=True)
    for cfg in configs:
        torch.save(out[cfg], f'{args.out}_{cfg}.shard{args.shard}')
    print('shard done', args.shard, flush=True)
