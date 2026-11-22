"""Blocked builder for large entity sets (NELL995): same truth matrices as build.py, but the
n x n score matrices are staged in host memory and the dual combination is done per head block.

A_r[h,t]  = P_r[h,t]^a  * P_r'[t,h]^(1-a) * (m_r[h] m_r'[t])^b
A_r'[t,h] = P_r'[t,h]^a * P_r[h,t]^(1-a)  * (m_r'[t] m_r[h])^b   (built from the same blocks)
"""
import argparse
import itertools

import torch

from duoprop.build import tag
from duoprop.scorer import make_scorer, read_stats, read_train


def stage_scores(sc, r, bs=256):
    n = sc.n
    S = torch.empty(n, n, dtype=torch.float32)
    for s in range(0, n, bs):
        hs = torch.arange(s, min(n, s + bs), device=sc.dev)
        S[s:s + len(hs)].copy_(sc.scores(hs, r).cpu())
    return S


def lse_rows(S, T, dev, bs=4096):
    return torch.cat([torch.logsumexp(S[s:s + bs].to(dev) / T, dim=1) for s in range(0, S.shape[0], bs)])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--temps', default='1.0')
    ap.add_argument('--alphas', default='0.5')
    ap.add_argument('--betas', default='0')
    ap.add_argument('--match', default=None)
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--block', type=int, default=2048)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()
    temps = [float(x) for x in args.temps.split(',')]
    alphas = [float(x) for x in args.alphas.split(',')]
    betas = [float(x) for x in args.betas.split(',')]
    dev = 'cuda'
    n, nrel = read_stats(args.data)
    sc = make_scorer(args.ckpt, dev)
    num = torch.ones(nrel, n, device=dev)
    raw = read_train(args.data)
    train = {r: ([], []) for r in range(nrel)}
    for (h, r), ts in raw.items():
        num[r, h] = len(ts)
        train[r][0].extend([h] * len(ts))
        train[r][1].extend(ts)
    train = {r: (torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)) for r, (a, b) in train.items()}
    match = torch.load(args.match, weights_only=False).float().to(dev) if args.match else None
    combos = list(itertools.product(temps, alphas, betas))
    out = {tag('dual', *c): {} for c in combos}
    pairs = list(range(0, nrel, 2))
    for k in pairs[args.shard::args.nshards]:
        r, ri = k, k + 1
        Sf, Sb = stage_scores(sc, r), stage_scores(sc, ri)
        for T in temps:
            lf, lb = lse_rows(Sf, T, dev), lse_rows(Sb, T, dev)
            # COO pieces per config: rows of A_r (h, t) and of A_r' (t, h)
            acc = {c: ([], [], [], [], [], []) for c in combos if c[0] == T}
            for s in range(0, n, args.block):
                hb = torch.arange(s, min(n, s + args.block))
                Pf = (torch.exp(Sf[s:s + len(hb)].to(dev) / T - lf[s:s + len(hb)].unsqueeze(1))
                      * num[r, s:s + len(hb)].unsqueeze(1)).clamp_(max=1 - 1e-4)          # (Bh, n) = P_r[h, :]
                Pb = (torch.exp(Sb[:, s:s + len(hb)].to(dev) / T - lb.unsqueeze(1))
                      * num[ri].unsqueeze(1)).clamp_(max=1 - 1e-4).t().contiguous()     # (Bh, n) = P_r'[:, h]^T
                hh, tt = train[r]
                m = (hh >= s) & (hh < s + len(hb))
                Pf[(hh[m] - s).to(dev), tt[m].to(dev)] = 1.0
                Pb[(hh[m] - s).to(dev), tt[m].to(dev)] = 1.0   # inverse edge (t, r', h) is also observed
                for (T_, a, b) in acc:
                    A1 = Pf.pow(a) * Pb.pow(1 - a)        # A_r[h, t]
                    A2 = Pb.pow(a) * Pf.pow(1 - a)        # A_r'[t, h], stored transposed (rows = h)
                    if b > 0 and match is not None:
                        mm = (match[r, s:s + len(hb)].unsqueeze(1) * match[ri].unsqueeze(0)).pow(b)
                        A1 *= mm
                        A2 *= mm
                    A1[(hh[m] - s).to(dev), tt[m].to(dev)] = 1.0
                    A2[(hh[m] - s).to(dev), tt[m].to(dev)] = 1.0
                    for A, (R0, C0, V0) in [(A1, (0, 1, 2)), (A2, (3, 4, 5))]:
                        i, j = torch.nonzero(A >= args.eps, as_tuple=True)
                        L = acc[(T_, a, b)]
                        L[R0].append((i + s).cpu())
                        L[C0].append(j.to(torch.int32).cpu())
                        L[V0].append(A[i, j].float().cpu())
                    del A1, A2
                del Pf, Pb
            for c, L in acc.items():
                for x, (rows, cols, vals) in [(r, (L[0], L[1], L[2])), (ri, (L[4], L[3], L[5]))]:
                    # A_r' entries were produced transposed as (h, t); L[4] holds t (its rows), L[3] holds h
                    rows, cols, vals = torch.cat(rows).long(), torch.cat(cols).long(), torch.cat(vals)
                    order = torch.argsort(rows * n + cols)
                    rows, cols, vals = rows[order], cols[order], vals[order]
                    crow = torch.zeros(n + 1, dtype=torch.long)
                    crow[1:] = torch.bincount(rows, minlength=n).cumsum(0)
                    out[tag('dual', *c)][x] = (crow, cols.to(torch.int32), vals)
        del Sf, Sb
        print('pair', k, 'done', flush=True)
    for cfg in out:
        torch.save(out[cfg], f'{args.out}_{cfg}.shard{args.shard}')
    print('shard done', args.shard, flush=True)
