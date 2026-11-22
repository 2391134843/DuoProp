"""Matchability of queries: m(h, r) = P(query (h, r) has at least one answer).

A per-relation logistic probe on the frozen features of the head entity (SpherE: modulus, phase as
cos/sin, radius; the other link predictors: the entity embedding), standardised per dimension.  Labels come from the training graph only: a (h, r) row
with an observed tail is positive; rows without one are unlabeled and down-weighted by
--neg_weight (positive-unlabeled setting: some empty rows are merely incomplete).
"""
import argparse
import collections
import os

import torch
import torch.nn.functional as F

from duoprop.scorer import make_scorer, read_stats

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--neg_weight', type=float, default=0.5)
    ap.add_argument('--l2', type=float, default=1e-4)
    ap.add_argument('--steps', type=int, default=800)
    args = ap.parse_args()
    dev = 'cuda'
    n, nrel = read_stats(args.data)
    X = make_scorer(args.ckpt, dev).features()
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Y = torch.zeros(nrel, n, device=dev)
    for line in open(os.path.join(args.data, 'train.txt')):
        h, r, t = map(int, line.split())
        Y[r, h] = 1.0
    held = collections.defaultdict(set)
    for fn in ['valid.txt']:
        for line in open(os.path.join(args.data, fn)):
            h, r, t = map(int, line.split())
            held[r].add(h)
    W = torch.zeros(nrel, X.shape[1], device=dev, requires_grad=True)
    prior = Y.mean(1).clamp(1e-4, 1 - 1e-4)
    b = torch.log(prior / (1 - prior)).clone().requires_grad_(True)
    opt = torch.optim.Adam([W, b], lr=0.01)
    wt = Y + args.neg_weight * (1 - Y)
    for it in range(args.steps):
        logits = W @ X.t() + b.unsqueeze(1)
        loss = (F.binary_cross_entropy_with_logits(logits, Y, weight=wt, reduction='sum') / wt.sum()
                + args.l2 * W.pow(2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        m = torch.sigmoid(W @ X.t() + b.unsqueeze(1))
        # diagnostic on validation: rows empty in train that gain an answer in valid vs. other empty rows
        newpos = torch.zeros_like(Y, dtype=torch.bool)
        for r, hs in held.items():
            newpos[r, torch.tensor(sorted(hs), device=dev)] = True
        newpos &= Y == 0
        empty = (Y == 0) & ~newpos
        print(f'loss={loss.item():.4f}  mean m: observed rows={m[Y == 1].mean():.3f}  '
              f'empty-in-train-but-answer-in-valid={m[newpos].mean():.3f}  other empty rows={m[empty].mean():.4f}')
    torch.save(m.half().cpu(), args.out)
