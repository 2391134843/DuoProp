"""Embedding kNN graph of Prop for a link predictor (same construction as duoprop.collective_eval.emb_knn, which looks the
checkpoint up in runs/jobs.txt): cosine of standardised entity features, 100 neighbours, self excluded.
python tools/emb_knn.py DATA CKPT TAG  ->  cache/art/knn_<ds>_emb_<TAG>.pt"""
import os
import sys

import torch

from duoprop.collective_eval import DS
from duoprop.scorer import make_scorer

data, ckpt, tag = sys.argv[1:4]
X = make_scorer(ckpt, 'cuda').features().float()
X = (X - X.mean(0)) / (X.std(0) + 1e-6)
X = X / X.norm(dim=1, keepdim=True)
n, k = X.shape[0], 100
nbr, val = torch.empty(n, k, dtype=torch.long, device='cuda'), torch.empty(n, k, device='cuda')
for s0 in range(0, n, 4096):
    G = X[s0:s0 + 4096] @ X.t()
    G[torch.arange(G.shape[0]), torch.arange(s0, s0 + G.shape[0])] = -1
    v, i = G.topk(k, dim=1)
    nbr[s0:s0 + len(v)], val[s0:s0 + len(v)] = i, v.clamp(min=0)
torch.save((nbr.cpu(), val.cpu()), f'cache/art/knn_{DS[os.path.basename(data.rstrip("/"))]}_emb_{tag}.pt')
print('PART DONE', flush=True)
