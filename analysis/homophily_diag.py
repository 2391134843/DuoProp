"""Which neighbourhood carries complementary evidence?  For 1p validation queries, among unobserved
candidates in the same truth bin, AUC of the kNN support S(T) for separating hard answers from wrong
candidates, for kNN graphs built from: relation-typed structural profile (rel), untyped neighbour set (ent),
relation signature (type), and the backbone's own embedding (emb, cosine of standardised features).
Usage: python analysis/homophily_diag.py <data> <name> <tag> <ckpt>"""
import json
import sys

import torch

from duoprop.collective_eval import CollReasoner, get_knn
from duoprop.reason import load_cache, load_split
from duoprop.scorer import make_scorer

if __name__ == '__main__':
    data, name, tag, ckpt = sys.argv[1:5]
    n = int(open(data + '/stats.txt').read().split('\n')[0].split(' ')[-1])
    sel = json.load(open('runs/results/FINAL_SELECTION.json'))
    b = next(v['best_valid'] for v in sel.values() if v['best_valid'].get('tag') == tag)
    adj = load_cache(b['cache'], n, 'cuda', None if b['eps'] == 'none' else float(b['eps']),
                     torch.load(f'cache/art/match_{tag}.pt', weights_only=False), b['row_beta'])
    graphs = {kind: tuple(get_knn(data, kind, 20)) for kind in ['rel', 'ent', 'type']}
    X = make_scorer(ckpt, 'cuda').features().float()
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = X / X.norm(dim=1, keepdim=True)
    nb, nv = torch.empty(n, 100, dtype=torch.long, device='cuda'), torch.empty(n, 100, device='cuda')
    for s0 in range(0, n, 4096):
        G = X[s0:s0 + 4096] @ X.t()
        G[torch.arange(G.shape[0]), torch.arange(s0, s0 + G.shape[0])] = -1
        v, i = G.topk(100, dim=1)
        nb[s0:s0 + len(v)], nv[s0:s0 + len(v)] = i, v.clamp(min=0)
    torch.save((nb.cpu(), nv.cpu()), f'cache/art/knn_{name}_emb.pt')  # backbone-specific (this tag only)
    nb, nv = nb[:, :20].contiguous(), nv[:, :20].contiguous()
    graphs['emb'] = (nb, nv)
    R = CollReasoner(adj, n, 'cuda', nbr=nb, val=nv, beta=0.0)
    q, e, h = load_split(data, 'valid')
    s1 = ('e', ('r',))
    bins = [(1e-3, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 0.999)]
    stat = {(g, bb): [[], []] for g in graphs for bb in bins}
    for qq in sorted(q[s1])[:3000]:
        T = R.execute(qq, s1)
        hard = torch.zeros(n, dtype=torch.bool, device='cuda')
        hard[list(h[qq])] = True
        easy = torch.zeros(n, dtype=torch.bool, device='cuda')
        easy[list(e[qq])] = True
        for g, (nbr, val) in graphs.items():
            S = (val * T[nbr]).sum(1) / val.sum(1).clamp(min=1e-12)
            for lo, hi in bins:
                m = (T >= lo) & (T < hi) & ~easy
                stat[(g, (lo, hi))][0].append(S[m & hard])
                stat[(g, (lo, hi))][1].append(S[m & ~hard])
    lines = [f'## Neighbour sources ({name}, validation 1p, 3000 queries): AUC of the neighbour support S for separating '
             'hard answers from wrong candidates within a truth bin', '',
             '| Truth bin | Hard answers | Relational profile rel | Untyped neighbours ent | Relation signature type '
             '| Link-predictor embedding emb |', '|---|---|---|---|---|---|']
    torch.manual_seed(0)
    for bb in bins:
        row = [f'[{bb[0]}, {bb[1]})']
        for g in ['rel', 'ent', 'type', 'emb']:
            pos, neg = (torch.cat(x) for x in stat[(g, bb)])
            if g == 'rel':
                row.append(str(len(pos)))
            sub = neg[torch.randperm(len(neg), device='cuda')[:20000]]
            auc = ((pos.unsqueeze(1) > sub.unsqueeze(0)).float().mean()
                   + 0.5 * (pos.unsqueeze(1) == sub.unsqueeze(0)).float().mean()).item()
            row.append(f'{auc:.3f}')
        lines.append('| ' + ' | '.join(row) + ' |')
    print('\n'.join(lines), flush=True)
    open(f'runs/homophily_{name}.md', 'w').write('\n'.join(lines) + '\n')
