"""Run the official FIT solver (Yin et al., ICLR 2024; code and EFO-1 data in $FIT_DIR):
Symbloic_FIT.solve_EFO1_new, the solver FIT uses with stored matrices, on the real EFO-1 benchmark with our truth matrices:
QTO's head-side matrices or Duo matrices (matchability, irreflexive prior), optionally with Prop after every projection.
FIT itself is unchanged except for the optional Prop hook: with the product t-norm, max_l M[l,a] x_l y_a = y_a max_l M[l,a] x_l,
so the projection max_l M[l,a] x_l is formed first, propagated, and then multiplied by the candidate vector; Prop is applied
only to projections without negated atoms (as in our reasoner, where negation acts on a projected set).  Standard protocol
(training edges = 1).  Ranking metrics as in FIT (QG_FIT.ranking2metrics).
Output: runs/paper/fit_<name>.json   (per type MRR / H@1 / H@3 / H@10 and number of queries)
FIT_DIR=<FIT> PYTHONPATH=.:<FIT> python tools/fit_run.py NAME CACHE EPS MATCH RB IRR BETA KNN_TAG SPLIT TYPES LIMIT
FIT needs pandas, python-constraint and pyyaml.  One change to FIT: the import of subgraph_matching in fof.py (line 50)
is removed; it is not defined in knowledge_graph.py and is used only for answer sampling, not by solve_EFO1."""
import json
import os
import sys
import time
from collections import defaultdict

import torch

import Symbloic_FIT as F  # noqa: E402
from src.utils.data import QueryAnsweringSeqDataLoader_v2  # noqa: E402

from duoprop.collective_eval import get_knn, irreflexive  # noqa: E402
from duoprop.reason import load_cache  # noqa: E402

FIT = os.environ.get('FIT_DIR', 'third_party/FIT')
name, cache, eps, match, rb, irr, beta, ktag, split, types, limit = sys.argv[1:12]
rb, beta, limit = float(rb), float(beta), int(limit)
data = 'data/FB15k-237-betae'
dev = torch.device('cuda:0')
n = 14505
m = torch.load(match, weights_only=False) if rb > 0 else None
adj = load_cache(cache, n, 'cuda', None if eps == 'none' else float(eps), m, rb)
if irr == '1':
    irreflexive(adj, data)
mats = []
for M in adj:
    rows = torch.repeat_interleave(torch.arange(n, device=M.col.device), M.crow[1:] - M.crow[:-1])
    mats.append(torch.sparse_coo_tensor(torch.stack([rows, M.col.long()]), M.val, (n, n)).coalesce())
del adj
torch.cuda.empty_cache()
nbr, val = get_knn(data, 'rel+emb', 20, ktag) if beta > 0 else (None, None)


def prop(x):
    if beta <= 0:
        return x
    S = (val * x[nbr]).sum(1) / val.sum(1).clamp(min=1e-12)
    return x + beta * S


def has_neg(a, b, neg_g):
    return bool(neg_g.ht2r[(a, b)] or neg_g.ht2r[(b, a)])


@torch.no_grad()
def existential_update(leaf_node, adjacency_node, sub_graph, neg_sub_graph, r_matrix_list, leaf_candidates, adj_candidates,
                       conj_tnorm, exist_tnorm):
    assert conj_tnorm == 'product' and exist_tnorm == 'Godel'
    M = F.construct_matrix_list(leaf_node, adjacency_node, sub_graph, neg_sub_graph, r_matrix_list, conj_tnorm)
    proj = torch.amax(M * leaf_candidates.unsqueeze(-1), dim=-2).squeeze()
    del M
    if not has_neg(leaf_node, adjacency_node, neg_sub_graph):
        proj = prop(proj)
    return proj * adj_candidates


@torch.no_grad()
def extend_ans(ans_node, sub_ans_node, sub_graph, neg_sub_graph, relation_matrix, leaf_candidate, sub_ans, conj_tnorm,
               exist_tnorm):
    assert conj_tnorm == 'product' and exist_tnorm == 'Godel'
    M = F.construct_matrix_list(sub_ans_node, ans_node, sub_graph, neg_sub_graph, relation_matrix, conj_tnorm)
    proj = torch.amax(M * sub_ans.unsqueeze(-1), dim=-2).squeeze()
    del M
    if not has_neg(sub_ans_node, ans_node, neg_sub_graph):
        proj = prop(proj)
    return leaf_candidate * proj


F.existential_update, F.extend_ans = existential_update, extend_ans
out = {}
for t in types.split(','):
    path = f'{FIT}/data/FB15k-237-EFO1/{split}_type{int(t):04d}_real_EFO1_qaa.json'
    loader = QueryAnsweringSeqDataLoader_v2(path, size_limit=limit if limit > 0 else None, batch_size=20, shuffle=False,
                                            num_workers=0)
    metrics, t0 = defaultdict(float), time.time()
    for fof in loader.get_fof_list_no_shuffle():
        ans = torch.stack([F.solve_EFO1_new(fof, mats, 'product', 'Godel', i, dev, 10)
                           for i in range(len(fof.easy_answer_list))])
        argsort = torch.argsort(ans, dim=1, descending=True)
        ranking = torch.empty(ans.shape, device=dev).scatter_(
            1, argsort, torch.arange(n, dtype=torch.float, device=dev).repeat(argsort.shape[0], 1))
        for i in range(ans.shape[0]):  # filtered ranks of the hard answers, as in QG_FIT.ranking2metrics
            easy = [x[0] for x in fof.easy_answer_list[i]['f1']]
            hard = [x[0] for x in fof.hard_answer_list[i]['f1']]
            cur, idx = torch.sort(ranking[i, easy + hard])
            cur = (cur - torch.arange(len(easy) + len(hard), dtype=torch.float, device=dev) + 1)[idx >= len(easy)]
            metrics['MRR'] += (1 / cur).mean().item()
            for k in (1, 3, 10):
                metrics[f'HITS{k}'] += (cur <= k).float().mean().item()
            metrics['num_queries'] += 1
    q = metrics['num_queries']
    out[f'type{int(t):04d}'] = dict(n=q, **{k: 100 * metrics[k] / q for k in ('MRR', 'HITS1', 'HITS3', 'HITS10')})
    print(name, t, json.dumps(out[f'type{int(t):04d}']), f'{time.time() - t0:.0f}s', flush=True)
    json.dump(dict(name=name, cache=cache, eps=eps, rb=rb, irr=irr, beta=beta, split=split, res=out),
              open(f'runs/paper/fit_{name}.json', 'w'), indent=1)
print('PART DONE', flush=True)
