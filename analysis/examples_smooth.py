"""Motivation statistics + named examples for Prop (neighbourhood propagation), FB15k-237 test split.
(1) how often a hard answer of an intersection query is a 'hole' (truth < 1e-3) in at least one branch, and
    how many such answers smoothing brings into the top 10;
(2) queries whose hard answer jumps into the top 3, with the branch truths before / after smoothing and the
    answer's structural twins (relation-typed profile neighbours) that the branch did retrieve.
Writes runs/examples_smooth.md."""
import json
import os
import pickle
import sys

import torch

from duoprop.collective_eval import CollReasoner, get_knn
from duoprop.reason import load_cache, load_split

data, n = 'data/FB15k-237-betae', 14505
KIND = sys.argv[1] if len(sys.argv) > 1 else 'rel'
BETA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
sel = json.load(open('runs/results/FINAL_SELECTION.json'))['FB15k-237 (paper ckpt, d=256)']['best_valid']
adj = load_cache(sel['cache'], n, 'cuda', None, torch.load('cache/art/match_fb237_hiT.pt', weights_only=False),
                 sel['row_beta'])
nbr, val = get_knn(data, KIND, 20, 'fb237_hiT')
R0 = CollReasoner(adj, n, 'cuda', nbr=nbr, val=val, beta=0.0)
R1 = CollReasoner(adj, n, 'cuda', nbr=nbr, val=val, beta=BETA)
id2ent = pickle.load(open(data + '/id2ent.pkl', 'rb'))
id2rel = pickle.load(open(data + '/id2rel.pkl', 'rb'))
mid = dict(l.rstrip('\n').split('\t') for l in open('aux_names/FB15k-237_entity2text.txt'))
name = lambda i: mid.get(id2ent[i], id2ent[i])
q, e, h = load_split(data, 'test')


def rank_of(score, a, answers):
    ranking = torch.argsort(torch.argsort(score, descending=True))
    ra = ranking[a]
    return int(ra) - sum(int(ranking[x] < ra) for x in answers if x != a) + 1


S = {'2i': (('e', ('r',)), ('e', ('r',))), '3i': (('e', ('r',)), ('e', ('r',)), ('e', ('r',))),
     'pi': (('e', ('r', 'r')), ('e', ('r',)))}
out = ['# Prop (neighbourhood propagation): statistics and examples (FB15k-237 test split, generated)', '',
       f'Setting: Duo atoms of the main table, one propagation step after every projection, beta={BETA}, '
       f'neighbour graph {KIND} (k=20 per view).', '',
       '## 1. Holes in intersection queries', '',
       '| Type | Hard answers | At least one branch with truth < 1e-3 (hole) | Holes in top 10: before | after |',
       '|---|---|---|---|---|']
examples = []
cases, holes = [], []  # structured copies for plotting (runs/figdata/prop_*.json), up to 6 cases per kind and type
ncase = lambda kind, t: sum(c['kind'] == kind and c['type'] == t for c in cases)
branches = lambda qq, s: [dict(anchor=name(qq[i][0]), relations=[id2rel[r] for r in qq[i][1]]) for i in range(len(s))]
for sname, s in S.items():
    tot = hole = top_b = top_s = 0
    for qq in sorted(q[s])[:2000]:
        ans = list(e[qq]) + list(h[qq])
        T0, T1 = R0.execute(qq, s), R1.execute(qq, s)
        br0 = [R0.execute(qq[i], s[i]) for i in range(len(s))]
        br1 = [R1.execute(qq[i], s[i]) for i in range(len(s))]
        for a in h[qq]:
            tot += 1
            if min(b[a].item() for b in br0) >= 1e-3:
                rb, rs = rank_of(T0, a, ans), rank_of(T1, a, ans)
                md = sum(x[0] == 'B' + sname for x in examples) < 2
                if rb >= 10 and rs <= 2 and (md or ncase('rerank', sname) < 6):
                    allans = torch.zeros(n, dtype=torch.bool, device='cuda')
                    allans[ans] = True
                    wrong = [int(x) for x in torch.argsort(T0, descending=True)[:50] if not allans[x]][:3]
                    sup = lambda b, x: ((val[x] * b[nbr[x]]).sum() / val[x].sum().clamp(min=1e-12)).item()
                    cases.append(dict(kind='rerank', type=sname, answer=name(a), rank_before=rb, rank_after=rs,
                                      branches=branches(qq, s), candidates=[dict(
                                          entity=name(x), role='answer' if x == a else 'wrong candidate',
                                          branch_truth=[round(br0[i][x].item(), 4) for i in range(len(s))],
                                          branch_support=[round(sup(br0[i], x), 4) for i in range(len(s))],
                                          branch_truth_after=[round(br1[i][x].item(), 4) for i in range(len(s))],
                                          score_before=round(T0[x].item(), 5), score_after=round(T1[x].item(), 5))
                                          for x in [a] + wrong]))
                    if not md:
                        continue
                    lines = [f'**{sname} (ranking corrected)** hard answer **{name(a)}**: rank {rb} before, {rs} after', '',
                             'Branches: ' + '; '.join(' → '.join([name(qq[i][0])] + [id2rel[r] for r in qq[i][1]])
                                               for i in range(len(s))), '',
                             '| Entity | Role | ' + ' | '.join(f'Branch {i + 1} truth (neighbour support S)' for i in range(len(s)))
                             + ' | Score before | Score after |', '|' + '---|' * (len(s) + 4)]
                    for x, role in [(a, 'answer')] + [(w, 'wrong candidate') for w in wrong]:
                        lines.append(f'| {name(x)} | {role} | ' + ' | '.join(
                            f'{br0[i][x].item():.3f} ({sup(br0[i], x):.3f})' for i in range(len(s)))
                            + f' | {T0[x].item():.4f} | {T1[x].item():.4f} |')
                    examples.append(('B' + sname, lines + ['']))
                continue
            hole += 1
            rb, rs = rank_of(T0, a, ans), rank_of(T1, a, ans)
            top_b += rb <= 10
            top_s += rs <= 10
            md = sum(x[0] == sname for x in examples) < 2
            if rb >= 20 and rs <= 3 and (md or ncase('hole', sname) < 6):
                k = min(range(len(s)), key=lambda i: br0[i][a].item())  # the branch with the hole
                views = KIND.split('+')
                tw = {vw: sorted([(int(j), float(v)) for j, v in zip(nbr[a][20 * i:20 * i + 20], val[a][20 * i:20 * i + 20])],
                                 key=lambda x: -x[1])[:8] for i, vw in enumerate(views)}
                cases.append(dict(kind='hole', type=sname, answer=name(a), rank_before=rb, rank_after=rs, hole_branch=k,
                                  branches=[dict(b, truth_before=round(br0[i][a].item(), 4), truth_after=round(br1[i][a].item(), 4))
                                            for i, b in enumerate(branches(qq, s))],
                                  neighbours={vw: [dict(entity=name(j), similarity=round(v, 3),
                                                        truth_in_hole_branch=round(br0[k][j].item(), 4)) for j, v in lst]
                                              for vw, lst in tw.items()}))
                if not md:
                    continue
                tw = {vw: lst[:4] for vw, lst in tw.items()}
                lines = [f'**{sname}** hard answer **{name(a)}**: rank {rb} before, {rs} after', '',
                         '| Branch | Truth before | Truth after |', '|---|---|---|']
                for i in range(len(s)):
                    desc = ' → '.join([name(qq[i][0])] + [id2rel[r] for r in qq[i][1]])
                    lines.append(f'| {desc} | {br0[i][a].item():.4f} | {br1[i][a].item():.4f} |')
                label = {'rel': 'structural neighbours (shared relation-neighbour profile)',
                         'emb': 'embedding neighbours (cosine of the link-predictor representation)',
                         'ent': 'untyped-neighbour neighbours', 'type': 'relation-signature neighbours'}
                for vw, lst in tw.items():
                    lines += ['', f'In the branch with the lowest truth, the {label[vw]} of the answer and their truth there:', '']
                    lines += [f'- {name(j)} (similarity {v:.2f}): {br0[k][j].item():.3f}' for j, v in lst]
                examples.append((sname, lines + ['']))
    out.append(f'| {sname} | {tot} | {hole} ({100 * hole / max(tot, 1):.1f}%) | {top_b} | {top_s} |')
    holes.append(dict(type=sname, n_hard_answers=tot, n_holes=hole, holes_in_top10_before=top_b, holes_in_top10_after=top_s))
out += ['', '## 2. Examples', '']
for _, lines in examples:
    out += lines
open('runs/examples_smooth.md', 'w').write('\n'.join(out) + '\n')
os.makedirs('runs/figdata', exist_ok=True)
setting = f'FB15k-237 test, paper checkpoint d=256, Duo atoms of the main table, beta={BETA}, kNN graph {KIND} (k=20 per view)'
json.dump(dict(setting=setting, cases=cases), open('runs/figdata/prop_cases.json', 'w'), indent=1, ensure_ascii=False)
json.dump(dict(setting=setting, holes=holes), open('runs/figdata/prop_holes.json', 'w'), indent=1, ensure_ascii=False)
print('\n'.join(out))
