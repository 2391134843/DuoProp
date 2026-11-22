"""(a) Homophily diagnostic: on 1p queries, among unobserved candidates in the same truth bin, do hard answers
    have more support from their structural twins (kNN mean of the truth vector) than wrong candidates?
    Reports per truth bin the mean support of hard answers vs. non-answers and the AUC of support.
(b) Form of the smoothing step (validation): add  T + b*S(T);  convex (1-b)*T + b*S(T);  clamp min(1, T + b*S(T)).
Usage: python analysis/homophily_probe.py <data> <name> <tag> <beta> [limit]"""
import json
import sys

import torch

from duoprop.collective_eval import CollReasoner, get_knn
from duoprop.reason import evaluate, load_cache, load_split


class FormReasoner(CollReasoner):
    def __init__(self, *a, form='add', **kw):
        super().__init__(*a, **kw)
        self.form = form

    def project(self, T, r, mats=None):
        out = super(CollReasoner, self).project(T, r, mats)
        S = (self.val * out[self.nbr]).sum(1) / self.val.sum(1).clamp(min=1e-12)
        if self.form == 'add':
            return out + self.beta * S
        if self.form == 'convex':
            return (1 - self.beta) * out + self.beta * S
        return (out + self.beta * S).clamp(max=1.0)


if __name__ == '__main__':
    data, name, tag, beta = sys.argv[1:5]
    beta = float(beta)
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    n = int(open(data + '/stats.txt').read().split('\n')[0].split(' ')[-1])
    sel = json.load(open('runs/results/FINAL_SELECTION.json'))
    b = next(v['best_valid'] for v in sel.values() if v['best_valid'].get('tag') == tag)
    adj = load_cache(b['cache'], n, 'cuda', None if b['eps'] == 'none' else float(b['eps']),
                     torch.load(f'cache/art/match_{tag}.pt', weights_only=False), b['row_beta'])
    nbr, val = get_knn(data, 'rel', 20)
    q, e, h = load_split(data, 'valid')
    # (a)
    R = CollReasoner(adj, n, 'cuda', nbr=nbr, val=val, beta=0.0)
    s1 = ('e', ('r',))
    bins = [(1e-3, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 0.999)]
    stat = {bb: [[], []] for bb in bins}
    for qq in sorted(q[s1])[:3000]:
        T = R.execute(qq, s1)
        S = (val * T[nbr]).sum(1) / val.sum(1).clamp(min=1e-12)
        hard = torch.zeros(n, dtype=torch.bool, device='cuda')
        hard[list(h[qq])] = True
        easy = torch.zeros(n, dtype=torch.bool, device='cuda')
        easy[list(e[qq])] = True
        for lo, hi in bins:
            m = (T >= lo) & (T < hi) & ~easy
            stat[(lo, hi)][0].append(S[m & hard])
            stat[(lo, hi)][1].append(S[m & ~hard])
    lines = [f'## Homophily ({name}, validation 1p, 3000 queries; S = mean truth of the structural neighbours)', '',
             '| Truth bin | Hard answers | Non-answers | Mean S of hard answers | Mean S of non-answers | AUC(S) |',
             '|---|---|---|---|---|---|']
    for bb, (pos, neg) in stat.items():
        pos, neg = torch.cat(pos), torch.cat(neg)
        if len(pos) == 0 or len(neg) == 0:
            continue
        sub = neg[torch.randperm(len(neg), device='cuda')[:20000]]
        auc = ((pos.unsqueeze(1) > sub.unsqueeze(0)).float().mean()
               + 0.5 * (pos.unsqueeze(1) == sub.unsqueeze(0)).float().mean()).item()
        lines.append(f'| [{bb[0]}, {bb[1]}) | {len(pos)} | {len(neg)} | {pos.mean():.4f} | {neg.mean():.4f} | {auc:.3f} |')
    print('\n'.join(lines), flush=True)
    open(f'runs/homophily_{name}.md', 'w').write('\n'.join(lines) + '\n')
    # (b)
    for form in ['add', 'convex', 'clamp']:
        res = evaluate(FormReasoner(adj, n, 'cuda', nbr=nbr, val=val, beta=beta, form=form,
                                    neg_scale=b['neg_scale']), q, e, h, limit=limit)
        with open(f'runs/results/{name}_form_valid.jsonl', 'a') as f:
            f.write(json.dumps(dict(form=form, beta=beta, res=res)) + '\n')
        print(f"form={form:6s} beta={beta} AVG_p={res['AVG_p']:.2f} AVG_n={res['AVG_n']:.2f} | " +
              ' '.join(f"{t}:{res[t]['MRR']:.1f}" for t in res if t not in ('AVG_p', 'AVG_n')), flush=True)
