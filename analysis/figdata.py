"""Plot-ready data (csv / json) for the motivation and analysis figures.  FB15k-237, published SpherE checkpoint (d=256),
Duo setting of the main table (T=3, alpha=0.7, matchability rb=2).  Output: runs/figdata/.

  dual     Duo, two-sided calibration. Top-20 unobserved forward candidates of sampled queries (h, r, ?): head view P_r[h,t],
           tail view P_r'[t,h] (r' = inverse relation), bidirectional truth, label = hidden fact (valid / test edge).
           -> dual_candidates.csv, dual_cases.json, dual_summary.json
  match    Duo, matchability. Rows (h, r) without observed answers: matchability m(h, r), whether the row has a hidden
           answer, the truth mass the row spreads (original QTO T=1 / bidirectional / x m^rb); 2p queries rescued by
           matchability.  -> match_rows.csv, match_cases.json, match_summary.json
Entity names: aux_names/FB15k-237_entity2text.txt (entity2text.txt of FB15k-237 from https://github.com/yao8839836/kg-bert).
Set DATA, CKPT, CACHE, FWD1 and MATCH below to your own checkpoint and matrices.
Usage: python analysis/figdata.py [dual] [match]"""
import collections
import json
import os
import pickle
import random
import sys

import torch
from scipy.stats import rankdata

from duoprop.reason import Reasoner, load_cache, load_split
from duoprop.scorer import make_scorer, read_stats

DATA = 'data/FB15k-237-betae'
CKPT = 'logs/FB15k-237-betae/sphere/g-24.0-dim-256/2025.05.11-13.07.54/checkpoint'
CACHE = 'cache/mats/fb237_hiT_dual_T3.0_a0.7_b0.0.pt'
FWD1 = 'cache/mats/fb237_base_forward_T1.0.pt'
MATCH = 'cache/art/match_fb237_hiT.pt'
T, ALPHA, RB = 3.0, 0.7, 2.0
OUT = 'runs/figdata'

n, nrel = read_stats(DATA)
id2ent = pickle.load(open(f'{DATA}/id2ent.pkl', 'rb'))
id2rel = pickle.load(open(f'{DATA}/id2rel.pkl', 'rb'))
mid = dict(l.rstrip('\n').split('\t') for l in open('aux_names/FB15k-237_entity2text.txt'))
ent = lambda i: mid.get(id2ent[int(i)], id2ent[int(i)])
rel = lambda r: id2rel[int(r)]


def triples(split):
    out = collections.defaultdict(set)
    for line in open(f'{DATA}/{split}.txt'):
        h, r, t = map(int, line.split())
        out[(h, r)].add(t)
    return out


def hidden_facts(train):
    hidden = collections.defaultdict(set)
    for split in ('valid', 'test'):
        for key, ts in triples(split).items():
            hidden[key] |= ts - train.get(key, set())
    return hidden


def auc(pos, neg):
    ranks = rankdata(list(pos) + list(neg))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def write_csv(path, rows, cols):
    fmt = lambda v: ('"' + v.replace('"', "'") + '"' if isinstance(v, str) else f'{v:.6g}' if isinstance(v, float) else str(v))
    with open(path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(fmt(r[c]) for c in cols) + '\n')


def frank(rk, a, ans):
    """filtered rank of answer a (rk: rank of every entity, 0 = best; ans: tensor of all answers)"""
    return int(rk[a]) - int((rk[ans] < rk[a]).sum()) + 1


class Calib:
    """Forward calibration of build.py: min(1 - 1e-4, n_h * softmax_t(s(h,r,t) / T)), training edges = 1."""

    def __init__(self):
        self.sc = make_scorer(CKPT, 'cuda')
        self.train = triples('train')
        self.num = torch.ones(nrel, n, device='cuda')
        for (h, r), ts in self.train.items():
            self.num[r, h] = len(ts)

    def rows(self, r, hs, temps=(T,)):
        """one calibrated (len(hs), n) block per temperature; a single block if one temperature is given"""
        hs = torch.as_tensor(hs, device='cuda')
        S, out = self.sc.scores(hs, r), []
        for temp in temps:
            P = torch.softmax(S / temp, dim=1) * self.num[r, hs].unsqueeze(1)
            P.clamp_(max=1 - 1e-4)
            for i, h in enumerate(hs.tolist()):
                ts = self.train.get((h, r))
                if ts:
                    P[i, list(ts)] = 1.0
            out.append(P)
        return out if len(temps) > 1 else out[0]


def dual(seed=0, n_hidden_rows=1500, n_random_rows=1500, k=20):
    """T1 columns: the original QTO calibration (T=1), where the problem is visible; plain columns: our temperature."""
    C = Calib()
    M = torch.load(MATCH, weights_only=False).float()
    hidden = hidden_facts(C.train)
    rng = random.Random(seed)
    rows = rng.sample(sorted(x for x, v in hidden.items() if v), n_hidden_rows)
    rows += [(rng.randrange(n), rng.randrange(nrel)) for _ in range(n_random_rows)]
    by_rel = collections.defaultdict(list)
    for h, r in dict.fromkeys(rows):
        by_rel[r].append(h)
    cands, partners = [], {}  # partners[(r, t)]: top-3 heads of the tail view (T=1), i.e. who t thinks its r-partners are
    for r, hs in sorted(by_rel.items()):
        Pf1, Pf = C.rows(r, hs, (1.0, T))
        for i, h in enumerate(hs):
            if (h, r) in C.train:
                Pf[i, list(C.train[(h, r)])] = -1.0  # observed facts are 1 by construction: unobserved candidates only
        val, idx = Pf.topk(k, dim=1)
        val1 = Pf1.gather(1, idx)
        del Pf, Pf1
        need = collections.defaultdict(list)
        for i, h in enumerate(hs):
            for t in idx[i].tolist():
                need[t].append(h)
        U, pb, pb1 = sorted(need), {}, {}
        for s0 in range(0, len(U), 512):
            Pb1, Pb = C.rows(r ^ 1, U[s0:s0 + 512], (1.0, T))
            tv, ti = Pb1.topk(3, dim=1)
            for j, t in enumerate(U[s0:s0 + 512]):
                hh = torch.tensor(need[t], device='cuda')
                pb.update(zip(((h, t) for h in need[t]), Pb[j, hh].tolist()))
                pb1.update(zip(((h, t) for h in need[t]), Pb1[j, hh].tolist()))
                partners[(r, t)] = list(zip(ti[j].tolist(), tv[j].tolist()))
            del Pb, Pb1
        for i, h in enumerate(hs):
            observed, m = (h, r) in C.train, M[r, h].item()
            for v, v1, t in zip(val[i].tolist(), val1[i].tolist(), idx[i].tolist()):
                d = v ** ALPHA * pb[(h, t)] ** (1 - ALPHA)
                cands.append(dict(h=h, r=r, t=t, head=ent(h), relation=rel(r), tail=ent(t),
                                  label=int(t in hidden.get((h, r), ())), row_observed=int(observed),
                                  p_head_T1=v1, p_tail_T1=pb1[(h, t)], dual_T1=v1 ** ALPHA * pb1[(h, t)] ** (1 - ALPHA),
                                  p_head_view=v, p_tail_view=pb[(h, t)], dual=d, matchability=m,
                                  final=d if observed else d * m ** RB))
    cols = ['h', 'r', 't', 'head', 'relation', 'tail', 'label', 'row_observed', 'p_head_T1', 'p_tail_T1', 'dual_T1',
            'p_head_view', 'p_tail_view', 'dual', 'matchability', 'final']
    write_csv(f'{OUT}/dual_candidates.csv', cands, cols)
    summary = {'setting': f'FB15k-237, paper checkpoint d=256; *_T1 = original QTO calibration (T=1), plain = T={T}; '
                          f'alpha={ALPHA}, rb={RB}; top-{k} unobserved candidates of {n_hidden_rows} rows with a hidden '
                          f'(valid/test) answer + {n_random_rows} uniformly random rows (h, r)',
               'n_candidates': len(cands), 'n_hidden_facts': sum(c['label'] for c in cands)}
    scores = ('p_head_T1', 'dual_T1', 'p_head_view', 'dual', 'final')
    for name, sub in [('all', cands), ('rows_without_observed_answer', [c for c in cands if not c['row_observed']]),
                      ('rows_with_observed_answer', [c for c in cands if c['row_observed']])]:
        p, q = [c for c in sub if c['label']], [c for c in sub if not c['label']]
        if p and q:
            summary[name] = {f'AUC_{s}': round(auc([c[s] for c in p], [c[s] for c in q]), 4) for s in scores}
            summary[name].update(n_pos=len(p), n_neg=len(q))
    for hv, tv in [('p_head_T1', 'p_tail_T1'), ('p_head_view', 'p_tail_view')]:
        conf = [c for c in cands if c[hv] >= 0.5]
        cp, cn = [c for c in conf if c['label']], [c for c in conf if not c['label']]
        summary[f'{hv}>=0.5'] = {
            'n': len(conf), 'fraction_hidden_fact': round(len(cp) / max(len(conf), 1), 4),
            'fraction_in_rows_without_observed_answer': round(sum(1 - c['row_observed'] for c in conf) / max(len(conf), 1), 4),
            'non_facts_with_tail_view<0.01': round(sum(c[tv] < 0.01 for c in cn) / max(len(cn), 1), 4),
            'hidden_facts_with_tail_view<0.01': round(sum(c[tv] < 0.01 for c in cp) / max(len(cp), 1), 4)}
    per_row = collections.defaultdict(list)
    for c in cands:
        per_row[(c['h'], c['r'])].append(c)
    for s in scores:
        hits = [max(v, key=lambda c: c[s])['label'] for v in per_row.values() if any(c['label'] for c in v)]
        summary.setdefault('top1_of_row_is_hidden_fact', {})[s] = round(sum(hits) / max(len(hits), 1), 4)
    json.dump(summary, open(f'{OUT}/dual_summary.json', 'w'), indent=1, ensure_ascii=False)

    def case(c, tops):
        return dict({x: c[x] for x in ('head', 'relation', 'tail')}, hidden_fact=bool(c['label']),
                    row_has_observed_answer=bool(c['row_observed']),
                    **{x: round(c[x], 4) for x in cols[8:]},
                    tail_view_top_partners_T1=[dict(head=ent(j), p_tail_T1=round(x, 4)) for j, x in tops])

    def pick(cond, key, per_relation=True, limit=10):
        seen, out = set(), []
        for c in sorted(filter(cond, cands), key=key):
            if not per_relation or c['r'] not in seen:
                seen.add(c['r'])
                out.append(case(c, partners[(c['r'], c['t'])]))
            if len(out) == limit:
                break
        return out

    cases = {
        'T1_non_fact_confident_head_view_rejected_by_tail_view': pick(
            lambda c: not c['label'] and c['p_head_T1'] >= 0.7 and c['p_tail_T1'] < 0.01, lambda c: -c['p_head_T1']),
        'non_fact_confident_head_view_rejected_by_tail_view': pick(
            lambda c: not c['label'] and c['p_head_view'] >= 0.5 and c['p_tail_view'] < 0.05, lambda c: -c['p_head_view']),
        'hidden_fact_both_views_agree': pick(
            lambda c: c['label'] and c['p_head_view'] >= 0.3 and c['p_tail_view'] >= 0.3, lambda c: -c['dual']),
        'hidden_fact_weak_head_view_strong_tail_view': pick(
            lambda c: c['label'] and c['p_head_view'] < 0.3 and c['p_tail_view'] >= 0.5, lambda c: -c['p_tail_view'])}
    name2ids = collections.defaultdict(list)  # the cases cited in the paper (ambiguous names -> most confident id)
    for i in range(n):
        name2ids[ent(i)].append(i)
    rpd = [r for r in range(nrel) if rel(r).startswith('+') and rel(r).endswith('place_of_death')]
    cited = []
    for hn, tn in [('The Runaways', 'Los Angeles'), ('Jesus Christ Superstar', 'New York City'),
                   ('Geraldine Chaplin', 'Santa Monica')]:
        best = None
        for r in rpd:
            for h in name2ids.get(hn, []):
                for t in name2ids.get(tn, []):
                    f1, f3 = (P[0] for P in C.rows(r, [h], (1.0, T)))
                    b1, b3 = (P[0] for P in C.rows(r ^ 1, [t], (1.0, T)))
                    c = dict(h=h, r=r, t=t, head=hn, relation=rel(r), tail=tn, label=int(t in hidden.get((h, r), ())),
                             row_observed=int((h, r) in C.train), p_head_T1=f1[t].item(), p_tail_T1=b1[h].item(),
                             p_head_view=f3[t].item(), p_tail_view=b3[h].item(), matchability=M[r, h].item())
                    c['dual_T1'] = c['p_head_T1'] ** ALPHA * c['p_tail_T1'] ** (1 - ALPHA)
                    c['dual'] = c['p_head_view'] ** ALPHA * c['p_tail_view'] ** (1 - ALPHA)
                    c['final'] = c['dual'] if c['row_observed'] else c['dual'] * c['matchability'] ** RB
                    tv, ti = b1.topk(3)
                    if best is None or c['p_head_T1'] > best[0]['p_head_T1']:
                        best = (c, list(zip(ti.tolist(), tv.tolist())))
        if best:
            cited.append(case(*best))
    cases['cited_in_notes'] = cited
    json.dump(cases, open(f'{OUT}/dual_cases.json', 'w'), indent=1, ensure_ascii=False)
    print('dual:', json.dumps(summary, ensure_ascii=False), flush=True)


def match(seed=0, n_random=20000, n_cases=6):
    train, M = triples('train'), torch.load(MATCH, weights_only=False).float()
    hidden = hidden_facts(train)
    mass = {}
    for col, path in [('mass_original_qto_T1', FWD1), ('mass_bidirectional', CACHE)]:
        if os.path.exists(path):
            d = torch.load(path, map_location='cpu', weights_only=False)
            mass[col] = [torch.zeros(n).index_add_(0, torch.repeat_interleave(torch.arange(n), c[1:] - c[:-1]), v.float())
                         for c, v in zip(d['crow'], d['val'])]
            del d
    rng = random.Random(seed)
    rows = set()
    while len(rows) < n_random:
        h, r = rng.randrange(n), rng.randrange(nrel)
        if (h, r) not in train:
            rows.add((h, r))
    rows |= {x for x, v in hidden.items() if v and x not in train}
    out = []
    for h, r in sorted(rows):
        m = M[r, h].item()
        rec = dict(h=h, r=r, head=ent(h), relation=rel(r), has_hidden_answer=int(bool(hidden.get((h, r)))),
                   n_hidden_answers=len(hidden.get((h, r), ())), matchability=m)
        for col, ms in mass.items():
            rec[col] = ms[r][h].item()
        if 'mass_bidirectional' in rec:
            rec['mass_after_matchability'] = rec['mass_bidirectional'] * m ** RB
        out.append(rec)
    cols = ['h', 'r', 'head', 'relation', 'has_hidden_answer', 'n_hidden_answers', 'matchability'] + list(mass) + \
           (['mass_after_matchability'] if 'mass_bidirectional' in mass else [])
    write_csv(f'{OUT}/match_rows.csv', out, cols)
    pos = [o for o in out if o['has_hidden_answer']]
    neg = [o for o in out if not o['has_hidden_answer']]
    summary = {'setting': f'rows (h, r) without any training tail: {n_random} uniformly sampled + every such row with a '
                          f'hidden (valid/test) answer; matchability {MATCH}, rb={RB}; truth mass = row sum of the stored '
                          f'cache (entries >= 1e-3)',
               'n_rows_with_hidden_answer': len(pos), 'n_rows_without_any_answer': len(neg),
               'AUC_matchability': round(auc([o['matchability'] for o in pos], [o['matchability'] for o in neg]), 4)}
    for c in [c for c in cols if c.startswith('mass')]:
        summary[f'mean_{c}'] = {'rows_with_hidden_answer': round(sum(o[c] for o in pos) / len(pos), 4),
                                'rows_without_any_answer': round(sum(o[c] for o in neg) / len(neg), 4)}
    json.dump(summary, open(f'{OUT}/match_summary.json', 'w'), indent=1, ensure_ascii=False)
    print('match:', json.dumps(summary, ensure_ascii=False), flush=True)

    # 2p queries whose hard answer is rescued by matchability, and why the original top answers were wrong
    A0 = load_cache(CACHE, n, 'cuda')
    A1 = load_cache(CACHE, n, 'cuda', None, M, RB)
    R0, R1 = Reasoner(A0, n, 'cuda'), Reasoner(A1, n, 'cuda')
    q, e, h = load_split(DATA, 'test')
    s2p, s1p = ('e', ('r', 'r')), ('e', ('r',))
    found = []
    for qq in sorted(q[s2p]):
        if (R0.execute((qq[0], (qq[1][0],)), s1p) >= 0.1).sum() > 20:
            continue  # hub anchors (e.g. 'Male') make the path explanation meaningless: keep readable queries only
        ans = torch.tensor(list(e[qq]) + list(h[qq]), device='cuda')
        S0, S1 = R0.execute(qq, s2p), R1.execute(qq, s2p)
        rk0 = torch.argsort(torch.argsort(S0, descending=True))
        rk1 = torch.argsort(torch.argsort(S1, descending=True))
        for a in h[qq]:
            b, f = frank(rk0, a, ans), frank(rk1, a, ans)
            if b >= 10 and f <= 3:
                found.append((b - f, qq, a, b, f))
    found.sort(key=lambda x: -x[0])
    cases, used = [], set()
    for _, qq, a, b, f in found:
        if qq in used:
            continue
        used.add(qq)
        anchor, (r1, r2) = qq[0], qq[1]
        ans = list(e[qq]) + list(h[qq])
        S0, S1 = R0.execute(qq, s2p), R1.execute(qq, s2p)
        x1 = R0.execute((anchor, (r1,)), s1p)
        allans = torch.zeros(n, dtype=torch.bool, device='cuda')
        allans[ans] = True
        wrong = [int(x) for x in torch.argsort(S0, descending=True)[:20] if not allans[x]][:3]

        def via(target):
            vs = torch.nonzero(x1 >= 1e-3).flatten()
            vs = vs[torch.argsort(x1[vs], descending=True)[:3000]]
            hop2 = A0[r2].dense_rows(vs)[:, target]
            j = int(torch.argmax(x1[vs] * hop2))
            v = int(vs[j])
            return dict(intermediate=ent(v), truth_hop1=round(x1[v].item(), 4), truth_hop2=round(hop2[j].item(), 4),
                        row_has_observed_answer=(v, r2) in train, matchability=round(M[r2, v].item(), 4))

        cases.append(dict(query=f'{ent(anchor)} --{rel(r1)}--> ? --{rel(r2)}--> ?', answer=ent(a),
                          rank_without_matchability=b, rank_with_matchability=f, answer_path=via(a),
                          top_wrong_without_matchability=[dict(entity=ent(w), score=round(S0[w].item(), 4),
                                                               score_with_matchability=round(S1[w].item(), 4), path=via(w))
                                                          for w in wrong]))
        if len(cases) == n_cases:
            break
    json.dump(cases, open(f'{OUT}/match_cases.json', 'w'), indent=1, ensure_ascii=False)
    print('match cases:', len(cases), 'of', len(found), 'rescued answers', flush=True)



if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for part in sys.argv[1:] or ['dual', 'match']:
        globals()[part]()
