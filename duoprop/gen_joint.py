"""Append queue jobs for a joint validation search over Duo (temperature T, mixing alpha) and
Prop (propagation beta, aggregation) for one checkpoint to runs/jobs.txt.

[build shards -> merge] (only with --build_ckpt: rebuild the truth matrices under a new cache tag)
-> one collective_eval --no_test job per cache -> collective_eval --select_from (best setting, and the best
beta=0 setting of the same pool, evaluated once on test).  cache/mats/<tag>.hold keeps autopipe's cleanup
away from the caches until the selection job has finished.
"""
import argparse
import os

P = 'python -W ignore -m'
DS = {'FB15k-237-betae': 'fb237', 'FB15k-betae': 'fb15k', 'NELL-betae': 'nell'}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='job / result prefix')
    ap.add_argument('--tag', required=True, help='cache tag: cache/mats/<tag>_dual_T*_a*_b0.0.pt')
    ap.add_argument('--knn_tag', required=True, help='pipeline tag whose checkpoint builds the embedding kNN graph')
    ap.add_argument('--data', required=True)
    ap.add_argument('--temps', required=True)
    ap.add_argument('--alphas', required=True)
    ap.add_argument('--betas', default='0,0.1,0.2,0.3')
    ap.add_argument('--aggs', default='max,2,4')
    ap.add_argument('--eps', default='none')
    ap.add_argument('--rb', default='2')
    ap.add_argument('--neg_scale', default='1')
    ap.add_argument('--match', required=True)
    ap.add_argument('--weight', type=int, default=1)
    ap.add_argument('--build_ckpt', help='checkpoint (shell expression allowed); rebuild the caches first')
    ap.add_argument('--nshards', type=int, default=2)
    ap.add_argument('--blocked', action='store_true')
    ap.add_argument('--sel_tag', help='take eps / rb / neg_scale from the Duo selection of this pipeline tag '
                                      '(jobs then also wait for runs/results/<tag>_test.jsonl)')
    ap.add_argument('--irreflexive', default='off', choices=['off', 'on', 'both'],
                    help="irreflexive prior (see collective_eval); 'both' adds it to the searched grid")
    ap.add_argument('--neg_fixed', help="negation scaling fixed to the base reasoner's published value")
    ap.add_argument('--store_eps', default='1e-3', help='storage threshold of rebuilt caches')
    ap.add_argument('--wait', help='shell condition the builds wait for (e.g. a training run still in progress)')
    ap.add_argument('--select_extra', default='', help='more result prefixes joining the selection pool (older searches)')
    ap.add_argument('--refine_top', type=int, default=0, help='two-stage selection (see collective_eval --refine_top)')
    ap.add_argument('--criteria', default='pos,sum', help="comma list of selection criteria (collective_eval --criterion): "
                                                      "'sum' -> <name>_final, 'pos' -> <name>_finalP")
    ap.add_argument('--select_wait', help='extra shell condition for the selection job (older pools finished)')
    ap.add_argument('--prune', action='store_true', help='after the selection, delete the caches of this tag it did not pick')
    a = ap.parse_args()
    calib = f'--sel_tag {a.sel_tag}' if a.sel_tag else f'--eps {a.eps} --rb {a.rb} --neg_scale {a.neg_scale}'
    calib += f' --neg_fixed {a.neg_fixed}' if a.neg_fixed else ''
    irr = {'off': [''], 'on': ['_irr'], 'both': ['', '_irr']}[a.irreflexive]
    temps = [str(float(t)) for t in a.temps.split(',')]
    alphas = [str(float(x)) for x in a.alphas.split(',')]
    pre = f'cache/mats/{a.tag}'
    knn = f'cache/art/knn_{DS[os.path.basename(a.data.rstrip("/"))]}_emb_{a.knn_tag}.pt'
    lines = []
    if a.build_ckpt:
        grid = f"--temps {','.join(temps)} --alphas {','.join(alphas)} --betas 0"
        builder = 'duoprop.build_blocked' if a.blocked else 'duoprop.build --mode dual'
        wait = f'until {a.wait}; do sleep 120; done; ' if a.wait else ''
        for k in range(a.nshards):
            lines.append(f'1 {a.name}_build{k} {wait}{P} {builder} --data {a.data} --ckpt {a.build_ckpt} --out {pre} '
                         f'--match {a.match} {grid} --eps {a.store_eps} --shard {k} --nshards {a.nshards}')
        done = ' && '.join(f"grep -qs 'shard done' runs/logs/{a.name}_build{k}.log" for k in range(a.nshards))
        lines.append(f'0 {a.name}_merge until {done}; do sleep 30; done; {P} duoprop.build --mode dual --data {a.data} '
                     f'--out {pre} {grid} --nshards {a.nshards} --merge && touch {pre}.merged')
    names = []
    for i, (t, al, x) in enumerate((t, al, x) for t in temps for al in alphas for x in irr):
        nm = f'{a.name}_T{t}_a{al}{x}'
        names.append(nm)
        cond = f'[ -f {pre}.merged ]' + ('' if i == 0 or os.path.exists(knn) else f' && [ -f {knn} ]')
        cond += f' && [ -f runs/results/{a.sel_tag}_test.jsonl ]' if a.sel_tag else ''
        lines.append(f'{a.weight} {nm} until {cond}; do sleep 60; done; sleep 30; {P} duoprop.collective_eval '
                     f'--kind rel+emb --betas {a.betas} --aggs {a.aggs} --extra= --data {a.data} --name {nm} '
                     f'--tag {a.knn_tag} --cache {pre}_dual_T{t}_a{al}_b0.0.pt {calib}'
                     f'{" --irreflexive" if x else ""} --match {a.match} --no_test')
    done = ' && '.join(f"grep -qs 'PART DONE' runs/logs/{nm}.log" for nm in names)
    done += f' && {a.select_wait}' if a.select_wait else ''
    pool = ','.join([f'{a.name}_T'] + [x for x in a.select_extra.split(',') if x])
    refine = f' --refine_top {a.refine_top}' if a.refine_top else ''
    finals = {'sum': f'{a.name}_final', 'pos': f'{a.name}_finalP'}  # one selection per criterion, same pool
    sel = ' && '.join(f'{P} duoprop.collective_eval --kind rel+emb --data {a.data} --name {finals[c]} --tag {a.knn_tag} '
                      f'--match {a.match} --select_from {pool}{refine} --criterion {c}' for c in a.criteria.split(','))
    keep = ','.join(finals[c] for c in a.criteria.split(','))
    lines.append(f'{a.weight} {a.name}_select until {done}; do sleep 60; done; {sel} && rm -f {pre}.hold'
                 + (f' && {P} duoprop.prune --tag {a.tag} --keep_from {keep}' if a.prune else ''))
    open(f'{pre}.hold', 'w').close()
    with open('runs/jobs.txt', 'a') as f:
        f.write(f'# ===== joint search {a.name} ({len(names)} caches) =====\n' + '\n'.join(lines) + '\n')
    print(f'{a.name}: {len(lines)} jobs')
