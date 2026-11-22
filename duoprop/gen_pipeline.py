"""Append a dependency chain of queue jobs for one checkpoint to runs/jobs.txt.

ready-condition -> match -> build shards -> merge -> validation sweep parts -> select + test.
Every job waits (cheap sleep loop) for the artefact of its predecessor.
"""
import argparse
import shlex

P = 'python -W ignore -m'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--logdir', required=True, help='directory whose */train.log and */checkpoint belong to the run')
    ap.add_argument('--temps', default='1.5,2.0,3.0')
    ap.add_argument('--alphas', default='0.5,0.6,0.7')
    ap.add_argument('--betas', default='0')
    ap.add_argument('--nshards', type=int, default=2)
    ap.add_argument('--nparts', type=int, default=2)
    ap.add_argument('--blocked', action='store_true')
    ap.add_argument('--build_weight', type=int, default=2)
    ap.add_argument('--paper_eps', default=None)
    ap.add_argument('--paper_neg', default='1')
    ap.add_argument('--neg_scales', default='1,2,3')
    ap.add_argument('--eps_eval', default='none')
    ap.add_argument('--row_betas', default='0,2,3')
    ap.add_argument('--store_eps', default='1e-3')
    args = ap.parse_args()
    ck = f'$(ls -d {args.logdir}/*/ | tail -n 1)checkpoint'
    ready = f"until grep -qs 'Evaluating on Test Dataset' {args.logdir}/*/train.log; do sleep 60; done; sleep 30;"
    grid = f'--temps {args.temps} --alphas {args.alphas} --betas {args.betas}'
    match = f'cache/art/match_{args.tag}.pt'
    pre = f'cache/mats/{args.tag}'
    builder = 'duoprop.build_blocked' if args.blocked else 'duoprop.build --mode dual'
    lines = [f'1 {args.tag}_match {ready} {P} duoprop.match --data {args.data} --ckpt {ck} --out {match}']
    for k in range(args.nshards):
        lines.append(f'{args.build_weight} {args.tag}_build{k} until [ -f {match} ]; do sleep 30; done; sleep 10; '
                     f'{P} {builder} --data {args.data} --ckpt {ck} --out {pre} --match {match} {grid} '
                     f'--eps {args.store_eps} --shard {k} --nshards {args.nshards}')
    done = ' && '.join(f"grep -qs 'shard done' runs/logs/{args.tag}_build{k}.log" for k in range(args.nshards))
    lines.append(f'0 {args.tag}_merge until {done}; do sleep 30; done; {P} duoprop.build --mode dual --data {args.data} '
                 f'--out {pre} {grid} --nshards {args.nshards} --merge && touch {pre}.merged')
    extra = ''
    if args.paper_eps:
        extra = f'--paper_eps {args.paper_eps} --paper_neg {args.paper_neg}'
    for k in range(args.nparts):
        lines.append(f'1 {args.tag}_sweep{k} until [ -f {pre}.merged ]; do sleep 30; done; {P} duoprop.run_ckpt '
                     f'--data {args.data} --ckpt {ck} --tag {args.tag} --skip_build --part {k} --nparts {args.nparts} '
                     f'--neg_scales {args.neg_scales} --eps_eval {args.eps_eval} --row_betas {args.row_betas} {extra if k == 0 else ""}')
    done = ' && '.join(f"grep -qs 'PART DONE' runs/logs/{args.tag}_sweep{k}.log" for k in range(args.nparts))
    lines.append(f'1 {args.tag}_test until {done}; do sleep 30; done; {P} duoprop.run_ckpt --data {args.data} --ckpt {ck} '
                 f'--tag {args.tag} --skip_build --select_only')
    with open('runs/jobs.txt', 'a') as f:
        f.write(f'# ===== pipeline {args.tag} =====\n' + '\n'.join(lines) + '\n')
    print('\n'.join(l[:150] for l in lines))
