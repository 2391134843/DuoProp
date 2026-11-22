"""End-to-end evaluation of one SpherE checkpoint with our reasoning pipeline.

1. matchability probe   2. build dual truth matrices for a (T, alpha, beta) grid
3. validation sweep over the grid x eval-eps x neg_scale (hyper-parameters chosen on VALID only)
4. full test evaluation of the selected configuration.
Results are appended to runs/results/<tag>_{valid,test}.jsonl.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import torch

from duoprop.build import tag as cfg_tag
from duoprop.reason import NEG, POS, Reasoner, evaluate, fmt, load_cache, load_split
from duoprop.scorer import read_stats

PY = [sys.executable, '-W', 'ignore', '-m']


def run(cmd):
    print('$', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def sweep(cache, n, q, e, h, eps_list, neg_scales, limit, row_match=None, row_betas=(0.0,)):
    out = []
    for eps in eps_list:
        for rb in row_betas:
            adj = load_cache(cache, n, 'cuda', None if eps == 'none' else float(eps), row_match, rb)
            pos = evaluate(Reasoner(adj, n, 'cuda'), q, e, h, POS, limit)
            for s in neg_scales:
                neg = evaluate(Reasoner(adj, n, 'cuda', float(s)), q, e, h, NEG, limit)
                res = {**{k: v for k, v in pos.items() if k != 'AVG_n'}, **{k: v for k, v in neg.items() if k != 'AVG_p'}}
                out.append(dict(cache=cache, eps=eps, neg_scale=float(s), row_beta=rb, res=res))
            del adj
            torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--temps', default='1.25,1.5,2.0')
    ap.add_argument('--alphas', default='0.5,0.6,0.7')
    ap.add_argument('--betas', default='0,0.5,1')
    ap.add_argument('--eps_eval', default='none,3e-4')
    ap.add_argument('--neg_scales', default='1,2,3')
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--blocked', action='store_true')
    ap.add_argument('--skip_build', action='store_true')
    ap.add_argument('--extra_caches', default='', help='additional caches to include in the sweep')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1, help='split the validation sweep over several jobs')
    ap.add_argument('--select_only', action='store_true', help='pick the best record of <tag>_valid.jsonl, run test')
    ap.add_argument('--paper_eps', type=float, default=None,
                    help='also evaluate the original forward calibration (T=1) with the paper eps / neg_scale')
    ap.add_argument('--paper_neg', type=float, default=1.0)
    ap.add_argument('--row_betas', default='0', help='query-side matchability exponents to sweep')
    ap.add_argument('--caches', default=None, help='explicit comma-separated caches (overrides the glob)')
    ap.add_argument('--suffix', default='', help='suffix for the results files')
    args = ap.parse_args()
    os.makedirs('runs/results', exist_ok=True)
    n, _ = read_stats(args.data)
    match = f'cache/art/match_{args.tag}.pt'
    prefix = f'cache/mats/{args.tag}'
    grid = dict(temps=args.temps, alphas=args.alphas, betas=args.betas)
    if not args.skip_build:
        run(PY + ['duoprop.match', '--data', args.data, '--ckpt', args.ckpt, '--out', match])
        builder = 'duoprop.build_blocked' if args.blocked else 'duoprop.build'
        extra = [] if args.blocked else ['--mode', 'dual']
        run(PY + [builder, '--data', args.data, '--ckpt', args.ckpt, '--out', prefix, '--match', match,
                  '--temps', grid['temps'], '--alphas', grid['alphas'], '--betas', grid['betas'],
                  '--eps', '1e-4', '--shard', '0', '--nshards', '1'] + extra)
        run(PY + ['duoprop.build', '--data', args.data, '--out', prefix, '--mode', 'dual', '--merge', '--nshards', '1',
                  '--temps', grid['temps'], '--alphas', grid['alphas'], '--betas', grid['betas']])
    if args.paper_eps is not None and not args.select_only and args.part == 0:
        run(PY + ['duoprop.build', '--data', args.data, '--ckpt', args.ckpt, '--out', prefix, '--mode', 'forward',
                  '--temps', '1.0', '--eps', str(args.paper_eps), '--shard', '0', '--nshards', '1'])
        run(PY + ['duoprop.build', '--data', args.data, '--out', prefix, '--mode', 'forward', '--merge',
                  '--nshards', '1', '--temps', '1.0'])
        qt, et, ht = load_split(args.data, 'test')
        adj = load_cache(prefix + '_forward_T1.0.pt', n, 'cuda')
        res = evaluate(Reasoner(adj, n, 'cuda', args.paper_neg), qt, et, ht)
        with open(f'runs/results/{args.tag}_forward_test.jsonl', 'a') as f:
            f.write(json.dumps(dict(tag=args.tag + '_forward', ckpt=args.ckpt, cache=prefix + '_forward_T1.0.pt',
                                    eps=args.paper_eps, neg_scale=args.paper_neg, test=res)) + '\n')
        print('TEST (paper calibration)\n' + fmt(res), flush=True)
        del adj
        torch.cuda.empty_cache()
    caches = sorted(glob.glob(prefix + '_dual_*.pt')) + [c for c in args.extra_caches.split(',') if c]
    if args.caches:
        caches = args.caches.split(',')
    row_betas = [float(x) for x in args.row_betas.split(',')]
    row_match = torch.load(match, weights_only=False) if any(b > 0 for b in row_betas) else None
    caches = caches[args.part::args.nparts]
    qv, ev, hv = load_split(args.data, 'valid')
    results = []
    for c in ([] if args.select_only else caches):
        t0 = time.time()
        eps_list = args.eps_eval.split(',')
        size_gb = os.path.getsize(c) / 1e9  # large (near-dense) matrices must be thresholded to fit the GPU
        if size_gb > 15:
            eps_list = ['1e-3']
        elif size_gb > 4:
            eps_list = ['3e-4', '1e-3']
        for rec in sweep(c, n, qv, ev, hv, eps_list, args.neg_scales.split(','), args.limit, row_match, row_betas):
            rec['score'] = rec['res']['AVG_p'] + rec['res']['AVG_n']
            results.append(rec)
            with open(f'runs/results/{args.tag}{args.suffix}_valid.jsonl', 'a') as f:
                f.write(json.dumps(rec) + '\n')
        print(f'valid sweep {os.path.basename(c)} done ({time.time() - t0:.0f}s)', flush=True)
    if args.nparts > 1:
        print('PART DONE', flush=True)
        raise SystemExit
    if args.select_only:
        results = [json.loads(l) for l in open(f'runs/results/{args.tag}{args.suffix}_valid.jsonl')]
    best = max(results, key=lambda x: x['score'])
    print('BEST on valid:', os.path.basename(best['cache']), best['eps'], best['neg_scale'],
          f"AVG_p={best['res']['AVG_p']:.2f} AVG_n={best['res']['AVG_n']:.2f}", flush=True)
    qt, et, ht = load_split(args.data, 'test')
    rb = best.get('row_beta', 0.0)
    rm = torch.load(match, weights_only=False) if rb > 0 else None
    adj = load_cache(best['cache'], n, 'cuda', None if best['eps'] == 'none' else float(best['eps']), rm, rb)
    res = evaluate(Reasoner(adj, n, 'cuda', best['neg_scale']), qt, et, ht)
    rec = dict(tag=args.tag, ckpt=args.ckpt, cache=best['cache'], eps=best['eps'], neg_scale=best['neg_scale'],
               row_beta=rb, valid=best['res'], test=res)
    with open(f'runs/results/{args.tag}{args.suffix}_test.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print('TEST\n' + fmt(res), flush=True)
