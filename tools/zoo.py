"""Validation-only member choice for diversity ensembles of evaluated runs (round 26: ComplEx backbones trained with
different hyper-parameters instead of different seeds).  A run's validation score is the best AVG_p over its DuoProp
joint search on the 1000-query validation subsets (the quantity its positives-only pick maximises); test is never read.
Usage: python tools/zoo.py check <run,...>          exit 0 once every run has its pick or one of its jobs failed
       python tools/zoo.py zoo <base,...> <cand,...> <max_extra>
                              base members + up to max_extra candidates scoring at least the weakest base member
       python tools/zoo.py top <run,...> <k>         the k best runs
       python tools/zoo.py caches <rho> <run> [<run> ...]   cache:match:rho of each run's positives-only pick
(the chosen runs are printed space-separated, the scores go to stderr)"""
import glob
import json
import os
import sys


def vscore(run):
    s = [json.loads(l)['res']['AVG_p'] for f in glob.glob(f'runs/results/MS_{run}_T*_coll_valid.jsonl') for l in open(f)]
    return max(s) if s else None


def ours(run, job):
    return job == run or job.startswith(run + '_') or job.endswith('_' + run) or f'_{run}_' in job


if __name__ == '__main__':
    if sys.argv[1] == 'caches':
        rho = sys.argv[2]
        recs = {r: [json.loads(l) for l in open(f'runs/results/MS_{r}_finalP_coll_test.jsonl')][-1] for r in sys.argv[3:]}
        print(' '.join(f"{recs[r]['cache']}:cache/art/match_{r}.pt:{rho}" for r in sys.argv[3:]))
        sys.exit(0)
    mode, runs = sys.argv[1], sys.argv[2].split(',')
    if mode == 'check':
        state = json.load(open('runs/queue_state.json'))
        failed = {r for r in runs for j, v in state.items() if ours(r, j) and v.get('status') == 'failed'}
        failed |= {r for r in runs if os.path.exists(f'runs/ckpt/{r}.failed')}  # training inside a bundle job
        done = {r for r in runs if os.path.exists(f'runs/results/MS_{r}_finalP_coll_test.jsonl')}
        print(f'{len(done)} picked, {len(failed - done)} failed, {len(set(runs) - done - failed)} pending', file=sys.stderr)
        sys.exit(0 if done | failed >= set(runs) else 1)
    if mode == 'zoo':
        cands, extra = sys.argv[3].split(','), int(sys.argv[4])
        floor = min(vscore(r) for r in runs)
        ok = sorted((r for r in cands if os.path.exists(f'runs/results/MS_{r}_finalP_coll_test.jsonl')
                     and vscore(r) is not None and vscore(r) >= floor), key=vscore, reverse=True)[:extra]
        chosen = runs + ok
        print(f'floor {floor:.2f} (weakest base member); added ' + ', '.join(f'{r} {vscore(r):.2f}' for r in ok), file=sys.stderr)
    else:
        k = int(sys.argv[3])
        chosen = sorted((r for r in runs if os.path.exists(f'runs/results/MS_{r}_finalP_coll_test.jsonl')
                         and vscore(r) is not None), key=vscore, reverse=True)[:k]
        print('top: ' + ', '.join(f'{r} {vscore(r):.2f}' for r in chosen), file=sys.stderr)
    print(' '.join(chosen))
