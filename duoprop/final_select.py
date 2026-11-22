"""Final model selection: for every checkpoint, merge the validation records of all its grids and pick the
single best configuration by validation score (AVG_p + AVG_n).  Its test record (computed by the grid
that contained it) is reported.  Test numbers never influence the choice."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = {  # checkpoint -> tags of all validation sweeps run on it
    'FB15k-237 (paper ckpt, d=256)': ['fb237_hiT', 'fb237_hiT2', 'fb237_base_rm2_a'],
    'FB15k (d=256)': ['fb15k_base', 'fb15k_ref'],
    'NELL995 (d=256)': ['nell_base', 'nell_ref'],
    'FB15k-237 (d=512, Table 4 setting)': ['fb237_d512', 'fb237_d512b'],
    'FB15k-237 (d=1000, existing ckpt)': ['fb237_d1000', 'fb237_d1000b'],
}


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def key(r):
    return (os.path.basename(r['cache']), str(r['eps']), float(r.get('neg_scale', 1)), float(r.get('row_beta', 0)))


if __name__ == '__main__':
    import glob
    grouped = {t for ts in GROUPS.values() for t in ts}
    for f in sorted(glob.glob(os.path.join(ROOT, 'runs/results/*_valid.jsonl'))):
        t = os.path.basename(f)[:-len('_valid.jsonl')]
        if t not in grouped and not t.startswith('fb237_base') and not any(x in t for x in ('evid', 'role', 'relT', 'negset', 'obs', 'fuse')):
            GROUPS[f'ckpt {t}'] = [t]
    out = {}
    for group, tags in GROUPS.items():
        valid = []
        tests = {}
        for t in tags:
            for r in load(os.path.join(ROOT, f'runs/results/{t}_valid.jsonl')):
                if 'cache' not in r or 'AVG_p' not in r['res'] or 'AVG_n' not in r['res']:
                    continue  # probe files with a different format
                r['score'] = r['res']['AVG_p'] + r['res']['AVG_n']
                r['tag'] = t
                valid.append(r)
            for r in load(os.path.join(ROOT, f'runs/results/{t}_test.jsonl')):
                tests[key(r)] = r
        if not valid:
            continue
        best = max(valid, key=lambda r: r['score'])
        test = tests.get(key(best))
        print(f'{group}: best on valid = {key(best)}  valid AVG_p={best["res"]["AVG_p"]:.2f} AVG_n={best["res"]["AVG_n"]:.2f}'
              + (f'  -> TEST AVG_p={test["test"]["AVG_p"]:.2f} AVG_n={test["test"]["AVG_n"]:.2f}' if test else '  -> test missing'))
        out[group] = dict(best_valid=best, test=test['test'] if test else None)
    json.dump(out, open(os.path.join(ROOT, 'runs/results/FINAL_SELECTION.json'), 'w'), indent=1)
