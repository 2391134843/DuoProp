"""Pick the best ComplEx variant of one dataset by link-prediction validation MRR (kbc 'Best valid mrr'), and, if it
beats the QTO-setting checkpoint, register its DuoProp evaluation: autopipe pipeline (runs/autopipe_extra.json, with
the QTO baseline on the same checkpoint where feasible) + joint search with both selection criteria.  Only the chosen
variant gets caches, so the variant sweep itself costs no disk beyond the checkpoints.
Usage: python tools/pick_kbc.py <family> <base run dir> <variant run dir> [...]   (family: cxv237 | cxv15k | cxvnell)"""
import json
import os
import re
import subprocess
import sys

FAM = {  # data, pipeline args, temps, alphas, weight (widened grids around the multi-seed selections)
    'cxv237': ('data/FB15k-237-betae', ['--nshards', '1', '--neg_scales', '3', '--paper_eps', '2e-4', '--paper_neg', '3',
                                        '--store_eps', '3e-4'], '0.5,0.75,1.0,1.25', '0.7,0.85,0.95', 1),
    'cxv15k': ('data/FB15k-betae', ['--nshards', '4', '--neg_scales', '6', '--paper_eps', '1e-3', '--paper_neg', '6'],
               '0.75,1.0,1.25', '0.7,0.85,0.95', 2),
    'cxvnell': ('data/NELL-betae', ['--nshards', '6', '--neg_scales', '6', '--blocked', '--store_eps', '3e-4'],
                '0.75,1.0,1.25,1.5', '0.7,0.85,0.95', 1),
}


def best_valid(run):
    f = os.path.join(run, 'kbc', 'train.log')
    m = re.findall(r'Best valid mrr: ([0-9.]+)', open(f).read()) if os.path.exists(f) else []
    return float(m[-1]) if m else -1.0  # failed / missing run


if __name__ == '__main__':
    fam, base, variants = sys.argv[1], sys.argv[2], sys.argv[3:]
    data, args, temps, alphas, w = FAM[fam]
    scores = {r: best_valid(r) for r in [base] + variants}
    lines = [f'# {fam}: ComplEx variants by link-prediction validation MRR'] + \
            [f'- {os.path.basename(r)}: {s:.4f}' for r, s in sorted(scores.items(), key=lambda x: -x[1])]
    best = max(variants, key=lambda r: scores[r])
    lines.append(f'best variant {os.path.basename(best)} ({scores[best]:.4f}) vs QTO-setting checkpoint {scores[base]:.4f}')
    if scores[best] > scores[base]:
        tag = f'{fam}_best'
        extra = json.load(open('runs/autopipe_extra.json'))
        extra[tag] = [data, best, ['--nparts', '1', '--row_betas', '2', '--build_weight', '1'] + args +
                      ['--temps', temps, '--alphas', alphas]]
        json.dump(extra, open('runs/autopipe_extra.json.tmp', 'w'), indent=1)
        os.replace('runs/autopipe_extra.json.tmp', 'runs/autopipe_extra.json')
        subprocess.run(['python', 'duoprop/gen_joint.py', '--name', 'MS_' + tag, '--tag', tag, '--knn_tag', tag,
                        '--data', data, '--temps', temps, '--alphas', alphas, '--match', f'cache/art/match_{tag}.pt',
                        '--sel_tag', tag, '--irreflexive', 'on', '--weight', str(w), '--prune', '--criteria', 'sum,pos', '--refine_top', '5'],
                       check=True)
        if '--paper_eps' in args:
            with open('runs/jobs.txt', 'a') as f:
                f.write(f'0 rmfwd_{tag} until [ -f runs/results/{tag}_forward_test.jsonl ]; do sleep 120; done; '
                        f'rm -f cache/mats/{tag}_forward_T1.0.pt\n')
        lines.append(f'registered DuoProp evaluation as {tag} (pipeline + joint search MS_{tag})')
    else:
        lines.append('no variant beats the QTO-setting checkpoint: nothing registered')
    open(f'runs/{fam}_variants.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
