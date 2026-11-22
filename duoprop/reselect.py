"""Re-select existing table cells with another criterion (default: AVG_p only, see collective_eval --criterion) and
evaluate the picks once on test, without keeping caches around: a pick whose cache was deleted is rebuilt (single
T x alpha, original storage threshold), used, and deleted again.  Results: runs/results/RSP_<cell>_coll_test.jsonl.
Usage: python -m duoprop.reselect <cell> [<cell> ...]   (cells: see CELLS)"""
import glob
import json
import os
import re
import subprocess
import sys

PY = [sys.executable, '-W', 'ignore', '-m']
LS = lambda d: subprocess.check_output(['bash', '-c', f'echo $(ls -d {d}/*/ | tail -n 1)checkpoint']).decode().strip()
D237, D15K, DNELL = 'data/FB15k-237-betae', 'data/FB15k-betae', 'data/NELL-betae'
PAPER = 'logs/FB15k-237-betae/sphere/g-24.0-dim-256/2025.05.11-13.07.54/checkpoint'
# cache tag -> (checkpoint, storage eps, blocked builder)
TAGS = {
    'E_J237_paper': (PAPER, '3e-4', False), 'jb237_paper': (PAPER, '1e-3', False),
    'E_J237_d512': (lambda: LS('runs/ckpt/t237_d512/FB15k-237-betae/sphere/g-24.0-dim-512'), '3e-4', False),
    'jb237_d512': (lambda: LS('runs/ckpt/t237_d512/FB15k-237-betae/sphere/g-24.0-dim-512'), '1e-3', False),
    'E_J237_d1000': (lambda: LS('logs/FB15k-237-betae/sphere/g-24.0-dim-1000'), '3e-4', False),
    'jb237_d1000': (lambda: LS('logs/FB15k-237-betae/sphere/g-24.0-dim-1000'), '1e-3', False),
    'E_JNcx237': ('runs/ckpt/cx_fb237/r1/checkpoint', '3e-4', False), 'jbcx237': ('runs/ckpt/cx_fb237/r1/checkpoint', '1e-3', False),
    'E_c2i_fb237_gqe': (lambda: LS('runs/ckpt/tgqe_fb15k237/FB15k-237-betae/vec/g-24.0-dim-800'), '3e-4', False),
    'E_c2i_fb237_q2b': (lambda: LS('runs/ckpt/tq2b_fb15k237/FB15k-237-betae/box/g-24.0-dim-400'), '3e-4', False),
    'jb15k_ref': (lambda: LS('logs/FB15k-betae/sphere/g-36.0-dim-256'), '1e-3', False),
    'fb15kh_s1500k': (lambda: LS('runs/ckpt/h15k_s1500k/FB15k-betae/sphere/g-36.0-dim-256'), '1e-3', False),
    'cx15k': ('runs/ckpt/cx_fb15k/r1/checkpoint', '1e-3', False),
    'jbnell_ref': (lambda: LS('logs/NELL-betae/sphere/g-24.0-dim-256'), '1e-3', True),
    'nellh_s900k': (lambda: LS('runs/ckpt/hnell_s900k/NELL-betae/sphere/g-24.0-dim-256'), '1e-3', True),
    'cxnell': ('runs/ckpt/cx_nell/r1/checkpoint', '1e-3', True),
}
# cell -> (pool prefixes, data, match, knn tag)
CELLS = {
    'fb237_sph_paper': ('J237_paper_T', D237, 'fb237_hiT'), 'fb237_sph_d512': ('J237_d512_T', D237, 'fb237_d512'),
    'fb237_sph_d1000': ('J237_d1000_T', D237, 'fb237_d1000'), 'fb237_gqe': ('c2i_fb237_gqe', D237, 'fb15k237_gqe1p'),
    'fb237_q2b': ('c2i_fb237_q2b', D237, 'fb15k237_q2b1p'), 'fb237_cx': ('JNcx237_T', D237, 'cx237'),
    'fb15k_sph_ref': ('JN15k_ref_T', D15K, 'fb15k_ref'), 'fb15k_sph_s1500k': ('JN15kh_s1500k_T', D15K, 'fb15kh_s1500k'),
    'fb15k_cx': ('JNcx15k_T', D15K, 'cx15k'), 'nell_sph_ref': ('JNnell_ref_T', DNELL, 'nell_ref'),
    'nell_sph_s900k': ('JNnellh_s900k_T', DNELL, 'nellh_s900k'), 'nell_cx': ('JNcxnell_T', DNELL, 'cxnell'),
    'fb15k_gqe': ('c2N_fb15k_gqe,c2Ni_fb15k_gqe', D15K, 'fb15k_gqe1p'), 'fb15k_q2b': ('c2N_fb15k_q2b,c2Ni_fb15k_q2b', D15K, 'fb15k_q2b1p'),
    'nell_gqe': ('c2N_nell_gqe,c2Ni_nell_gqe', DNELL, 'nell_gqe1p'), 'nell_q2b': ('c2N_nell_q2b,c2Ni_nell_q2b', DNELL, 'nell_q2b1p'),
}


def run(cmd):
    print('+', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    criterion = os.environ.get('CRITERION', 'pos')
    for cell in sys.argv[1:]:
        prefixes, data, knn = CELLS[cell]
        match = f'cache/art/match_{knn}.pt'
        recs = [json.loads(l) for p in prefixes.split(',') for f in glob.glob(f'runs/results/{p}*_coll_valid.jsonl')
                for l in open(f)]
        score = (lambda r: r['res']['AVG_p']) if criterion == 'pos' else (lambda r: r['res']['AVG_p'] + r['res']['AVG_n'])
        picks = [max((r for r in recs if r['beta'] == 0 and r['agg'] == 'max'), key=score), max(recs, key=score)]
        rebuilt = []
        # autopipe's cleanup deletes caches no test record references unless the tag holds a .hold file: hold every
        # tag the picks use for the whole cell (only the holds created here are removed afterwards)
        holds = [f"cache/mats/{os.path.basename(p['cache']).split('_dual_')[0]}.hold" for p in picks]
        holds = [h for h in dict.fromkeys(holds) if not os.path.exists(h)]
        for h in holds:
            open(h, 'w').close()
        for cache in dict.fromkeys(p['cache'] for p in picks):
            if os.path.exists(cache):
                continue
            tag = os.path.basename(cache).split('_dual_')[0]
            T, al = re.search(r'_dual_T([0-9.]+)_a([0-9.]+)_b', cache).groups()
            ckpt, eps, blocked = TAGS[tag]
            ckpt = ckpt() if callable(ckpt) else ckpt
            grid = ['--temps', T, '--alphas', al, '--betas', '0']
            builder = ['duoprop.build_blocked'] if blocked else ['duoprop.build', '--mode', 'dual']
            run(PY + builder + ['--data', data, '--ckpt', ckpt, '--out', f'cache/mats/{tag}', '--match', match,
                                '--eps', eps, '--shard', '0', '--nshards', '1'] + grid)
            run(PY + ['duoprop.build', '--mode', 'dual', '--data', data, '--out', f'cache/mats/{tag}', '--nshards', '1',
                      '--merge'] + grid)
            rebuilt.append(cache)
        run(PY + ['duoprop.collective_eval', '--kind', 'rel+emb', '--data', data, '--name', f'RS{criterion[0].upper()}_{cell}',
                  '--tag', knn, '--match', match, '--select_from', prefixes, '--criterion', criterion])
        for cache in rebuilt:
            os.remove(cache)
        for h in holds:
            os.remove(h)
        print(f'CELL DONE {cell} (rebuilt and removed: {rebuilt})', flush=True)
    print('PART DONE', flush=True)
