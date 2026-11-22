"""QTO baseline on a checkpoint for large graphs (NELL995): QTO's forward calibration (T = 1) built with the blocked builder
(Duo with alpha = 1 is exactly the head-side matrix with training edges set to 1), stored at QTO's threshold, evaluated once
on test with QTO's negation scaling (standard protocol) and under the consistent protocol (validation edges = 1, s in 1/3/6).
Appends runs/results/<tag>_forward_test.jsonl and writes runs/paper/qtofwd_<tag>.json; deletes the matrix.
python tools/qto_forward_blocked.py DATA TAG EPS NEG"""
import json
import os
import subprocess
import sys

from duoprop.collective_eval import observe
from duoprop.reason import NEG, Reasoner, evaluate, fmt, load_cache, load_split
from duoprop.scorer import read_stats

data, tag, eps, neg = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
ckpt = f'runs/ckpt/{tag}/r1/checkpoint'
prefix = f'cache/mats/qtofwd_{tag}'
PY = [sys.executable, '-W', 'ignore', '-m']
subprocess.run(PY + ['duoprop.build_blocked', '--data', data, '--ckpt', ckpt, '--out', prefix, '--temps', '1.0', '--alphas', '1.0',
                     '--betas', '0', '--eps', eps, '--shard', '0', '--nshards', '1'], check=True)
subprocess.run(PY + ['duoprop.build', '--mode', 'dual', '--data', data, '--out', prefix, '--merge', '--nshards', '1', '--temps', '1.0',
                     '--alphas', '1.0', '--betas', '0'], check=True)
cache = prefix + '_dual_T1.0_a1.0_b0.0.pt'
n, _ = read_stats(data)
qt, et, ht = load_split(data, 'test')
adj = load_cache(cache, n, 'cuda')
res = evaluate(Reasoner(adj, n, 'cuda', neg), qt, et, ht)
with open(f'runs/results/{tag}_forward_test.jsonl', 'a') as f:
    f.write(json.dumps(dict(tag=tag + '_forward', ckpt=ckpt, cache=cache, eps=float(eps), neg_scale=neg, test=res)) + '\n')
print('TEST (QTO, standard protocol)\n' + fmt(res), flush=True)
observe(adj, data, 'valid')
out = dict(tag=tag, eps=eps, standard=res)
pos = evaluate(Reasoner(adj, n, 'cuda', 1.0), qt, et, ht, types=[t for t in ('1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up')])
out['consistent'] = dict(AVG_p=pos['AVG_p'])
for s in (1.0, 3.0, 6.0):
    out['consistent'][f'AVG_n_s{s:g}'] = evaluate(Reasoner(adj, n, 'cuda', s), qt, et, ht, types=NEG)['AVG_n']
os.makedirs('runs/paper', exist_ok=True)
json.dump(out, open(f'runs/paper/qtofwd_{tag}.json', 'w'), indent=1)
print('consistent', json.dumps(out['consistent']), flush=True)
os.remove(cache)
print('PART DONE', flush=True)
