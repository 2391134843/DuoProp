"""Offline cost of QTO and DuoProp on one link predictor, measured on one GPU used exclusively: QTO's forward truth matrices
(T = 1), Duo's dual matrices for one (T, alpha) setting, the matchability probes, and the two kNN graphs of Prop.  Each step
runs as in the pipeline (same builders); temporary outputs are deleted.  Output: runs/paper/build_bench_<name>.json.
python tools/build_bench.py NAME DATA CKPT QTO_EPS DUO_EPS T ALPHA"""
import json
import os
import subprocess
import sys
import time

import torch

name, data, ckpt, qeps, deps, T, alpha = sys.argv[1:8]
PY = [sys.executable, '-W', 'ignore', '-m']
tmp = f'cache/mats/bench_{name}'
res = dict(name=name, data=data, ckpt=ckpt, gpu=torch.cuda.get_device_name(0))


def timed(key, cmd):
    t0 = time.time()
    subprocess.run(cmd, check=True)
    res[key] = round(time.time() - t0, 1)
    print(key, res[key], flush=True)


timed('match_s', PY + ['duoprop.match', '--data', data, '--ckpt', ckpt, '--out', f'{tmp}_match.pt'])
timed('qto_build_s', PY + ['duoprop.build', '--data', data, '--ckpt', ckpt, '--out', f'{tmp}_q', '--mode', 'forward', '--temps', '1.0',
                           '--eps', qeps, '--shard', '0', '--nshards', '1'])
timed('qto_merge_s', PY + ['duoprop.build', '--data', data, '--out', f'{tmp}_q', '--mode', 'forward', '--merge', '--nshards', '1',
                           '--temps', '1.0'])
timed('duo_build_s', PY + ['duoprop.build', '--data', data, '--ckpt', ckpt, '--out', f'{tmp}_d', '--mode', 'dual', '--temps', T,
                           '--alphas', alpha, '--betas', '0', '--eps', deps, '--shard', '0', '--nshards', '1'])
timed('duo_merge_s', PY + ['duoprop.build', '--data', data, '--out', f'{tmp}_d', '--mode', 'dual', '--merge', '--nshards', '1',
                           '--temps', T, '--alphas', alpha, '--betas', '0'])
for f, key in [(f'{tmp}_q_forward_T1.0.pt', 'qto_file_GB'), (f'{tmp}_d_dual_T{float(T)}_a{float(alpha)}_b0.0.pt', 'duo_file_GB')]:
    res[key] = round(os.path.getsize(f) / 1e9, 2)
    os.remove(f)
os.remove(f'{tmp}_match.pt')
# kNN graphs of Prop: structural profiles (training graph) and backbone embeddings
from duoprop.neighbours import knn, profiles
from duoprop.scorer import make_scorer, read_stats
n, nrel = read_stats(data)
torch.cuda.synchronize()
t0 = time.time()
P = profiles(data, n, nrel, 'rel')
knn(P, 100)
torch.cuda.synchronize()
res['knn_structural_s'] = round(time.time() - t0, 1)
del P
torch.cuda.empty_cache()
t0 = time.time()
X = make_scorer(ckpt, 'cuda').features().float()
X = (X - X.mean(0)) / (X.std(0) + 1e-6)
X = X / X.norm(dim=1, keepdim=True)
for s0 in range(0, n, 4096):
    G = X[s0:s0 + 4096] @ X.t()
    G.topk(100, dim=1)
torch.cuda.synchronize()
res['knn_embedding_s'] = round(time.time() - t0, 1)
os.makedirs('runs/paper', exist_ok=True)
json.dump(res, open(f'runs/paper/build_bench_{name}.json', 'w'), indent=1)
print(json.dumps(res), flush=True)
print('PART DONE', flush=True)
