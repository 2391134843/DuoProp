"""QTO baseline on a given checkpoint (the original forward calibration T = 1 with the paper's storage eps and negation
scaling), exactly as duoprop/run_ckpt.py --paper_eps does, without the DuoProp sweep: build the forward truth matrix,
evaluate on test once, append runs/results/<tag>_forward_test.jsonl, delete the matrix.
python tools/qto_forward.py DATA TAG EPS NEG"""
import json
import os
import subprocess
import sys

import torch

from duoprop.reason import Reasoner, evaluate, fmt, load_cache, load_split
from duoprop.scorer import read_stats

data, tag, eps, neg = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
ckpt = f'runs/ckpt/{tag}/r1/checkpoint'
prefix = f'cache/mats/qtofwd_{tag}'
PY = [sys.executable, '-W', 'ignore', '-m']
subprocess.run(PY + ['duoprop.build', '--data', data, '--ckpt', ckpt, '--out', prefix, '--mode', 'forward', '--temps', '1.0',
                     '--eps', eps, '--shard', '0', '--nshards', '1'], check=True)
subprocess.run(PY + ['duoprop.build', '--data', data, '--out', prefix, '--mode', 'forward', '--merge', '--nshards', '1',
                     '--temps', '1.0'], check=True)
n, _ = read_stats(data)
qt, et, ht = load_split(data, 'test')
adj = load_cache(prefix + '_forward_T1.0.pt', n, 'cuda')
res = evaluate(Reasoner(adj, n, 'cuda', neg), qt, et, ht)
with open(f'runs/results/{tag}_forward_test.jsonl', 'a') as f:
    f.write(json.dumps(dict(tag=tag + '_forward', ckpt=ckpt, cache=prefix + '_forward_T1.0.pt', eps=float(eps), neg_scale=neg,
                            test=res)) + '\n')
print('TEST (QTO, paper calibration)\n' + fmt(res), flush=True)
del adj
os.remove(prefix + '_forward_T1.0.pt')
