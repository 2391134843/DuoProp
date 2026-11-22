"""Standardised entity representations of a link predictor (as used by the embedding view of Prop) for a list of entity
ids, for the answer-set visualisation.  python tools/viz_dump.py CKPT IDS_JSON OUT_JSON"""
import json
import sys

from duoprop.scorer import make_scorer

ckpt, ids_path, out = sys.argv[1:4]
ids = json.load(open(ids_path))
X = make_scorer(ckpt, 'cuda').features().float()
X = (X - X.mean(0)) / (X.std(0) + 1e-6)
X = X / X.norm(dim=1, keepdim=True)
json.dump({str(i): [round(v, 4) for v in X[i].tolist()] for i in ids}, open(out, 'w'))
print(len(ids), 'entities,', X.shape[1], 'dims')
