"""Convert the link predictor of CKGC (Xiao & Cao, 2024; KGC/main.py, ComplEx with entity/relation embeddings [re | im])
into the raw kbc state_dict read by duoprop.scorer.ComplExScorer (embeddings.0 = entities, embeddings.1 = relations,
same [re | im] layout and the same score Re(<e_h, w_r, conj(e_t)>)).
python tools/ckgc2kbc.py CKGC_KGC_EPOCH_BEST OUT"""
import os
import sys

import torch

src, out = sys.argv[1:3]
st = torch.load(src, map_location='cpu', weights_only=False)['model']
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({'embeddings.0.weight': st['entity.weight'], 'embeddings.1.weight': st['relation.weight']}, out)
print('saved', out, tuple(st['entity.weight'].shape), tuple(st['relation.weight'].shape))
