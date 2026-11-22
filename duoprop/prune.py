"""Delete the caches of one cache tag that its final selection(s) did not pick (keeps the disk from filling up).
Usage: python -m duoprop.prune --tag kb237_paper --keep_from K237_paper_final[,K237_paper_finalP]"""
import argparse
import glob
import json
import os

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--keep_from', required=True, help='comma-separated result names whose test records name the caches to keep')
    a = ap.parse_args()
    keep = {os.path.basename(json.loads(l)['cache']) for name in a.keep_from.split(',')
            for l in open(f'runs/results/{name}_coll_test.jsonl')}
    freed = 0
    for f in glob.glob(f'cache/mats/{a.tag}_dual_*.pt'):
        if os.path.basename(f) not in keep:
            freed += os.path.getsize(f)
            os.remove(f)
    print(f'pruned {a.tag}: kept {sorted(keep)}, freed {freed / 1e9:.1f} GB', flush=True)
