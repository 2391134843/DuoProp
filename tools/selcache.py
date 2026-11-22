"""Print the matrix used by the selected setting of a finished selection: python tools/selcache.py SEL_NAME"""
import json
import sys

print(json.loads(open(f'runs/results/{sys.argv[1]}_coll_test.jsonl').read().strip().split('\n')[-1])['cache'])
