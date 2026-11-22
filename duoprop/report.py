"""Summarise test results in the format of Table 2 of the SpherE paper (MRR %, hard answers)."""
import glob
import json
import os

ORDER = ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up', '2in', '3in', 'inp', 'pin', 'pni']
PAPER = {  # SpherE + QTO, Table 2 of the paper (d = 256)
    'FB15k': [54.8, 29.9, 79.8, 43.9, 31.8, 65.1, 71.1, 54.5, 59.0, 52.1, 35.5, 39.4, 37.3, 27.9, 26.5, 18.5],
    'FB15k-237': [21.7, 8.3, 42.0, 10.6, 9.1, 29.9, 40.8, 16.5, 25.0, 12.1, 9.2, 9.4, 15.2, 7.0, 6.4, 3.3],
    'NELL': [24.6, 9.2, 53.6, 15.3, 13.8, 30.8, 37.6, 19.5, 24.2, 15.1, 11.2, 10.0, 13.7, 12.1, 6.1, 4.1],
}
DATASET = {'fb237': 'FB15k-237', 'fb15k': 'FB15k', 'nell': 'NELL'}


def row(name, vals, ref=None):
    cells = []
    for i, v in enumerate(vals):
        mark = '' if ref is None else ('+' if v > ref[i] + 1e-9 else ('=' if abs(v - ref[i]) < 0.05 else '-'))
        cells.append(f'{v:5.1f}{mark}')
    return f'{name:34s}' + ' '.join(f'{c:>7s}' for c in cells)


if __name__ == '__main__':
    print(f'{"":34s}' + ' '.join(f'{k:>7s}' for k in ['AVG_p', 'AVG_n'] + ORDER))
    for key, ds in DATASET.items():
        print(f'--- {ds}')
        print(row('SpherE+QTO (paper)', PAPER[ds]))
        for f in sorted(glob.glob(f'runs/results/{key}_*_test.jsonl')):
            for line in open(f):
                r = json.loads(line)
                t = r.get('test', r.get('res'))
                vals = [t['AVG_p'], t['AVG_n']] + [t[q]['MRR'] for q in ORDER]
                label = r.get('tag', os.path.basename(f))[:22] + f" {os.path.basename(r.get('cache', 'na.pt')).split('_', 2)[-1][:-3][:11]}"
                print(row(label, vals, PAPER[ds]))
