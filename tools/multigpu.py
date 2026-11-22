"""Spread the truth matrices of one run over all visible GPUs (for matrices larger than one GPU).
Each relation matrix lives on one GPU (greedy balance by number of entries); a projection moves the rows of the input
set to that GPU and the gathered entries back, so the reasoner, Prop and the evaluation are unchanged.
Use through tools/mg_run.py, which applies the patch before the module or script is executed."""
import torch

import duoprop.reason as R


class CSRx(R.CSR):
    def gather(self, rows):
        d = self.col.device
        if rows.device == d:
            return super().gather(rows)
        pos, col, val = R.CSR.gather(self, rows.to(d))
        return pos.to(rows.device), col.to(rows.device), val.to(rows.device)


_load = R.load_cache


def load_cache(path, n, device, *a, **kw):
    mats = _load(path, n, 'cpu', *a, **kw)
    ndev = torch.cuda.device_count()
    load = [0] * ndev
    out = []
    for M in mats:
        d = min(range(ndev), key=lambda i: load[i])
        load[d] += M.col.numel()
        dev = f'cuda:{d}'
        out.append(CSRx(M.crow.to(dev), M.col.to(dev), M.val.to(dev), n))
    print('multigpu: entries per GPU', load, flush=True)
    return out


R.load_cache = load_cache
