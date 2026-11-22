"""Batched triple scoring from SpherE, GQE and Query2Box checkpoints (trained by third_party/sphere/main.py) and from
ComplEx checkpoints (trained by the kbc code of QTO); make_scorer picks the model from the checkpoint."""
import collections
import os

import torch


def read_stats(data):
    with open(os.path.join(data, 'stats.txt')) as f:
        lines = f.read().split('\n')
    return int(lines[0].split(' ')[-1]), int(lines[1].split(' ')[-1])


def read_train(data):
    tails = collections.defaultdict(list)
    with open(os.path.join(data, 'train.txt')) as f:
        for line in f:
            h, r, t = map(int, line.split())
            tails[(h, r)].append(t)
    return tails


class SphereScorer:
    """Reproduces SpherE.cal_logit_sphere (clqa/src/model/sphere.py) for arbitrary head/tail blocks."""

    def __init__(self, ckpt, device, cen=0.02):
        sd = torch.load(ckpt, map_location=device, weights_only=False)['model_state_dict']
        er = sd['embedding_range'].item()
        pi = 3.14159262358979323846
        self.gamma = sd['gamma'].item()
        self.Em = sd['entity_mod_embedding']
        self.Ep = sd['entity_phase_embedding'] / (er / pi)
        self.Et = (sd['entity_radius_embedding'] / er).abs().squeeze(-1)
        self.Rm = sd['relation_mod_embedding'].abs()
        self.Rp = sd['relation_phase_embedding'] / (er / pi)
        self.Rb = sd['relation_bias_embedding']
        self.Rt = sd['relation_radius_embedding'].abs().squeeze(-1)
        self.mw, self.pw = sd['mod_weight'].item(), sd['phase_weight'].item()
        self.cen, self.dev = cen, device
        self.n = self.Em.shape[0]
        self.tau = torch.exp(sd['log_tau']) if 'log_tau' in sd else None  # learned relation-wise temperature

    @torch.no_grad()
    def scores(self, hs, r, ts=None, chunk=1024):
        """Logits of shape (len(hs), len(ts)); ts=None means all entities."""
        rb = self.Rb[r].clamp(max=1.0)
        rm = self.Rm[r]
        rb = torch.where(rb < -rm, -rm, rb)
        qm = self.Em[hs] * (rm + rb)
        qp = self.Ep[hs] + self.Rp[r]
        qt = self.Et[hs] * self.Rt[r]
        Tm = self.Em if ts is None else self.Em[ts]
        Tp = self.Ep if ts is None else self.Ep[ts]
        Tt = self.Et if ts is None else self.Et[ts]
        mod = torch.cdist(qm, Tm * (1 - rb))
        phase = torch.cat([torch.abs(torch.sin((qp.unsqueeze(1) - Tp[i:i + chunk].unsqueeze(0)) / 2)).sum(-1)
                           for i in range(0, Tp.shape[0], chunk)], dim=1)
        rad = (qt.unsqueeze(1) + Tt.unsqueeze(0)).abs()
        out = self.gamma - (self.mw * mod + self.pw * phase - self.cen * rad)
        return out if self.tau is None else out / self.tau[r]

    def features(self):
        return torch.cat([self.Em, torch.cos(self.Ep), torch.sin(self.Ep), self.Et.unsqueeze(1)], dim=1)


class VecScorer:
    """GQE: q = e_h + r, score = gamma - ||e_t - q||_1 (clqa/src/model/gqe.py)."""

    def __init__(self, ckpt, device):
        sd = torch.load(ckpt, map_location=device, weights_only=False)['model_state_dict']
        self.gamma = sd['gamma'].item()
        self.E, self.R = sd['entity_embedding'], sd['relation_embedding']
        self.n, self.dev = self.E.shape[0], device
        self.tau = torch.exp(sd['log_tau']) if 'log_tau' in sd else None

    @torch.no_grad()
    def scores(self, hs, r, ts=None, chunk=None):
        q = self.E[hs] + self.R[r]
        T = self.E if ts is None else self.E[ts]
        out = self.gamma - torch.cdist(q, T, p=1)
        return out if self.tau is None else out / self.tau[r]

    def features(self):
        return self.E


class BoxScorer:
    """Query2Box (center_mode none): c = e_h + r, o = r_offset,
    score = gamma - ||relu(|e_t - c| - o)||_1 - cen * ||min(|e_t - c|, o)||_1 (clqa/src/model/query2box.py)."""

    def __init__(self, ckpt, device, cen=0.02):
        sd = torch.load(ckpt, map_location=device, weights_only=False)['model_state_dict']
        self.gamma = sd['gamma'].item()
        self.E, self.R, self.O = sd['entity_embedding'], sd['relation_embedding'], sd['offset_embedding']
        self.cen, self.n, self.dev = cen, self.E.shape[0], device
        self.tau = torch.exp(sd['log_tau']) if 'log_tau' in sd else None

    @torch.no_grad()
    def scores(self, hs, r, ts=None, chunk=512):
        c = self.E[hs] + self.R[r]
        o = self.O[r]
        T = self.E if ts is None else self.E[ts]
        out = []
        for i in range(0, T.shape[0], chunk):
            delta = (T[i:i + chunk].unsqueeze(0) - c.unsqueeze(1)).abs()
            out.append(self.gamma - torch.relu(delta - o).sum(-1) - self.cen * torch.minimum(delta, o).abs().sum(-1))
        out = torch.cat(out, dim=1)
        return out if self.tau is None else out / self.tau[r]

    def features(self):
        return self.E


class ComplExScorer:
    """ComplEx-N3 of the kbc / ssl-relation-prediction code used by QTO (raw state_dict, embeddings = [re | im]):
    score = Re(<e_h, w_r, conj(e_t)>), i.e. models.ComplEx.forward(score_rhs=True)[0]."""

    def __init__(self, ckpt, device):
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        self.E, self.R = sd['embeddings.0.weight'], sd['embeddings.1.weight']
        self.k, self.n, self.dev = self.E.shape[1] // 2, self.E.shape[0], device
        self.tau = None

    @torch.no_grad()
    def scores(self, hs, r, ts=None, chunk=None):
        k, lhs, rel = self.k, self.E[hs], self.R[r]
        T = self.E if ts is None else self.E[ts]
        q_re = lhs[:, :k] * rel[:k] - lhs[:, k:] * rel[k:]
        q_im = lhs[:, :k] * rel[k:] + lhs[:, k:] * rel[:k]
        return q_re @ T[:, :k].t() + q_im @ T[:, k:].t()

    def features(self):
        return self.E


def make_scorer(ckpt, device):
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    if 'model_state_dict' not in sd and 'embeddings.0.weight' in sd:
        return ComplExScorer(ckpt, device)
    keys = sd['model_state_dict'].keys()
    if 'entity_mod_embedding' in keys:
        return SphereScorer(ckpt, device)
    if 'offset_embedding' in keys:
        return BoxScorer(ckpt, device)
    return VecScorer(ckpt, device)
