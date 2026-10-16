"""SVD, ER, SR — exact via svdvals, no rounding."""

import numpy as np
import torch
from .model import target_modules


@torch.no_grad()
def compute_metrics(model):
    er_vals, sr_vals = [], []
    for _, m in target_modules(model):
        s = torch.linalg.svdvals(m.weight.data.float())
        s = s[s > 1e-12]
        if len(s) >= 2:
            p = s / s.sum()
            er_vals.append(float(torch.exp(-(p * torch.log(p + 1e-12)).sum())))
        else:
            er_vals.append(float(len(s)))
        sr_vals.append(float((torch.norm(m.weight, 'fro') ** 2) / (s[0] ** 2 + 1e-12)))
    return np.mean(er_vals), np.mean(sr_vals)


def spectral_er(W):
    s = torch.linalg.svdvals(W.float())
    s = s[s > 1e-12]
    if len(s) < 2:
        return float(len(s))
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def spectral_sr(W):
    s = torch.linalg.svdvals(W.float())
    return float((torch.norm(W, 'fro') ** 2) / (s[0] ** 2 + 1e-12))
