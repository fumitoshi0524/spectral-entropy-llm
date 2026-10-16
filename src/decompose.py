"""SVD truncation — naive + activation-aware (SVD-LLM)."""

import torch
import torch.nn as nn
import numpy as np
import logging
from .model import target_modules

logger = logging.getLogger(__name__)


def svd_truncate(W: torch.Tensor, k: int) -> torch.Tensor:
    W32 = W.float()
    k = min(k, min(W32.shape[0], W32.shape[1]))
    U, S, Vt = torch.linalg.svd(W32, full_matrices=False)
    return (U[:, :k] * S[:k].unsqueeze(0)) @ Vt[:k, :]


def truncate_all(model, per_weight_rank: int) -> dict[str, torch.Tensor]:
    backup = {}
    for n, m in target_modules(model):
        backup[n] = m.weight.data.clone()
        m.weight.data.copy_(svd_truncate(m.weight.data, per_weight_rank))
    return backup


def restore_all(model, backup: dict):
    for n, m in target_modules(model):
        m.weight.data.copy_(backup[n].to(m.weight.device))


# ── SVD-LLM ──

@torch.no_grad()
def collect_activation_covariance(model, dataloader, device, max_batches=40):
    activations = {}

    def hook_fn(name):
        def h(module, inp, out):
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            activations.setdefault(name, []).append(x.cpu())
        return h

    hooks = []
    for n, m in target_modules(model):
        hooks.append(m.register_forward_hook(hook_fn(n)))

    model.eval()
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        model(input_ids=batch["input_ids"].to(device),
              attention_mask=batch["attention_mask"].to(device))

    for h in hooks:
        h.remove()

    covs = {}
    for name, xs in activations.items():
        X = torch.cat(xs, dim=0).float()
        X_c = X - X.mean(dim=0, keepdim=True)
        covs[name] = (X_c.T @ X_c / X.shape[0]).numpy()
    return covs


def svdllm_truncate(W: torch.Tensor, k: int, cov: np.ndarray, eps: float = 1e-6) -> torch.Tensor:
    W_np = W.float().cpu().numpy()
    if cov.shape[1] != W_np.shape[1]:
        cov = cov[:W_np.shape[1], :W_np.shape[1]]
    k = min(k, min(W_np.shape[0], W_np.shape[1]))

    cov_reg = cov + np.eye(cov.shape[0]) * eps
    try:
        L = np.linalg.cholesky(cov_reg)
    except np.linalg.LinAlgError:
        ev, evec = np.linalg.eigh(cov_reg)
        ev = np.maximum(ev, eps)
        L = evec @ np.diag(np.sqrt(ev)) @ evec.T

    W_w = W_np @ L
    U, S, Vt = np.linalg.svd(W_w, full_matrices=False)
    W_out = ((U[:, :k] * S[:k]) @ Vt[:k, :]) @ np.linalg.inv(L)
    return torch.from_numpy(W_out).to(W.dtype).to(W.device)
