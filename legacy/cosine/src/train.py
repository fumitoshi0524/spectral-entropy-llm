"""Training loops — LM-only, spectral entropy, L2, dropout regularized."""

import gc, logging, random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import target_modules
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Reproducibility: seed torch, numpy, and python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ══════════════════════════════════════════════════════════
# GaLore-style SVD cache
# ══════════════════════════════════════════════════════════

@dataclass
class SVDCache:
    U: torch.Tensor
    Vt: torch.Tensor

    @classmethod
    @torch.no_grad()
    def build(cls, W):
        # SVD in fp32 for numerical stability, then downcast to the weight dtype
        # (e.g. bfloat16 for 1.4B) to keep the cache small on GPU.
        W_fp32 = W.float()
        orig_dtype = W.dtype
        try:
            U, _, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
        except torch.linalg.LinAlgError:
            # Degenerate spectrum after aggressive spectral training: add tiny
            # isotropic jitter to break numerical ties and retry.
            eps = 1e-7 * W_fp32.norm()
            U, _, Vt = torch.linalg.svd(
                W_fp32 + eps * torch.randn_like(W_fp32), full_matrices=False
            )
        return cls(
            U=U.to(device=W.device, dtype=orig_dtype),
            Vt=Vt.to(device=W.device, dtype=orig_dtype),
        )

    def approx_sigma(self, W):
        # Compute cheap projection in the weight dtype, only cast the small
        # resulting vector to fp32 for the entropy log/exp arithmetic.
        sigma = torch.diag((self.U.T @ W) @ self.Vt.T)
        return sigma.float()


# ══════════════════════════════════════════════════════════
# Regularizer losses
# ══════════════════════════════════════════════════════════

def entropy_loss_layer(W, cache):
    σ = cache.approx_sigma(W)
    σ = torch.clamp(σ, min=1e-12)
    if len(σ) < 2:
        return torch.tensor(0.0, device=W.device)
    p = σ / σ.sum()
    return -(p * torch.log(p + 1e-12)).sum()


def topk_constraint_layer(W, U_k, Vt_k):
    """Fraction of Frobenius energy lying outside the dense top-k subspace.

    Projects W onto the fixed subspace spanned by the dense top-k left/right
    singular vectors and returns the normalized residual energy. Minimizing
    this keeps the trained weights aligned with the dense model's principal
    directions while spectral entropy concentrates the spectrum.

    The matrix products stay in the weight dtype (e.g. bfloat16) to save
    memory on large models; only the final scalar is cast to fp32.
    """
    # W_proj = U_k (U_k^T W Vt_k^T) Vt_k
    core = U_k.T @ W @ Vt_k.T
    W_proj = U_k @ core @ Vt_k
    residual = (W - W_proj).norm("fro") ** 2
    denom = W.norm("fro") ** 2 + 1e-8
    return (residual / denom).float()


def cosine_loss_layer_static(W, U, Vt, k):
    """Static-anchor cosine spectral penalty (old-paper formulation).

    U, Vt are the fixed SVD bases of the dense weight.  The loss is
    1 - sqrt( sum_{i<k} sigma_approx_i^2 / sum_i sigma_approx_i^2 ),
    where sigma_approx = diag(U^T W Vt^T).  This is exactly the penalty
    used in verify_cosine.py, but without periodic refresh.
    """
    sigma = torch.diag((U.T @ W) @ Vt.T)
    s2 = torch.clamp(sigma, min=0).pow(2)
    return 1.0 - torch.sqrt(s2[:k].sum() / (s2.sum() + 1e-12))


# ══════════════════════════════════════════════════════════
# Training loops
# ══════════════════════════════════════════════════════════

def _eval_step(model, eval_loader, ppl_fn):
    model.eval()
    ppl = ppl_fn(model, eval_loader)
    er, sr = compute_metrics(model)
    model.train()
    return ppl, er, sr


# ══════════════════════════════════════════════════════════
# Spectral gradient flow analysis (exact, analytical)
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def _entropy_gradient_flow(model) -> dict:
    """At current W, compute ∂H/∂σ_i analytically for every target layer.

    H = −Σ p_i log p_i,  p_i = σ_i / Σσ_j
    ∂H/∂σ_k = −(1 + log p_k)/S + C   where S = Σσ_j, C = Σ_j σ_j(1+log p_j)/S²

    Returns per-layer list of {"sigma_idx": int, "sigma_val": float, "grad": float}.
    """
    per_layer = {}
    for n, m in target_modules(model):
        W = m.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        S = S[S > 1e-12]
        if len(S) < 2:
            per_layer[n] = []
            continue
        S_sum = S.sum()
        p = S / S_sum
        log_p = torch.log(p + 1e-12)
        C = float((S * (1.0 + log_p)).sum() / (S_sum ** 2 + 1e-8))
        # ∂H/∂σ_k
        grad_s = C - (1.0 + log_p) / S_sum
        per_layer[n] = [
            {"idx": i, "sigma": float(S[i]), "grad": float(grad_s[i])}
            for i in range(len(S))
        ]

    # Pool across layers: bin sigma by rank index, average gradient
    max_r = max((len(v) for v in per_layer.values()), default=0)
    if max_r == 0:
        return {"per_layer": per_layer, "pooled": []}

    # Normalize rank index to [0,1] across r
    pooled = []
    for r_idx in range(max_r):
        grads = []
        sigmas = []
        for ldata in per_layer.values():
            if r_idx < len(ldata):
                grads.append(ldata[r_idx]["grad"])
                sigmas.append(ldata[r_idx]["sigma"])
        if grads:
            pooled.append({
                "rank_frac": round(r_idx / max(max_r - 1, 1), 4),
                "grad_mean": float(np.mean(grads)),
                "grad_std": float(np.std(grads)),
                "sigma_mean": float(np.mean(sigmas)),
                "n_layers": len(grads),
            })

    return {"per_layer": per_layer, "pooled": pooled}


def train_baseline(model, train_loader, eval_loader,
                   steps, lr, grad_accum, eval_every, weight_decay, seed, ppl_fn,
                   use_gc: bool = False):
    """LM-only training with AdamW weight decay."""
    set_seed(seed)
    if use_gc and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = iter(train_loader)
    log = []

    ppl, er, sr = _eval_step(model, eval_loader, ppl_fn)
    log.append({"step": 0, "ppl": round(ppl, 2), "er": round(er, 1), "sr": round(sr, 1)})
    logger.info("  [%s] step=0 PPL=%.2f ER=%.1f SR=%.1f", "baseline", ppl, er, sr)

    for step in range(1, steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(model.device)
        am = batch["attention_mask"].to(model.device)
        labels = ids.clone(); labels[am == 0] = -100
        out = model(input_ids=ids, attention_mask=am, labels=labels)
        out.loss.backward()

        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()

        if step % eval_every == 0:
            ppl, er, sr = _eval_step(model, eval_loader, ppl_fn)
            log.append({"step": step, "ppl": round(ppl, 2), "er": round(er, 1), "sr": round(sr, 1)})
            logger.info("  [%s] step=%d PPL=%.2f ER=%.1f SR=%.1f", "baseline", step, ppl, er, sr)

    del opt; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return log


def train_spectral(model, train_loader, eval_loader,
                   method, strength,
                   steps, lr, grad_accum, eval_every,
                   svd_interval, weight_decay, seed, ppl_fn,
                   use_gc: bool = False):
    """Spectral entropy regularized training."""
    set_seed(seed)
    if use_gc and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = iter(train_loader)
    loss_fn = {"entropy": entropy_loss_layer}[method]
    log = []

    def _refresh():
        return {n: SVDCache.build(m.weight.data) for n, m in target_modules(model)}

    caches = _refresh()

    # ---- step 0 ----
    def _log_eval(step):
        """Eval + optional gradient flow for entropy method."""
        model.eval()
        ppl = ppl_fn(model, eval_loader)
        er, sr = compute_metrics(model)
        entry = {"step": step, "ppl": round(ppl, 2), "er": round(er, 1), "sr": round(sr, 1)}
        if method == "entropy":
            gf = _entropy_gradient_flow(model)
            entry["grad_flow_pooled"] = gf["pooled"]   # per-rank-frac gradient profile
        model.train()
        return entry

    log.append(_log_eval(0))
    logger.info("  [%s] step=0 PPL=%.2f ER=%.1f SR=%.1f", method,
                log[-1]["ppl"], log[-1]["er"], log[-1]["sr"])

    for step in range(1, steps + 1):
        if step % svd_interval == 0:
            caches = _refresh()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(model.device)
        am = batch["attention_mask"].to(model.device)
        labels = ids.clone(); labels[am == 0] = -100
        out = model(input_ids=ids, attention_mask=am, labels=labels)
        lm_loss = out.loss

        reg_loss = torch.tensor(0.0, device=model.device)
        if strength > 0:
            for n, m in target_modules(model):
                c = caches.get(n)
                if c is not None:
                    reg_loss = reg_loss + loss_fn(m.weight, c)

        (lm_loss + strength * reg_loss).backward()

        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()

        if step % eval_every == 0:
            log.append(_log_eval(step))
            e = log[-1]
            logger.info("  [%s] step=%d PPL=%.2f ER=%.1f SR=%.1f",
                        method, e["step"], e["ppl"], e["er"], e["sr"])

    del opt; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return log


def train_spectral_topk(model, train_loader, eval_loader,
                        strength, mu, topk_fraction,
                        steps, lr, grad_accum, eval_every,
                        svd_interval, weight_decay, seed, ppl_fn,
                        use_gc: bool = False):
    """Spectral entropy + dense top-k subspace constraint.

    Minimizes spectral entropy (to lower effective rank) while penalizing
    energy outside the *dense* model's top-k singular subspace. The dense
    top-k bases (U_k, Vt_k) are computed once at the start and fixed.
    """
    set_seed(seed)
    if use_gc and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = iter(train_loader)
    log = []

    # Cache dense top-k subspace bases once.  Keep them in the model dtype
    # (e.g. bfloat16 for 1.4B) to avoid blowing up GPU memory.
    dense_topk = {}
    for n, m in target_modules(model):
        W = m.weight.data
        W_fp32 = W.float()
        k = max(1, int(min(W.shape) * topk_fraction))
        U, _, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
        dense_topk[n] = (U[:, :k].to(device=W.device, dtype=W.dtype),
                         Vt[:k, :].to(device=W.device, dtype=W.dtype))

    def _refresh():
        return {n: SVDCache.build(m.weight.data) for n, m in target_modules(model)}

    caches = _refresh()

    # Step 0 eval
    model.eval()
    ppl = ppl_fn(model, eval_loader)
    er, sr = compute_metrics(model)
    model.train()
    log.append({"step": 0, "ppl": round(ppl, 2), "er": round(er, 1), "sr": round(sr, 1)})
    logger.info("  [entropy+topk] step=0 PPL=%.2f ER=%.1f SR=%.1f", ppl, er, sr)

    for step in range(1, steps + 1):
        if step % svd_interval == 0:
            caches = _refresh()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(model.device)
        am = batch["attention_mask"].to(model.device)
        labels = ids.clone(); labels[am == 0] = -100
        out = model(input_ids=ids, attention_mask=am, labels=labels)
        lm_loss = out.loss

        reg_loss = torch.tensor(0.0, device=model.device)
        topk_loss = torch.tensor(0.0, device=model.device)
        for n, m in target_modules(model):
            c = caches.get(n)
            if c is not None:
                reg_loss = reg_loss + entropy_loss_layer(m.weight, c)
            U_k, Vt_k = dense_topk[n]
            topk_loss = topk_loss + topk_constraint_layer(m.weight, U_k, Vt_k)

        (lm_loss + strength * reg_loss + mu * topk_loss).backward()

        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()

        if step % eval_every == 0:
            model.eval()
            ppl = ppl_fn(model, eval_loader)
            er, sr = compute_metrics(model)
            model.train()
            log.append({"step": step, "ppl": round(ppl, 2), "er": round(er, 1), "sr": round(sr, 1)})
            logger.info("  [entropy+topk] step=%d PPL=%.2f ER=%.1f SR=%.1f",
                        step, ppl, er, sr)

    del opt; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return log


def train_fisher_cosine(model, train_loader, eval_loader,
                        anchors, mu,
                        steps, lr, grad_accum, eval_every, weight_decay, seed, ppl_fn,
                        use_gc: bool = False):
    """Fisher-guided adaptive cosine training.

    Fixed per-layer dense anchors (U_k, Vt_k) are pre-computed from Fisher
    sensitivity.  Training minimizes 1 - cos(W, anchor_subspace) for each
    target layer while continuing LM fine-tuning.
    """
    from .fisher import cosine_anchor_loss

    set_seed(seed)
    if use_gc and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = iter(train_loader)
    log = []

    ppl, er, sr = _eval_step(model, eval_loader, ppl_fn)
    log.append({"step": 0, "ppl": round(ppl, 2), "er": round(er, 1), "sr": round(sr, 1)})
    logger.info("  [fisher-cos] step=0 PPL=%.2f ER=%.1f SR=%.1f", ppl, er, sr)

    for step in range(1, steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(model.device)
        am = batch["attention_mask"].to(model.device)
        labels = ids.clone(); labels[am == 0] = -100
        out = model(input_ids=ids, attention_mask=am, labels=labels)
        lm_loss = out.loss

        reg_loss = torch.tensor(0.0, device=model.device)
        for n, m in target_modules(model):
            U_k, Vt_k, _ = anchors[n]
            reg_loss = reg_loss + cosine_anchor_loss(m.weight, U_k, Vt_k)

        (lm_loss + mu * reg_loss).backward()

        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()

        if step % eval_every == 0:
            ppl, er, sr = _eval_step(model, eval_loader, ppl_fn)
            log.append({"step": step, "ppl": round(ppl, 2), "er": round(er, 1), "sr": round(sr, 1)})
            logger.info("  [fisher-cos] step=%d PPL=%.2f ER=%.1f SR=%.1f",
                        step, ppl, er, sr)

    del opt; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return log


def train_cosine_static(model, train_loader, eval_loader,
                        strength, k_fraction,
                        steps, lr, grad_accum, eval_every, weight_decay, seed, ppl_fn,
                        use_gc: bool = False):
    """Static-anchor cosine spectral penalty (old-paper phase 3).

    The dense top-k subspace (U_k, Vt_k) is computed once and frozen.
    The regularizer is 1 - cos(W, W_k) where W_k is the projection of W
    onto the dense top-k subspace.
    """
    set_seed(seed)
    if use_gc and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = iter(train_loader)
    log = []

    # Cache dense SVD bases once.  We need the FULL U/Vt to compute the
    # cosine ratio; the top-k truncation is applied inside the loss.
    dense_topk = {}
    for n, m in target_modules(model):
        W_param = m.weight.data
        W_fp32 = W_param.float()
        k = max(1, int(min(W_fp32.shape) * k_fraction))
        U, _, Vt = torch.linalg.svd(W_fp32, full_matrices=False)
        dense_topk[n] = (U.to(device=W_param.device, dtype=W_param.dtype),
                         Vt.to(device=W_param.device, dtype=W_param.dtype),
                         k)

    ppl, er, sr = _eval_step(model, eval_loader, ppl_fn)
    log.append({"step": 0, "ppl": round(ppl, 2), "er": round(er, 1), "sr": round(sr, 1)})
    logger.info("  [cosine] step=0 PPL=%.2f ER=%.1f SR=%.1f", ppl, er, sr)

    for step in range(1, steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(model.device)
        am = batch["attention_mask"].to(model.device)
        labels = ids.clone(); labels[am == 0] = -100
        out = model(input_ids=ids, attention_mask=am, labels=labels)
        lm_loss = out.loss

        reg_loss = torch.tensor(0.0, device=model.device)
        for n, m in target_modules(model):
            U_k, Vt_k, k = dense_topk[n]
            reg_loss = reg_loss + cosine_loss_layer_static(m.weight, U_k, Vt_k, k)

        (lm_loss + strength * reg_loss).backward()

        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()

        if step % eval_every == 0:
            ppl, er, sr = _eval_step(model, eval_loader, ppl_fn)
            log.append({"step": step, "ppl": round(ppl, 2), "er": round(er, 1), "sr": round(sr, 1)})
            logger.info("  [cosine] step=%d PPL=%.2f ER=%.1f SR=%.1f",
                        step, ppl, er, sr)

    del opt; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return log


def train_dropout(model, train_loader, eval_loader,
                  dropout_rate, steps, lr, grad_accum, eval_every, weight_decay, seed, ppl_fn):
    """Training with dropout on hidden layers (no spectral intervention)."""
    set_seed(seed)
    model.train()
    # Patch all dropout layers
    old_ps = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Dropout):
            old_ps[n] = m.p
            m.p = dropout_rate

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = iter(train_loader)
    log = []

    ppl, er, sr = _eval_step(model, eval_loader, ppl_fn)
    log.append({"step": 0, "ppl": round(ppl, 2), "er": round(er, 1), "sr": round(sr, 1)})
    logger.info("  [dropout] step=0 PPL=%.2f ER=%.1f SR=%.1f", ppl, er, sr)

    for step in range(1, steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(model.device)
        am = batch["attention_mask"].to(model.device)
        labels = ids.clone(); labels[am == 0] = -100
        out = model(input_ids=ids, attention_mask=am, labels=labels)
        out.loss.backward()

        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()

        if step % eval_every == 0:
            ppl, er, sr = _eval_step(model, eval_loader, ppl_fn)
            log.append({"step": step, "ppl": round(ppl, 2), "er": round(er, 1), "sr": round(sr, 1)})
            logger.info("  [dropout] step=%d PPL=%.2f ER=%.1f SR=%.1f", step, ppl, er, sr)

    # Restore
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Dropout) and n in old_ps:
            m.p = old_ps[n]

    del opt; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return log
