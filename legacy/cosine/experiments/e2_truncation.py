"""E2 — Baseline SVD truncation (naive + SVD-LLM)."""

import json, logging
from pathlib import Path

import torch
from src.model import load_model, load_tokenizer, WeightBackup, target_modules
from src.data import build_dataloader
from src.eval_ppl import evaluate_ppl
from src.decompose import (
    svd_truncate, truncate_all, restore_all,
    collect_activation_covariance, svdllm_truncate,
)

logger = logging.getLogger(__name__)


def run(scale: str, cfg, output_dir: Path):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    d_model = cfg.model_spec(scale)["d_model"]
    rank_fractions = cfg.e2.get("rank_fractions", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    svdllm_batches = cfg.e2.get("svdllm_calibration_batches", 40)
    ev = cfg.eval
    max_batches = ev.get("max_eval_batches", 40)

    logger.info("E2 %s: loading model...", scale)
    model = load_model(scale, cfg, device=device)
    tok = load_tokenizer(scale, cfg)
    bs = 1 if scale == "1.4b" else ev.get("batch_size", 2)

    eval_loader = build_dataloader(tok, "validation",
                                   ev.get("eval_samples", 200),
                                   ev.get("max_seq_length", 512), bs)
    dense_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    ppl_dense = evaluate_ppl(model, eval_loader, max_batches)
    logger.info("E2 %s dense PPL=%.2f", scale, ppl_dense)

    # ── Naive SVD ──
    naive_results = []
    for frac in rank_fractions:
        k = max(1, int(d_model * frac))
        model.load_state_dict(dense_state)
        backup = truncate_all(model, k)
        ppl = evaluate_ppl(model, eval_loader, max_batches)
        restore_all(model, backup)
        naive_results.append({"rank": k, "fraction": frac, "ppl": round(ppl, 2),
                              "ratio": round(ppl / ppl_dense, 2)})
        logger.info("  naive rank=%d (%.0f%%): PPL=%.1f", k, frac * 100, ppl)

    # ── SVD-LLM ──
    model.load_state_dict(dense_state)
    train_loader = build_dataloader(tok, "train",
                                    ev.get("train_samples", 2000),
                                    ev.get("max_seq_length", 512), bs)
    covs = collect_activation_covariance(model, train_loader, device, svdllm_batches)

    svdllm_results = []
    for frac in rank_fractions:
        k = max(1, int(d_model * frac))
        model.load_state_dict(dense_state)
        backup = {}
        for n, m in target_modules(model):
            backup[n] = m.weight.data.clone()
            c = covs.get(n)
            if c is not None:
                m.weight.data.copy_(svdllm_truncate(m.weight.data, k, c))
            else:
                m.weight.data.copy_(svd_truncate(m.weight.data, k))
        ppl = evaluate_ppl(model, eval_loader, max_batches)
        for n, m in target_modules(model):
            m.weight.data.copy_(backup[n].to(device))
        svdllm_results.append({"rank": k, "fraction": frac, "ppl": round(ppl, 2),
                                "ratio": round(ppl / ppl_dense, 2)})
        logger.info("  svdllm rank=%d (%.0f%%): PPL=%.1f", k, frac * 100, ppl)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "truncation.json", "w") as f:
        json.dump({"scale": scale, "dense_ppl": ppl_dense,
                   "naive": naive_results, "svdllm": svdllm_results}, f, indent=2)
    logger.info("E2 %s complete → %s", scale, output_dir)
