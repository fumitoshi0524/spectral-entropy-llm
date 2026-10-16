"""Phase 3 — Old-paper static-anchor cosine spectral training.

Mirrors the original cosine-penalty experiments:
  * fixed dense top-k anchor (k = min(W.shape) * k_fraction)
  * lambda sweep over {10, 20, 50}
  * short QTA schedule (2k steps by default)
  * no periodic SVD refresh (static anchor)
"""

import json, logging
from pathlib import Path

import torch
from src.train import set_seed, train_baseline, train_cosine_static
from src.model import load_model, load_tokenizer
from src.data import build_dataloader
from src.eval_ppl import evaluate_ppl
from src.metrics import compute_metrics

logger = logging.getLogger(__name__)


def ppl_fn(model, loader):
    return evaluate_ppl(model, loader, 40)


def run(scale: str, cfg, output_dir: Path):
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    p3c = cfg.phase3
    strengths = p3c.get("strengths", [10.0, 20.0, 50.0])
    k_fraction = p3c.get("k_fraction", 0.5)
    steps = p3c.get("steps", 2000)
    lr_cfg = p3c.get("lr", {})
    if isinstance(lr_cfg, dict):
        lr = lr_cfg.get(scale, 5e-5)
    else:
        lr = lr_cfg
    ev = cfg.eval
    baseline_lr_cfg = p3c.get("baseline_lr", {})
    if isinstance(baseline_lr_cfg, dict):
        baseline_lr = baseline_lr_cfg.get(scale, 1e-5)
    else:
        baseline_lr = baseline_lr_cfg
    ga = p3c.get("grad_accum", 4)
    ee = p3c.get("eval_every", 500)
    wd = p3c.get("weight_decay", 0.01)
    train_samples = p3c.get("train_samples", ev.get("train_samples", 50000))
    seed = cfg.seed

    is_large = scale == "1.4b"
    bs = 1 if is_large else ev.get("batch_size", 2)
    use_gc = is_large

    logger.info("Phase3 %s: loading model...", scale)
    model = load_model(scale, cfg, device=device)
    tok = load_tokenizer(scale, cfg)
    eval_loader = build_dataloader(tok, "validation", ev.get("eval_samples", 200),
                                   ev.get("max_seq_length", 512), bs, seed)
    train_loader = build_dataloader(tok, "train", train_samples,
                                    ev.get("max_seq_length", 512), bs, seed)

    ppl_dense = evaluate_ppl(model, eval_loader, 40)
    er_dense, sr_dense = compute_metrics(model)
    dense_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    logger.info("Phase3 %s dense: PPL=%.2f ER=%.1f SR=%.1f",
                scale, ppl_dense, er_dense, sr_dense)

    # Baseline (LM-only QTA) with its own LR to avoid divergence.
    logger.info("Phase3 %s baseline (lr=%.2e)...", scale, baseline_lr)
    model.load_state_dict(dense_state)
    base_log = train_baseline(model, train_loader, eval_loader,
                              steps, baseline_lr, ga, ee, wd, seed, ppl_fn, use_gc)
    base_dir = output_dir / "baseline"
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(base_dir / "result.json", "w") as f:
        json.dump({"log": base_log, "ppl_final": base_log[-1]["ppl"],
                   "er_final": base_log[-1]["er"], "sr_final": base_log[-1]["sr"]},
                  f, indent=2)

    artifacts_base = cfg.artifacts_path / scale / "phase3"
    artifacts_base.mkdir(parents=True, exist_ok=True)

    for lam in strengths:
        label = f"lam{int(lam)}"
        lam_dir = output_dir / label
        lam_dir.mkdir(parents=True, exist_ok=True)
        art_dir = artifacts_base / label
        art_dir.mkdir(parents=True, exist_ok=True)

        already_done = (lam_dir / "result.json").exists() and \
                       (art_dir / "trained_weights.pt").exists()
        if already_done and not cfg.force:
            logger.info("Phase3 %s %s already done. Skip.", scale, label)
            continue

        logger.info("Phase3 %s %s starting (lr=%.2e, k_fraction=%.2f)...",
                    scale, label, lr, k_fraction)
        model.load_state_dict(dense_state)
        slog = train_cosine_static(model, train_loader, eval_loader,
                                   lam, k_fraction,
                                   steps, lr, ga, ee, wd, seed, ppl_fn, use_gc)

        torch.save({k: v.clone().cpu() for k, v in model.state_dict().items()},
                   art_dir / "trained_weights.pt")
        with open(lam_dir / "result.json", "w") as f:
            json.dump({"lam": lam, "k_fraction": k_fraction, "lr": lr, "log": slog,
                       "ppl_final": slog[-1]["ppl"],
                       "er_final": slog[-1]["er"],
                       "sr_final": slog[-1]["sr"]}, f, indent=2)
        logger.info("Phase3 %s %s done: PPL=%.1f ER=%.1f SR=%.1f",
                    scale, label, slog[-1]["ppl"], slog[-1]["er"], slog[-1]["sr"])

    logger.info("Phase3 %s complete -> %s", scale, output_dir)
