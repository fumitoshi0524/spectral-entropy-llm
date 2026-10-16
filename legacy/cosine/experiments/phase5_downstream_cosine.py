"""Phase 5 — Downstream evaluation for old-paper cosine-trained weights."""

import json, logging
from pathlib import Path

import torch
from src.model import load_model, load_tokenizer, target_modules
from src.data import build_dataloader
from src.eval_downstream import evaluate_downstream
from src.decompose import svd_truncate

logger = logging.getLogger(__name__)


def run(scale: str, cfg, output_dir: Path):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    d_model = cfg.model_spec(scale)["d_model"]
    p5c = cfg.phase5
    tasks = p5c.get("tasks", ["hellaswag", "arc_easy", "piqa"])
    fewshot = p5c.get("num_fewshot", 0)
    trunc_frac = p5c.get("trunc_rank_fraction", 0.70)
    ev = cfg.eval
    is_large = scale == "1.4b"
    bs = 1 if is_large else ev.get("batch_size", 2)

    model = load_model(scale, cfg, device=device)
    tok = load_tokenizer(scale, cfg)

    all_results = {}

    # Dense
    logger.info("Phase5 %s dense...", scale)
    all_results["dense"] = evaluate_downstream(model, tok, tasks, fewshot)

    # Naive SVD truncation at target fraction
    k = max(1, int(d_model * trunc_frac))
    backup = {}
    for n, m in target_modules(model):
        backup[n] = m.weight.data.clone()
        m.weight.data.copy_(svd_truncate(m.weight.data, k))
    logger.info("Phase5 %s naive-trunc@%d%%...", scale, int(trunc_frac * 100))
    all_results[f"naive_trunc_r{int(trunc_frac * 100)}"] = evaluate_downstream(model, tok, tasks, fewshot)
    for n, m in target_modules(model):
        m.weight.data.copy_(backup[n].to(device))

    # Phase3 variants
    art_dir = cfg.artifacts_path / scale / "phase3"
    if art_dir.exists():
        for sub in sorted(art_dir.iterdir()):
            if not sub.is_dir():
                continue
            pt = sub / "trained_weights.pt"
            if not pt.exists():
                continue
            logger.info("Phase5 %s %s...", scale, sub.name)
            trained = torch.load(pt, map_location="cpu")
            model.load_state_dict(trained, strict=False)
            all_results[f"phase3_{sub.name}"] = evaluate_downstream(model, tok, tasks, fewshot)

            # Truncated version
            backup2 = {}
            for n, m in target_modules(model):
                backup2[n] = m.weight.data.clone()
                m.weight.data.copy_(svd_truncate(m.weight.data, k))
            all_results[f"phase3_{sub.name}_trunc_r{int(trunc_frac * 100)}"] = \
                evaluate_downstream(model, tok, tasks, fewshot)
            for n, m in target_modules(model):
                m.weight.data.copy_(backup2[n].to(device))
    else:
        logger.warning("Phase5 %s: no Phase3 artifacts found at %s", scale, art_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "downstream.json", "w") as f:
        json.dump({"scale": scale, "results": all_results}, f, indent=2)
    logger.info("Phase5 %s complete -> %s", scale, output_dir)
