"""Phase 4 — Post-cosine SVD truncation (old paper)."""

import json, logging
from pathlib import Path

import torch
from src.model import load_model, load_tokenizer, target_modules
from src.data import build_dataloader
from src.eval_ppl import evaluate_ppl
from src.decompose import svd_truncate

logger = logging.getLogger(__name__)


def run(scale: str, cfg, output_dir: Path):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    d_model = cfg.model_spec(scale)["d_model"]
    p4c = cfg.phase4
    rank_fractions = p4c.get("rank_fractions", [1.0, 0.90, 0.80, 0.70, 0.60, 0.50])
    ev = cfg.eval
    max_batches = ev.get("max_eval_batches", 40)
    is_large = scale == "1.4b"
    bs = 1 if is_large else ev.get("batch_size", 2)

    art_dir = cfg.artifacts_path / scale / "phase3"
    if not art_dir.exists():
        logger.error("Phase4 %s: no Phase3 artifacts at %s. Run Phase3 first.", scale, art_dir)
        return

    tok = load_tokenizer(scale, cfg)
    eval_loader = build_dataloader(tok, "validation", ev.get("eval_samples", 200),
                                   ev.get("max_seq_length", 512), bs)

    model = load_model(scale, cfg, device=device)
    dense_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    available = []
    for sub in sorted(art_dir.iterdir()):
        if sub.is_dir() and (sub / "trained_weights.pt").exists():
            available.append(sub.name)

    all_results = {}

    # Dense naive baseline from E2 if available
    e2_file = cfg.output_path / scale / "e2" / "truncation.json"
    if e2_file.exists():
        with open(e2_file) as f:
            e2_data = json.load(f)
        all_results["baseline"] = e2_data.get("naive", [])

    for variant in available:
        logger.info("Phase4 %s %s truncating...", scale, variant)
        trained = torch.load(art_dir / variant / "trained_weights.pt", map_location="cpu")
        model.load_state_dict(trained, strict=False)

        results = []
        for frac in rank_fractions:
            k = max(1, int(d_model * frac))
            backup = {}
            for n, m in target_modules(model):
                backup[n] = m.weight.data.clone()
                m.weight.data.copy_(svd_truncate(m.weight.data, k))
            ppl = evaluate_ppl(model, eval_loader, max_batches)
            for n, m in target_modules(model):
                m.weight.data.copy_(backup[n].to(device))
            results.append({"rank": k, "fraction": frac, "ppl": round(ppl, 2)})
            logger.info("  rank=%d (%.0f%%): PPL=%.1f", k, frac * 100, ppl)
        all_results[variant] = results

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "post_trunc.json", "w") as f:
        json.dump({"scale": scale, "results": all_results}, f, indent=2)
    logger.info("Phase4 %s complete -> %s", scale, output_dir)
