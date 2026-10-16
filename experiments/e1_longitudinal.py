"""E1 — Longitudinal spectral tracking across training checkpoints."""

import gc, json, logging, shutil, time
from pathlib import Path

import torch
from src.train import set_seed
from src.model import load_model, load_layer_weights, model_name
from src.metrics import spectral_er, spectral_sr

logger = logging.getLogger(__name__)


def run(scale: str, cfg, output_dir: Path):
    set_seed(cfg.seed)
    spec = cfg.model_spec(scale)
    num_layers = spec["num_layers"]
    name = model_name(scale, cfg)

    from src.model import WEIGHT_KEY_MAP
    from huggingface_hub import list_repo_refs

    refs = list_repo_refs(name)
    revisions = [b.name for b in refs.branches if b.name.startswith("step")]
    revisions.sort(key=lambda s: int(s.replace("step", "")))

    interval = cfg.e1.get("checkpoint_interval", 10000)
    max_ckpts = cfg.e1.get("max_checkpoints", 20)
    selected = [r for r in revisions if int(r.replace("step", "")) % interval == 0][:max_ckpts]

    logger.info("E1 %s: %d checkpoints", scale, len(selected))

    all_results = []
    for rev in selected:
        step_num = int(rev.replace("step", ""))
        logger.info("E1 %s step=%d loading...", scale, step_num)
        t0 = time.perf_counter()

        model = load_model(scale, cfg, revision=rev, device="cpu")
        weights = load_layer_weights(model, num_layers)
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        entry = {"step": step_num, "layers": {}}
        for layer_idx in range(num_layers):
            lw = {}
            for wname, w in weights.get(layer_idx, {}).items():
                lw[wname] = {
                    "er": spectral_er(w),
                    "sr": spectral_sr(w),
                }
            entry["layers"][str(layer_idx)] = lw
        all_results.append(entry)
        logger.info("E1 %s step=%d done in %.0fs", scale, step_num, time.perf_counter() - t0)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "longitudinal.json", "w") as f:
        json.dump({"scale": scale, "results": all_results}, f, indent=2)
    logger.info("E1 %s complete. %d checkpoints → %s", scale, len(all_results), output_dir)
