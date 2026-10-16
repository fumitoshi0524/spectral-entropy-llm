#!/usr/bin/env python3
"""Old-paper cosine pipeline.

Reuses E1 (longitudinal) and E2 (truncation baselines) and runs the original
cosine-penalty experiments:

  Phase 3: static-anchor cosine spectral training (lambda sweep)
  Phase 4: post-cosine SVD truncation
  Phase 5: downstream evaluation

Usage:
  python pipeline.py --exp all --scales 160m 410m 1.4b
  python pipeline.py --exp e1 e2 phase3 phase4 phase5 --scales 160m
  python pipeline.py --exp phase3 --scales 160m --override phase3.lr.160m=1.0e-4
  python pipeline.py --dry-run --exp all
"""

import argparse, logging, sys, time
from pathlib import Path

from src.train import set_seed
from src.config import load_config
from experiments import (
    e1_longitudinal, e2_truncation,
    phase3_cosine, phase4_post_cosine, phase5_downstream_cosine,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("pipeline")
for lib in ("huggingface_hub","datasets","httpx","httpcore","urllib3"):
    logging.getLogger(lib).setLevel(logging.WARNING)

EXP_MAP = {
    "e1": e1_longitudinal,
    "e2": e2_truncation,
    "phase3": phase3_cosine,
    "phase4": phase4_post_cosine,
    "phase5": phase5_downstream_cosine,
}


def main():
    parser = argparse.ArgumentParser(description="Old-Paper Cosine Pipeline")
    parser.add_argument("--exp", type=str, nargs="+",
                        default=["all"],
                        help="Experiments to run: e1 e2 phase3 phase4 phase5 or 'all'")
    parser.add_argument("--scales", type=str, nargs="+",
                        default=["160m","410m","1.4b"],
                        help="Model scales (default: 160m 410m 1.4b)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Config file path")
    parser.add_argument("--override", type=str, nargs="*",
                        help="Override config values: key.path=value")
    parser.add_argument("--force", action="store_true", help="Force re-run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running")
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--seed", type=int, help="Override seed")
    args = parser.parse_args()

    overrides = {}
    if args.override:
        for kv in args.override:
            k, v = kv.split("=", 1)
            overrides[k] = v
    if args.force:
        overrides["pipeline.force"] = True
    if args.device:
        overrides["pipeline.device"] = args.device
    if args.seed is not None:
        overrides["pipeline.seed"] = args.seed

    cfg = load_config(args.config, overrides or None)
    set_seed(cfg.seed)

    if "all" in args.exp:
        exps = ["e1","e2","phase3","phase4","phase5"]
    else:
        exps = [e.lower() for e in args.exp]
    invalid = [e for e in exps if e not in EXP_MAP]
    if invalid:
        logger.error("Unknown experiments: %s. Valid: %s", invalid, list(EXP_MAP))
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Old-Paper Cosine Pipeline: %s", exps)
    logger.info("Scales: %s", args.scales)
    logger.info("Device: %s", cfg.device)
    logger.info("Seed: %d", cfg.seed)
    logger.info("=" * 60)

    if args.dry_run:
        for e in exps:
            for s in args.scales:
                logger.info("  [DRY-RUN] %s / %s", e, s)
        return

    t_start = time.perf_counter()

    for e in exps:
        mod = EXP_MAP[e]
        for s in args.scales:
            if s not in cfg.models:
                logger.error("Unknown scale: %s", s)
                continue

            out_dir = cfg.output_path / s / e
            marker = cfg.output_path / s / f"{e}_DONE"

            if marker.exists() and not cfg.force:
                logger.info("=== %s / %s SKIP (marker: %s) ===", e.upper(), s, marker)
                continue

            logger.info("=== %s / %s ===", e.upper(), s)
            try:
                mod.run(s, cfg, out_dir)
                marker.touch()
                logger.info("=== %s / %s DONE (marker: %s) ===", e.upper(), s, marker)
            except Exception as exc:
                logger.exception("%s %s FAILED: %s", e, s, exc)

    elapsed = time.perf_counter() - t_start
    logger.info("Pipeline done in %.1f hours.", elapsed / 3600)


if __name__ == "__main__":
    main()
