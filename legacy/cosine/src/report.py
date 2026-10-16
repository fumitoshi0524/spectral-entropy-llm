"""Summary report generation."""

import json
from pathlib import Path


def generate_summary(results_dir: Path) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("PIPELINE SUMMARY")
    lines.append("=" * 80)

    for scale_dir in sorted(results_dir.iterdir()):
        if not scale_dir.is_dir():
            continue
        scale = scale_dir.name
        lines.append(f"\n── {scale} ──")

        for sub in sorted(scale_dir.iterdir()):
            if not sub.is_dir():
                continue
            for jf in sorted(sub.glob("*.json")):
                try:
                    with open(jf) as f:
                        data = json.load(f)
                    # Brief summary
                    if isinstance(data, dict):
                        keys = [k for k in data if k not in ("history", "results")]
                        lines.append(f"  {sub.name}/{jf.name}: keys={keys}")
                except Exception:
                    pass
    return "\n".join(lines)
