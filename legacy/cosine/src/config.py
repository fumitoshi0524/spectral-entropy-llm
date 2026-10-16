"""YAML config loader."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    scales: list[str] = field(default_factory=lambda: ["160m","410m","1.4b"])
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "./results"
    artifacts_dir: str = "./artifacts"
    force: bool = False

    # per-experiment overrides
    e1: dict = field(default_factory=dict)
    e2: dict = field(default_factory=dict)
    e3: dict = field(default_factory=dict)
    e4: dict = field(default_factory=dict)
    e6: dict = field(default_factory=dict)
    phase3: dict = field(default_factory=dict)
    phase4: dict = field(default_factory=dict)
    phase5: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    target_suffixes: tuple = ()

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    @property
    def target_weights(self) -> tuple:
        return tuple(self.target_suffixes)

    def model_spec(self, scale: str) -> dict:
        return self.models.get(scale, {})


def load_config(path: str = "config.yaml", overrides: dict | None = None) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    pipe = raw.get("pipeline", {})
    cfg = Config(
        scales=pipe.get("scales", ["160m","410m","1.4b"]),
        device=pipe.get("device", "cuda"),
        seed=pipe.get("seed", 42),
        output_dir=pipe.get("output_dir", "./results"),
        artifacts_dir=pipe.get("artifacts_dir", "./artifacts"),
        force=pipe.get("force", False),
        e1=raw.get("e1", {}),
        e2=raw.get("e2", {}),
        e3=raw.get("e3", {}),
        e4=raw.get("e4", {}),
        e6=raw.get("e6", {}),
        phase3=raw.get("phase3", {}),
        phase4=raw.get("phase4", {}),
        phase5=raw.get("phase5", {}),
        eval=raw.get("eval", {}),
        models=raw.get("models", {}),
        target_suffixes=tuple(raw.get("target_suffixes", [])),
    )

    if overrides:
        for k, v in overrides.items():
            parts = k.split(".")
            obj = cfg.__dict__
            for p in parts[:-1]:
                obj = obj.setdefault(p, {})
            obj[parts[-1]] = v

    return cfg
