"""YAML config loading. Every experiment is driven by a file in configs/."""

from pathlib import Path

import yaml


def load_config(path):
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    if "seed" not in cfg:
        raise ValueError(f"{path} must define a 'seed'")
    return cfg
