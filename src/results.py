"""JSON results logging: one file per named result under results/."""

import json
import platform
import sys
from pathlib import Path


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers"):
        try:
            m = __import__(mod)
            info[mod] = m.__version__
        except ImportError:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def save_result(name, data, config=None, results_dir="results"):
    """Write results/<name>.json bundling the payload with its config and env."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"
    payload = {"name": name, "config": config, "env": env_info(), "data": data}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[results] wrote {path}")
    return path
