"""Artifact provenance: content hashes + a manifest for every train/unlearn/
mutate/probe output. Nebius job disks are ephemeral, so each artifact must be
self-describing enough to trace back to its inputs.
"""

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def hash_bytes(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()


def hash_obj(obj):
    """Stable hash of a JSON-serializable object (sorted keys)."""
    return hash_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def hash_file(path):
    path = Path(path)
    if not path.exists():
        return None
    if path.is_dir():  # hash the sorted (relpath, filehash) listing of a checkpoint dir
        parts = []
        for p in sorted(path.rglob("*")):
            if p.is_file():
                parts.append((str(p.relative_to(path)), hash_file(p)))
        return hash_obj(parts)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def hardware():
    info = {"platform": platform.platform()}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        else:
            info["gpu"] = "cpu"
    except ImportError:
        info["gpu"] = "unknown"
    return info


def make_manifest(stage, *, seed, config=None, dataset_path=None,
                  checkpoint=None, parent=None, start=None, end=None, outputs=None):
    """Assemble a provenance manifest. Timestamps are passed in (scripts cannot
    call the wall clock deterministically); pass None to omit."""
    m = {
        "stage": stage,
        "seed": seed,
        "parent_artifact": parent,
        "git_commit": git_commit(),
        "container_image_digest": os.environ.get("DELETEBENCH_IMAGE_DIGEST", "n/a"),
        "config_hash": hash_obj(config) if config is not None else None,
        "dataset_hash": hash_file(dataset_path) if dataset_path else None,
        "checkpoint_hash": hash_file(checkpoint) if checkpoint else None,
        "hardware": hardware(),
        "start": start,
        "end": end,
        "runtime_seconds": (round(end - start, 1) if start is not None and end is not None else None),
        "outputs": outputs or [],
    }
    return m
